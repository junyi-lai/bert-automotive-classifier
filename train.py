import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from dataset import create_data_loaders
from bert_classifier import BertClassifier




def set_seed(seed=42):
    """
    设置所有随机种子以确保结果可复现
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"已设置随机种子: {seed}")


def worker_init_fn(worker_id):
    """
    DataLoader worker初始化函数，确保每个worker的随机性一致
    
    Args:
        worker_id: worker ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Trainer:
    """
    BERT分类器训练器
    
    负责模型的训练、验证和保存
    """
    
    def __init__(self, model, train_loader, val_loader, test_loader, device, label_mapping):
        """
        初始化训练器
        
        Args:
            model: BERT分类器模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            device: 训练设备
            label_mapping: 标签映射字典
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.label_mapping = label_mapping
        self.id_to_label = {v: k for k, v in label_mapping.items()}
        
    def train_epoch(self, optimizer, scheduler):
        """
        训练一个epoch
        
        Args:
            optimizer: 优化器
            scheduler: 学习率调度器
            
        Returns:
            平均损失
        """
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc='训练')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.model.zero_grad()
            
            outputs = self.model(input_ids, attention_mask)
            
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            loss_value = loss.item()
            total_loss += loss_value
            
            progress_bar.set_postfix({'loss': f'{loss_value:.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, data_loader):
        """
        评估模型
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            准确率和预测结果
        """
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='评估'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                
                preds = torch.argmax(outputs, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(true_labels, predictions)
        
        return accuracy, predictions, true_labels
    
    def train(self, epochs, learning_rate=2e-5, warmup_steps=0, save_dir='checkpoints', tokenizer=None):
        """
        完整训练流程
        
        Args:
            epochs: 训练轮数
            learning_rate: 学习率
            warmup_steps: 预热步数
            save_dir: 模型保存目录
            tokenizer: 分词器
        """
        os.makedirs(save_dir, exist_ok=True)
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        total_steps = len(self.train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        best_val_accuracy = 0
        best_train_loss = float('inf')
        training_history = {
            'train_loss': [],
            'val_accuracy': []
        }
        
        print("=" * 80)
        print("开始训练")
        print("=" * 80)
        print(f"设备: {self.device}")
        print(f"训练轮数: {epochs}")
        print(f"学习率: {learning_rate}")
        print(f"总训练步数: {total_steps}")
        print("=" * 80)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 80)
            
            train_loss = self.train_epoch(optimizer, scheduler)
            val_accuracy, _, _ = self.evaluate(self.val_loader)
            
            training_history['train_loss'].append(train_loss)
            training_history['val_accuracy'].append(val_accuracy)
            
            print(f"训练损失: {train_loss:.4f}")
            print(f"验证准确率: {val_accuracy:.4f}")
            
            if val_accuracy > best_val_accuracy or (val_accuracy == best_val_accuracy and train_loss < best_train_loss):
                best_val_accuracy = val_accuracy
                best_train_loss = train_loss
                self.save_model(save_dir, epoch, val_accuracy, train_loss, tokenizer)
                print(f"保存最佳模型 (验证准确率: {val_accuracy:.4f}, 训练损失: {train_loss:.4f})")
        
        print("\n" + "=" * 80)
        print("训练完成!")
        print(f"最佳验证准确率: {best_val_accuracy:.4f}")
        print("=" * 80)
        
        self.save_training_history(training_history, save_dir)
        
        return training_history, best_val_accuracy
    
    def save_model(self, save_dir, epoch, accuracy, train_loss, tokenizer):
        """
        保存模型（Hugging Face 标准格式）
        
        Args:
            save_dir: 保存目录
            epoch: 当前轮数
            accuracy: 当前准确率
            train_loss: 当前训练损失
            tokenizer: 分词器
        """
        hf_dir = os.path.join(save_dir, 'bert-automotive-classifier')
        self.save_hf_format(hf_dir, tokenizer)
    
    def save_hf_format(self, save_dir, tokenizer):
        """
        保存为 Hugging Face 标准格式
        
        Args:
            save_dir: 保存目录
            tokenizer: 分词器
        """
        from transformers import BertForSequenceClassification
        
        print(f"保存 Hugging Face 标准格式到: {save_dir}")
        
        hf_model = BertForSequenceClassification.from_pretrained(
            'bert-base-chinese',
            num_labels=len(self.label_mapping),
            local_files_only=True
        )
        
        hf_model.bert.load_state_dict(self.model.bert.state_dict())
        hf_model.classifier.weight.data = self.model.classifier.weight.data.clone()
        hf_model.classifier.bias.data = self.model.classifier.bias.data.clone()
        
        os.makedirs(save_dir, exist_ok=True)
        hf_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        
        import json
        config_path = os.path.join(save_dir, 'label_mapping.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.label_mapping, f, ensure_ascii=False, indent=2)
        
        print(f"Hugging Face 格式模型已保存")
    
    def save_training_history(self, history, save_dir):
        """
        保存训练历史
        
        Args:
            history: 训练历史字典
            save_dir: 保存目录
        """
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    def load_model(self, model_path):
        """
        加载模型
        
        Args:
            model_path: 模型路径
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载模型: {model_path}")
        print(f"模型准确率: {checkpoint['accuracy']:.4f}")


def main():
    """
    主函数：训练BERT分类器
    """
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_dir = os.path.join(base_dir, 'processed_data')
    
    label_mapping_path = os.path.join(processed_data_dir, 'label_mapping.json')
    with open(label_mapping_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    
    print(f"标签映射: {label_mapping}")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', clean_up_tokenization_spaces=True, local_files_only=True)
    
    train_path = os.path.join(processed_data_dir, 'train.csv')
    val_path = os.path.join(processed_data_dir, 'val.csv')
    test_path = os.path.join(processed_data_dir, 'test.csv')
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_path, val_path, test_path, tokenizer, batch_size=16, max_length=512
    )
    
    model = BertClassifier(num_classes=5, dropout_prob=0.1, model_path='bert-base-chinese')
    
    trainer = Trainer(model, train_loader, val_loader, test_loader, device, label_mapping)
    
    training_history, best_accuracy = trainer.train(
        epochs=10,
        learning_rate=2e-5,
        warmup_steps=0,
        save_dir=os.path.join(base_dir, 'checkpoints'),
        tokenizer=tokenizer
    )
    
    print("\n" + "=" * 80)
    print("在测试集上评估最佳模型")
    print("=" * 80)
    
    test_accuracy, predictions, true_labels = trainer.evaluate(trainer.test_loader)
    
    print(f"\n测试集准确率: {test_accuracy:.4f}")
    
    print("\n分类报告:")
    target_names = [trainer.id_to_label[i] for i in range(len(trainer.id_to_label))]
    print(classification_report(true_labels, predictions, target_names=target_names))
    
    print("\n混淆矩阵:")
    cm = confusion_matrix(true_labels, predictions)
    print(cm)


if __name__ == "__main__":
    main()
