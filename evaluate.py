import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from dataset import CarQuestionDataset

os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'


class Evaluator:
    """
    BERT分类器评估器
    
    负责加载训练好的模型并进行详细评估
    """
    
    def __init__(self, model_dir, test_data_path, device):
        """
        初始化评估器
        
        Args:
            model_dir: Hugging Face 格式模型目录
            test_data_path: 测试数据路径
            device: 评估设备
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"加载模型: {model_dir}")
        
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        
        label_mapping_path = os.path.join(model_dir, 'label_mapping.json')
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            self.label_mapping = json.load(f)
        
        self.id_to_label = {v: k for k, v in self.label_mapping.items()}
        
        self.model.to(self.device)
        self.model.eval()
        
        self.test_dataset = CarQuestionDataset(test_data_path, self.tokenizer)
        self.test_loader = DataLoader(self.test_dataset, batch_size=16, shuffle=False)
        
        print(f"已加载模型: {model_dir}")
        print(f"标签映射: {self.label_mapping}")
    
    def evaluate(self):
        """
        在测试集上评估模型
        
        Returns:
            准确率、预测结果和真实标签
        """
        predictions = []
        true_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        accuracy = accuracy_score(true_labels, predictions)
        
        return accuracy, predictions, true_labels, all_probabilities
    
    def print_classification_report(self, true_labels, predictions):
        """
        打印分类报告
        
        Args:
            true_labels: 真实标签
            predictions: 预测标签
        """
        target_names = [self.id_to_label[i] for i in range(len(self.id_to_label))]
        
        print("=" * 80)
        print("分类报告")
        print("=" * 80)
        print(classification_report(true_labels, predictions, target_names=target_names))
    
    def plot_confusion_matrix(self, true_labels, predictions, save_path=None):
        """
        绘制混淆矩阵
        
        Args:
            true_labels: 真实标签
            predictions: 预测标签
            save_path: 保存路径（可选）
        """
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[self.id_to_label[i] for i in range(len(self.id_to_label))],
                    yticklabels=[self.id_to_label[i] for i in range(len(self.id_to_label))])
        plt.title('混淆矩阵', fontsize=16)
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存: {save_path}")
        
        plt.show()
    
    def analyze_errors(self, test_data_path, predictions, true_labels, output_path=None):
        """
        分析错误分类的样本
        
        Args:
            test_data_path: 测试数据路径
            predictions: 预测标签
            true_labels: 真实标签
            output_path: 输出路径（可选）
        """
        df = pd.read_csv(test_data_path)
        
        errors = []
        for idx, (pred, true) in enumerate(zip(predictions, true_labels)):
            if pred != true:
                errors.append({
                    'index': idx,
                    'text': df.iloc[idx]['text'],
                    'true_label': self.id_to_label[true],
                    'predicted_label': self.id_to_label[pred]
                })
        
        error_df = pd.DataFrame(errors)
        
        print("\n" + "=" * 80)
        print(f"错误分析 (共{len(errors)}个错误)")
        print("=" * 80)
        
        for idx, row in error_df.head(10).iterrows():
            print(f"\n错误 {idx + 1}:")
            print(f"  文本: {row['text'][:100]}...")
            print(f"  真实标签: {row['true_label']}")
            print(f"  预测标签: {row['predicted_label']}")
        
        if output_path:
            error_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"\n错误分析已保存: {output_path}")
    
    def get_class_statistics(self, true_labels, predictions):
        """
        获取每个类别的统计信息
        
        Args:
            true_labels: 真实标签
            predictions: 预测标签
            
        Returns:
            类别统计字典
        """
        class_stats = {}
        
        for label_id, label_name in self.id_to_label.items():
            true_mask = [t == label_id for t in true_labels]
            pred_mask = [p == label_id for p in predictions]
            
            true_count = sum(true_mask)
            pred_count = sum(pred_mask)
            
            correct = sum([t == label_id and p == label_id 
                          for t, p in zip(true_labels, predictions)])
            
            precision = correct / pred_count if pred_count > 0 else 0
            recall = correct / true_count if true_count > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_stats[label_name] = {
                'true_count': true_count,
                'pred_count': pred_count,
                'correct': correct,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return class_stats


def main():
    """
    主函数：评估训练好的模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'checkpoints', 'bert-automotive-classifier')
    test_data_path = os.path.join(base_dir, 'processed_data', 'test.csv')
    
    if not os.path.exists(model_dir):
        print(f"错误: 模型目录不存在: {model_dir}")
        print("请先运行 train.py 训练模型")
        return
    
    evaluator = Evaluator(model_dir, test_data_path, device)
    
    accuracy, predictions, true_labels, probabilities = evaluator.evaluate()
    
    print("\n" + "=" * 80)
    print("评估结果")
    print("=" * 80)
    print(f"测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    evaluator.print_classification_report(true_labels, predictions)
    
    class_stats = evaluator.get_class_statistics(true_labels, predictions)
    
    print("\n" + "=" * 80)
    print("各类别统计")
    print("=" * 80)
    for label_name, stats in class_stats.items():
        print(f"\n{label_name}:")
        print(f"  真实样本数: {stats['true_count']}")
        print(f"  预测样本数: {stats['pred_count']}")
        print(f"  正确预测数: {stats['correct']}")
        print(f"  精确率: {stats['precision']:.4f}")
        print(f"  召回率: {stats['recall']:.4f}")
        print(f"  F1分数: {stats['f1']:.4f}")
    
    confusion_matrix_path = os.path.join(base_dir, 'checkpoints', 'confusion_matrix.png')
    evaluator.plot_confusion_matrix(true_labels, predictions, confusion_matrix_path)
    
    error_analysis_path = os.path.join(base_dir, 'checkpoints', 'error_analysis.csv')
    evaluator.analyze_errors(test_data_path, predictions, true_labels, error_analysis_path)
    
    print("\n" + "=" * 80)
    print("评估完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
