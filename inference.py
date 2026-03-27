import os
import json
import torch
from transformers import BertForSequenceClassification, BertTokenizer


class BertRouter:
    """
    BERT汽车问题路由器
    
    用于对单个问题文本进行分类，路由到相应的处理模块
    """
    
    def __init__(self, model_dir, device=None):
        """
        初始化路由器
        
        Args:
            model_dir: Hugging Face 格式模型目录
            device: 计算设备（可选，默认自动选择）
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
        
        print(f"路由器已初始化")
        print(f"使用设备: {self.device}")
        print(f"标签映射: {self.label_mapping}")
    
    def predict(self, text, return_probabilities=False):
        """
        对单个文本进行预测
        
        Args:
            text: 输入文本
            return_probabilities: 是否返回所有类别的概率
            
        Returns:
            预测结果字典，包含类别和概率
        """
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        result = {
            'text': text,
            'predicted_category': self.id_to_label[predicted_class],
            'predicted_label': predicted_class,
            'confidence': confidence
        }
        
        if return_probabilities:
            all_probs = {}
            for label_id, label_name in self.id_to_label.items():
                all_probs[label_name] = probabilities[0][label_id].item()
            result['all_probabilities'] = all_probs
        
        return result
    
    def predict_batch(self, texts, return_probabilities=False):
        """
        对多个文本进行批量预测
        
        Args:
            texts: 文本列表
            return_probabilities: 是否返回所有类别的概率
            
        Returns:
            预测结果列表
        """
        results = []
        
        for text in texts:
            result = self.predict(text, return_probabilities)
            results.append(result)
        
        return results
    
    def print_prediction(self, result):
        """
        打印预测结果
        
        Args:
            result: 预测结果字典
        """
        print("\n" + "=" * 80)
        print("预测结果")
        print("=" * 80)
        print(f"输入文本: {result['text']}")
        print(f"预测类别: {result['predicted_category']}")
        print(f"置信度: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        
        if 'all_probabilities' in result:
            print("\n各类别概率:")
            for label_name, prob in result['all_probabilities'].items():
                print(f"  {label_name:12s}: {prob:.4f} ({prob*100:.2f}%)")
        
        print("=" * 80)


def main():
    """
    主函数：演示路由器使用
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'checkpoints', 'bert-automotive-classifier')
    
    if not os.path.exists(model_dir):
        print(f"错误: 模型目录不存在: {model_dir}")
        print("请先运行 train.py 训练模型")
        return
    
    router = BertRouter(model_dir)
    
    print("\n" + "=" * 80)
    print("BERT路由器测试")
    print("=" * 80)
    
    test_questions = [
        "仪表盘出现显示故障的原因",
        "车辆电子手刹怎么释放？",
        "什么是A级车？",
        "写一条4S店汽车促销短文案",
        "感冒了吃什么好得快？"
    ]
    
    for question in test_questions:
        result = router.predict(question, return_probabilities=True)
        router.print_prediction(result)
    
    print("\n" + "=" * 80)
    print("批量预测示例")
    print("=" * 80)
    
    batch_results = router.predict_batch(test_questions)
    
    print(f"\n批量预测结果 (共{len(batch_results)}条):")
    for i, result in enumerate(batch_results, 1):
        print(f"{i}. {result['text'][:30]:30s} -> {result['predicted_category']:12s} ({result['confidence']:.2f})")


if __name__ == "__main__":
    main()
