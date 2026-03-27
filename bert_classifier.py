import torch
import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    """
    BERT文本分类器
    
    基于BERT-base-chinese的文本分类模型，采用参数高效微调策略：
    - 冻结前10层编码器参数
    - 仅微调最后2层编码器（第11层和第12层）
    - 添加轻量级分类头
    """
    
    def __init__(self, num_classes=5, dropout_prob=0.1, model_path='bert-base-chinese', bert_model=None):
        """
        初始化BERT分类器
        
        Args:
            num_classes: 分类类别数
            dropout_prob: Dropout丢弃概率
            model_path: BERT模型路径
            bert_model: 预加载的BERT模型（可选）
        """
        super(BertClassifier, self).__init__()
        
        if bert_model is not None:
            self.bert = bert_model
        else:
            self.bert = BertModel.from_pretrained(model_path, local_files_only=True)
        
        self.freeze_encoder_layers(num_freeze=10)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(768, num_classes)
        
    def freeze_encoder_layers(self, num_freeze=10):
        """
        冻结BERT编码器的前num_freeze层
        
        Args:
            num_freeze: 要冻结的编码器层数
        """
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        for i in range(num_freeze):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
        
        print(f"已冻结前{num_freeze}层编码器，仅微调最后{12 - num_freeze}层")
    
    def unfreeze_all_layers(self):
        """
        解冻所有BERT层（用于完整微调）
        """
        for param in self.bert.parameters():
            param.requires_grad = True
        print("已解冻所有BERT层")
    
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            分类logits [batch_size, num_classes]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_predictions(self, input_ids, attention_mask):
        """
        获取预测类别和概率
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            
        Returns:
            预测类别和概率分布
        """
        logits = self.forward(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        
        return predictions, probabilities


class BertClassifierWithConfig(BertClassifier):
    """
    带配置的BERT分类器，支持自定义冻结策略
    """
    
    def __init__(self, num_classes=5, dropout_prob=0.1, freeze_strategy='last_2'):
        """
        初始化带配置的BERT分类器
        
        Args:
            num_classes: 分类类别数
            dropout_prob: Dropout丢弃概率
            freeze_strategy: 冻结策略 ('last_2', 'last_4', 'none')
        """
        super(BertClassifierWithConfig, self).__init__(num_classes, dropout_prob)
        
        if freeze_strategy == 'last_2':
            self.freeze_encoder_layers(num_freeze=10)
        elif freeze_strategy == 'last_4':
            self.freeze_encoder_layers(num_freeze=8)
        elif freeze_strategy == 'none':
            self.unfreeze_all_layers()
        else:
            print(f"未知冻结策略: {freeze_strategy}，使用默认策略（冻结前10层）")
            self.freeze_encoder_layers(num_freeze=10)
