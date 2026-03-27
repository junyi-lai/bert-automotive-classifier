import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import random
import numpy as np


def worker_init_fn(worker_id):
    """
    DataLoader worker初始化函数，确保每个worker的随机性一致
    
    Args:
        worker_id: worker ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CarQuestionDataset(Dataset):
    """
    汽车问题分类数据集类
    
    用于加载和预处理汽车问题分类数据，支持BERT模型的输入格式
    """
    
    def __init__(self, csv_path, tokenizer, max_length=512):
        """
        初始化数据集
        
        Args:
            csv_path: CSV文件路径
            tokenizer: BERT分词器
            max_length: 最大序列长度
        """
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        """
        返回数据集大小
        
        Returns:
            数据集样本数量
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含input_ids, attention_mask和label的字典
        """
        text = str(self.data.iloc[idx]['text'])
        label = int(self.data.iloc[idx]['label'])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def create_data_loaders(train_path, val_path, test_path, tokenizer, batch_size=16, max_length=512):
    """
    创建训练、验证和测试数据加载器
    
    Args:
        train_path: 训练集CSV路径
        val_path: 验证集CSV路径
        test_path: 测试集CSV路径
        tokenizer: BERT分词器
        batch_size: 批次大小
        max_length: 最大序列长度
        
    Returns:
        训练、验证和测试数据加载器
    """
    train_dataset = CarQuestionDataset(train_path, tokenizer, max_length)
    val_dataset = CarQuestionDataset(val_path, tokenizer, max_length)
    test_dataset = CarQuestionDataset(test_path, tokenizer, max_length)
    
    generator = torch.Generator()
    generator.manual_seed(42)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        num_workers=0,
        generator=generator,
        persistent_workers=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        num_workers=0,
        persistent_workers=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        num_workers=0,
        persistent_workers=False
    )
    
    return train_loader, val_loader, test_loader
