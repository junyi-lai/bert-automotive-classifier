# BERT汽车问题分类器

基于BERT-base-chinese的汽车领域问题分类模型。

## 功能

将用户问题准确分类到5个类别：

- **cockpit** - 智能座舱与交互
- **guide** - 车辆使用指南
- **knowledge** - 汽车理解与知识
- **marketing** - 汽车营销
- **other** - 其他场景

## 技术方案

### 模型架构

```
BERT-base-chinese (12层编码器)
├── 冻结层 (1-10层): 参数固定
├── 微调层 (11-12层): 参数可更新
└── 分类头:
    ├── Dropout (p=0.1)
    ├── Linear (768 → 5)
    └── Softmax
```

### 核心特性

- **参数高效微调**: 仅微调最后2层编码器
- **轻量级分类头**: 单层全连接网络
- **高准确率**: 测试集准确率100%
- **本地部署**: 无需API调用，推理速度毫秒级
- **Hugging Face标准格式**: 无需自定义模型定义，易于部署

## 快速开始

### 1. 环境要求

安装核心依赖（除torch外，通用版）

```bash
pip install transformers pandas numpy scikit-learn matplotlib seaborn tqdm
```

### 2. 数据准备

将5个CSV数据集（可更换）放在项目根目录：

- 智能座舱.csv
- 使用指南.csv
- 汽车知识.csv
- 汽车营销.csv
- 其他场景.csv

**数据格式要求**：

- 第一行为数据，无标题行
- 每个文件500条样本
- UTF-8编码

### 3. 数据处理

```bash
python preprocess_data.py
```

生成文件：

- `processed_data/train.csv` - 训练集（2000条）
- `processed_data/val.csv` - 验证集（250条）
- `processed_data/test.csv` - 测试集（250条）
- `processed_data/label_mapping.json` - 标签映射

### 4. 训练模型

```bash
python train.py
```

训练参数：

- Epochs: 10
- 学习率: 2e-5
- Batch size: 16
- 随机种子: 42

输出文件：

- `checkpoints/bert-automotive-classifier/` - Hugging Face标准格式模型（包含模型权重、分词器和标签映射）
- `checkpoints/training_history.json` - 训练历史

**注意**：首次运行时，脚本会自动从HuggingFace下载BERT-base-chinese模型。训练完成后，模型以Hugging Face标准格式保存，推理和评估时无需再次下载。

### 5. 评估模型

```bash
python evaluate.py
```

生成评估报告：

- 测试集准确率
- 分类报告（精确率、召回率、F1分数）
- 混淆矩阵可视化（`checkpoints/confusion_matrix.png`）
- 错误样本分析（`checkpoints/error_analysis.csv`）

### 6. 推理预测

```bash
python inference.py
```

对单个或多个问题进行分类预测。

## 项目目录

```
bert-automotive-classifier/
├── processed_data/                 # 数据预处理结果
│   ├── train.csv                   # 训练集（2000条）
│   ├── val.csv                     # 验证集（250条）
│   ├── test.csv                    # 测试集（250条）
│   └── label_mapping.json          # 标签映射
│
├── checkpoints/                    # 训练和评估结果
│   ├── bert-automotive-classifier/ # Hugging Face标准格式模型
│   │   ├── config.json            # 模型配置
│   │   ├── model.safetensors      # 模型权重
│   │   ├── tokenizer.json         # 分词器
│   │   ├── tokenizer_config.json  # 分词器配置
│   │   └── label_mapping.json    # 标签映射
│   ├── training_history.json       # 训练历史
│   ├── confusion_matrix.png        # 混淆矩阵可视化
│   └── error_analysis.csv        # 错误样本分析
│
├── 智能座舱.csv                   # 智能座舱类问题数据（500条）
├── 使用指南.csv                   # 车辆使用指南类问题数据（500条）
├── 汽车知识.csv                   # 汽车知识类问题数据（500条）
├── 汽车营销.csv                   # 汽车营销类问题数据（500条）
├── 其他场景.csv                   # 其他场景类问题数据（500条）
│
├── preprocess_data.py             # 数据预处理脚本
├── dataset.py                     # 数据集类定义
├── bert_classifier.py             # BERT分类器模型定义
├── train.py                       # 训练脚本
├── evaluate.py                    # 评估脚本
└── inference.py                   # 推理脚本
```

## 文件说明

### 核心文件

| 文件               | 作用       | 输入           | 输出                 |
| ------------------ | ---------- | -------------- | -------------------- |
| preprocess_data.py | 数据预处理 | 5个CSV文件     | processed_data/*.csv |
| dataset.py         | 数据加载类 | CSV文件        | DataLoader           |
| bert_classifier.py | 模型定义   | BERT预训练模型 | BertClassifier实例   |
| train.py           | 模型训练   | 数据+模型      | bert-automotive-classifier/ |
| evaluate.py        | 模型评估   | bert-automotive-classifier/ | 评估报告             |
| inference.py       | 模型推理   | 文本           | 分类结果             |

### 产生的文件

#### processed_data/ - 数据预处理结果

```
processed_data/
├── train.csv          # 训练集（2000条，每类400条）
├── val.csv            # 验证集（250条，每类50条）
├── test.csv           # 测试集（250条，每类50条）
└── label_mapping.json # 标签映射：{cockpit:0, guide:1, knowledge:2, marketing:3, other:4}
```

#### checkpoints/ - 训练和评估结果

```
checkpoints/
├── bert-automotive-classifier/ # Hugging Face标准格式模型
│   ├── config.json            # 模型配置
│   ├── model.safetensors      # 模型权重
│   ├── tokenizer.json         # 分词器
│   ├── tokenizer_config.json  # 分词器配置
│   └── label_mapping.json    # 标签映射
├── training_history.json # 训练历史
│   ├── train_loss: 每epoch训练损失
│   └── val_accuracy: 每epoch验证准确率
├── confusion_matrix.png  # 混淆矩阵可视化（运行evaluate.py后生成）
└── error_analysis.csv   # 错误样本分析（运行evaluate.py后生成）
```

## 注意事项

1. **首次运行**：训练脚本会自动从HuggingFace下载BERT-base-chinese模型
2. **随机种子**：训练脚本已设置随机种子为42
3. **GPU支持**：自动检测并使用GPU（如果可用）
4. **数据格式**：CSV文件第一行必须是数据，无标题行
5. **跨平台兼容**：所有路径使用相对路径，支持Windows/Linux/Mac
6. **模型格式**：模型以Hugging Face标准格式保存，无需自定义模型定义文件

## 技术栈

- **PyTorch**: 深度学习框架
- **Transformers**: Hugging Face模型库
- **scikit-learn**: 评估指标
- **pandas**: 数据处理
- **matplotlib/seaborn**: 可视化
