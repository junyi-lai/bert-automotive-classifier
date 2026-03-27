import pandas as pd
import os
from sklearn.model_selection import train_test_split
import json


def load_and_label_data(file_path, category, category_id):
    """
    加载CSV文件并为数据添加标签
    
    Args:
        file_path: CSV文件路径
        category: 类别名称（字符串）
        category_id: 类别ID（整数）
        
    Returns:
        包含文本和标签的DataFrame
    """
    df = pd.read_csv(file_path, header=None)
    df.columns = ['text']
    df['category'] = category
    df['label'] = category_id
    return df


def preprocess_and_split():
    """
    主函数：加载所有数据集，打标签，并划分训练/验证/测试集
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    datasets = [
        ('智能座舱.csv', 'cockpit', 0),
        ('使用指南.csv', 'guide', 1),
        ('汽车知识.csv', 'knowledge', 2),
        ('汽车营销.csv', 'marketing', 3),
        ('其他场景.csv', 'other', 4)
    ]
    
    print("=" * 80)
    print("数据预处理和划分")
    print("=" * 80)
    
    all_data = []
    
    for filename, category, category_id in datasets:
        file_path = os.path.join(base_dir, filename)
        print(f"\n处理文件: {filename}")
        print(f"  类别: {category} (ID: {category_id})")
        
        df = load_and_label_data(file_path, category, category_id)
        all_data.append(df)
        print(f"  样本数: {len(df)}")
    
    print("\n" + "=" * 80)
    print("合并所有数据集")
    print("=" * 80)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"总样本数: {len(combined_df)}")
    
    print("\n各类别分布:")
    print(combined_df['category'].value_counts())
    
    print("\n" + "=" * 80)
    print("数据集划分 (8:1:1)")
    print("=" * 80)
    
    train_df, temp_df = train_test_split(
        combined_df,
        test_size=0.2,
        random_state=42,
        stratify=combined_df['label']
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['label']
    )
    
    print(f"训练集: {len(train_df)} 条 ({len(train_df)/len(combined_df)*100:.1f}%)")
    print(f"验证集: {len(val_df)} 条 ({len(val_df)/len(combined_df)*100:.1f}%)")
    print(f"测试集: {len(test_df)} 条 ({len(test_df)/len(combined_df)*100:.1f}%)")
    
    print("\n训练集类别分布:")
    print(train_df['category'].value_counts())
    
    print("\n验证集类别分布:")
    print(val_df['category'].value_counts())
    
    print("\n测试集类别分布:")
    print(test_df['category'].value_counts())
    
    print("\n" + "=" * 80)
    print("保存数据集")
    print("=" * 80)
    
    output_dir = os.path.join(base_dir, 'processed_data')
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False, encoding='utf-8-sig')
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False, encoding='utf-8-sig')
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False, encoding='utf-8-sig')
    
    print(f"训练集已保存: {os.path.join(output_dir, 'train.csv')}")
    print(f"验证集已保存: {os.path.join(output_dir, 'val.csv')}")
    print(f"测试集已保存: {os.path.join(output_dir, 'test.csv')}")
    
    label_mapping = {
        'cockpit': 0,
        'guide': 1,
        'knowledge': 2,
        'marketing': 3,
        'other': 4
    }
    
    with open(os.path.join(output_dir, 'label_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)
    
    print(f"标签映射已保存: {os.path.join(output_dir, 'label_mapping.json')}")
    
    print("\n" + "=" * 80)
    print("数据样本预览")
    print("=" * 80)
    print("\n训练集前3条样本:")
    for idx, row in train_df.head(3).iterrows():
        print(f"\n样本 {idx + 1}:")
        print(f"  类别: {row['category']} (ID: {row['label']})")
        print(f"  文本: {row['text'][:100]}...")
    
    print("\n" + "=" * 80)
    print("预处理完成!")
    print("=" * 80)


if __name__ == "__main__":
    preprocess_and_split()
