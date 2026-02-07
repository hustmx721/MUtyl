import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob

def process_dice_results(base_path):
    """
    处理DiCE MIA结果文件，按遗忘被试对三次随机实验取平均
    """
    all_results = []
    
    # 查找DiCE_MIA开头的CSV文件
    posible_csvs = glob.glob(os.path.join(base_path, "DiCE_MIA*.csv"))
    for csv_file in posible_csvs:
        print(f"处理文件: {csv_file}")
        
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 提取数据集和模型信息
        dataset = df['dataset'].iloc[0]
        model = df['model'].iloc[0]
        
        # 定义需要取平均的数值列
        numeric_cols = [
            'correctness_attack_acc', 'correctness_member_acc', 'correctness_non_member_acc',
            'confidence_attack_acc', 'confidence_member_acc', 'confidence_non_member_acc',
            'entropy_attack_acc', 'entropy_member_acc', 'entropy_non_member_acc',
            'modified_entropy_attack_acc', 'modified_entropy_member_acc', 'modified_entropy_non_member_acc'
        ]
        
        # 按forget_subject分组，计算平均值
        grouped = df.groupby('forget_subject')[numeric_cols].mean().reset_index()
        
        # 保留四位小数
        for col in numeric_cols:
            grouped[col] = grouped[col].round(4)
        
        # 添加数据集和模型信息
        grouped['dataset'] = dataset
        grouped['model'] = model
        grouped['paradigm'] = base_path.split('/')[-1]  # 从路径中提取范式信息
        
        # 重新排列列顺序
        cols = ['dataset', 'paradigm', 'model', 'forget_subject'] + numeric_cols
        grouped = grouped[cols]
        
        all_results.append(grouped)

    if all_results:
        # 合并所有结果
        final_df = pd.concat(all_results, ignore_index=True)
        
        # 按数据集和遗忘被试排序
        final_df = final_df.sort_values(['dataset', 'forget_subject']).reset_index(drop=True)
        
        return final_df
    else:
        print("未找到任何DiCE_MIA开头的CSV文件")
        return pd.DataFrame()

def save_summary_results(df, output_path="summary_results.csv"):
    """保存汇总结果"""
    if not df.empty:
        df.to_csv(output_path, index=False)
        print(f"结果已保存到: {output_path}")
        print(f"总计处理了 {len(df)} 行数据")
        print(f"包含 {df['dataset'].nunique()} 个数据集")
        print(f"包含 {df['model'].nunique()} 种模型")
        print(f"包含 {df['forget_subject'].nunique()} 个遗忘被试")
    else:
        print("没有数据可保存")

# 使用示例
if __name__ == "__main__":
    # 设置基础路径（根据您的实际路径修改）
    paradigms = ["001","004","MI","SSVEP","ERP"]
    for task in paradigms:
        base_path = f"/mnt/data1/tyl/MachineUnlearning/MUtyl/csv/{task}"  # 替换为您的实际路径
        
        # 处理所有结果文件
        summary_df = process_dice_results(base_path)
        
        # 保存结果
        save_summary_results(summary_df, f"/mnt/data1/tyl/MachineUnlearning/MUtyl/csv/DiCE_MIA_{task}_Summary.csv")
        
        # 显示前几行数据
        if not summary_df.empty:
            print("\n前5行汇总结果:")
            print(summary_df.head())