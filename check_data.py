import pandas as pd
import numpy as np

# 读取Excel文件
try:
    file_path = 'xiu-15.xlsx'
    df = pd.read_excel(file_path)
    
    # 显示文件基本信息
    print(f"文件 {file_path} 包含 {df.shape[0]} 行和 {df.shape[1]} 列")
    print("\n列名列表:")
    for col in df.columns:
        print(f"- {col}")
    
    print("\n前5行数据:")
    print(df.head())
    
    print("\n数据类型信息:")
    print(df.dtypes)
    
    # 检查是否包含用户指定的列
    required_columns = ['PS', 'OEFR', 'SIR', 'PP', 'TCT', 'CIV', 'CKR', 'CAFR']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"\n警告: 数据中缺少以下列: {missing_columns}")
    else:
        print("\n数据包含所有需要的列")
        
        # 显示每列的统计信息
        print("\n数据统计信息:")
        print(df[required_columns].describe())
    
except Exception as e:
    print(f"读取文件时出错: {e}")