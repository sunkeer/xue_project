import pandas as pd

# 读取Excel文件
try:
    df = pd.read_excel('xiu-15.xlsx')
    # 显示文件基本信息
    print(f"文件包含 {df.shape[0]} 行和 {df.shape[1]} 列")
    print("\n列名列表:")
    for col in df.columns:
        print(f"- {col}")
    
    print("\n前5行数据:")
    print(df.head())
    
    print("\n数据类型信息:")
    print(df.dtypes)
    
except Exception as e:
    print(f"读取文件时出错: {e}")