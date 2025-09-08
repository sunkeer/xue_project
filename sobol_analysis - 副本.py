import pandas as pd
import numpy as np
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze
from SALib.plotting.bar import plot as bar_plot
import matplotlib.pyplot as plt
import logging
import sys

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sobol_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# 1. 读取并检查数据
try:
    df = pd.read_excel('xiu-151.xlsx')
    logger.info(f"成功读取数据，共 {df.shape[0]} 行，{df.shape[1]} 列")
    logger.info("列名: " + ', '.join(df.columns.tolist()))
    
    # 详细数据检查
    logger.info("\n数据前5行:\n" + str(df.head()))
    logger.info("\n数据统计描述:\n" + str(df.describe()))
    
except Exception as e:
    logger.error(f"读取文件时出错: {e}")
    raise
    print(f"成功读取数据，共 {df.shape[0]} 行，{df.shape[1]} 列")
    print("列名:", df.columns.tolist())
    
    # 详细数据检查
    print("\n数据前5行:")
    print(df.head())
    
    print("\n数据统计描述:")
    print(df.describe())
    
except Exception as e:
    print(f"读取文件时出错: {e}")
    raise

# 2. 验证目标变量
if 'IV' not in df.columns:
    print("\n错误: 未找到'钒含量'列，请检查Excel文件结构")
    print("可用列:", df.columns.tolist())
    raise ValueError("目标变量'钒含量'不存在")

# 2. 定义问题
# 假设这八个参数是PS, OEFR, SIR, PP, TCT, CIV, CKR, CAFR
# 我们需要从数据中确定每个参数的范围
# 根据工业实际定义参数范围 (示例值，需根据实际情况调整)
INDUSTRIAL_RANGES = {
    'PS': [120, 180],        # 喷煤量 kg/t
    'OEFR': [1.0, 2.0],      # 示例范围
    'SIR': [0.5, 1.5],       # 示例范围
    'PP': [100, 200],        # 示例范围
    'TCT': [1850, 1950],     # 理论燃烧温度 ℃
    'CIV': [0.5, 1.5],       # 示例范围
    'CKR': [300, 400],       # 焦比 kg/t
    'CAFR': [1.0, 3.0]       # 示例范围
}

parameters = []
for name in ['PS', 'OEFR', 'SIR', 'PP', 'TCT', 'CIV', 'CKR', 'CAFR']:
    if name in INDUSTRIAL_RANGES:
        parameters.append({'name': name, 'bounds': INDUSTRIAL_RANGES[name]})
        logger.info(f"使用工业范围 {name}: {INDUSTRIAL_RANGES[name]}")
    else:
        logger.warning(f"警告: {name} 未定义工业范围，使用数据范围")
        parameters.append({'name': name, 'bounds': [df[name].min(), df[name].max()]})

problem = {
    'num_vars': len(parameters),
    'names': [param['name'] for param in parameters],
    'bounds': [param['bounds'] for param in parameters]
}

# 3. 生成Sobol样本，样本量为1000
N = 1000  # 基本样本量
logger.info(f"生成 {N} 个Sobol样本...")
samples = sobol.sample(problem, N, calc_second_order=True)
logger.info(f"样本生成完成，形状: {samples.shape}")

# 4. 准备模型输入和输出
# 假设我们要分析的输出变量是铁水钒含量，假设列名为'钒含量'
# 我们需要从数据中构建一个预测模型
# 这里简单使用随机森林作为预测模型
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 假设目标变量是'钒含量'
if 'IV' not in df.columns:
    print("警告: 未找到'钒含量'列。请确保数据中包含目标变量。")
    # 假设最后一列是目标变量
    target_col = df.columns[-1]
    print(f"使用 {target_col} 作为目标变量")
else:
    target_col = 'IV'

X = df[[param['name'] for param in parameters]]
y = df[target_col]

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 评估模型
score = model.score(X_test, y_test)
logger.info(f"模型R²评分: {score:.4f}")

# 模型验证阈值
if score < 0.7:
    logger.warning("警告: 模型精度较低(R² < 0.7)，敏感性分析结果可能不可靠")
    logger.warning("建议: 检查数据质量、特征工程或尝试其他模型")

# 5. 使用模型预测Sobol样本的输出
logger.info("预测Sobol样本的输出...")
y_pred = model.predict(samples)
logger.info("预测完成")

# 6. 执行Sobol敏感性分析
logger.info("执行Sobol敏感性分析...")
Si = sobol_analyze.analyze(problem, y_pred, calc_second_order=True)
logger.info("分析完成")

# 7. 输出结果
# 7. 输出结果
logger.info("\n一阶敏感性指数 (S1):")
for i, param in enumerate(problem['names']):
    logger.info(f"{param}: {Si['S1'][i]:.6f} ± {Si['S1_conf'][i]:.6f}")

logger.info("\n总敏感性指数 (ST):")
for i, param in enumerate(problem['names']):
    logger.info(f"{param}: {Si['ST'][i]:.6f} ± {Si['ST_conf'][i]:.6f}")

# # 8. 绘制敏感性指数图
# plt.figure(figsize=(12, 6))

# # 使用更兼容的参数调用方式
# try:
#     # 尝试使用 'index' 参数 (新版本SALib)
#     plt.subplot(1, 2, 1)
#     bar_plot(Si, index='S1', show_confidence=True)
#     plt.title('一阶敏感性指数')
    
#     plt.subplot(1, 2, 2)
#     bar_plot(Si, index='ST', show_confidence=True)
#     plt.title('总敏感性指数')
# except TypeError as e:
#     if "unexpected keyword argument 'index'" in str(e):
#         # 回退到 'order' 参数 (旧版本SALib)
#         plt.subplot(1, 2, 1)
#         bar_plot(Si, order='S1', show_confidence=True)
#         plt.title('一阶敏感性指数')
        
#         plt.subplot(1, 2, 2)
#         bar_plot(Si, order='ST', show_confidence=True)
#         plt.title('总敏感性指数')
#         logger.warning("使用旧版SALib参数 (order) 绘制图表")
#     else:
#         # 重新抛出其他类型的TypeError
#         raise
# except Exception as e:
#     logger.error(f"绘制图表时发生错误: {e}")
#     raise

# plt.tight_layout()
# plt.savefig('sobol_sensitivity.png', dpi=300)
# logger.info("敏感性指数图已保存为 sobol_sensitivity.png")

# # 9. 保存结果到Excel
# results_df = pd.DataFrame({
#     '参数': problem['names'],
#     '一阶指数 (S1)': Si['S1'],
#     'S1 置信区间': Si['S1_conf'],
#     '总指数 (ST)': Si['ST'],
#     'ST 置信区间': Si['ST_conf']
# })

# results_df.to_excel('sobol_results.xlsx', index=False)
# logger.info("分析结果已保存为 sobol_results.xlsx")

# logger.info("Sobol敏感性分析完成!")


# --- 8. 绘制敏感性指数图 ---
Si_df = Si.to_df()
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# 绘制一阶指数
ax1 = bar_plot(Si, ax=axes[0])
ax1.set_title('一阶敏感性指数 (S1)', fontsize=14)
ax1.set_ylabel('敏感性指数', fontsize=12)

# 绘制总指数
ax2 = bar_plot(Si, ax=axes[1])
ax2.set_title('总敏感性指数 (ST)', fontsize=14)
ax2.set_ylabel('敏感性指数', fontsize=12)

plt.tight_layout()
plt.savefig('sobol_sensitivity.png', dpi=300)
logger.info("敏感性指数图已保存为 sobol_sensitivity.png")

# --- 9. 保存结果到Excel ---
results_df = pd.DataFrame({
    '参数': problem['names'],
    '一阶指数 (S1)': Si['S1'],
    'S1 置信区间': Si['S1_conf'],
    '总指数 (ST)': Si['ST'],
    'ST 置信区间': Si['ST_conf']
})
results_df.to_excel('sobol_results.xlsx', index=False)
logger.info("分析结果已保存为 sobol_results.xlsx")
logger.info("Sobol敏感性分析完成!")
