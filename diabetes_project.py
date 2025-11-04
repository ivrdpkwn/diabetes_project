# ===============================
# Diabetes Progression Prediction
# ===============================
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# 1. 数据加载与探索
# ===============================
# 加载数据
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target


# ===============================
# 2. 数据预处理与划分
# ===============================
# 拆分训练集和测试集，比例 80%:20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state= 42)

print('训练集样本数', X_train.shape[0])
print('测试集样本数', X_test.shape[0])


# ===============================
# 3. 线性回归建模
# ===============================
# 创建线性回归模型对象
model = LinearRegression()

# 用训练集拟合模型
model.fit(X_train, y_train)

# 查看训练好的系数和截距
print("回归系数:", model.coef_)
print("截距:", model.intercept_)

# 用训练好的模型对测试集做预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred)
print(f"均方误差 (MSE): {mse:.2f}") # 越小越好
print(f"决定系数 (R²): {r2:.2f}") # 越接近 1 模型拟合越好


# ===============================
# 4. 可视化预测结果
# ===============================
# 绘制真实值 vs 预测值散点图
plt.figure(figsize=(8, 6))
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.scatter(y_test, y_pred, color='skyblue', edgecolor='k')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2 ) # 理想的预测线 y=x
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('线性回归预测 vs 真实值')
plt.show()


