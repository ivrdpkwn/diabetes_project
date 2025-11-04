# 糖尿病进展预测项目

## 项目简介
使用 Python 对糖尿病数据集进行线性回归建模，预测糖尿病患者的一年后疾病进展指标（target）。项目涵盖完整的机器学习工作流程，包括数据探索、预处理、模型训练、评估及可视化。

## 项目功能
1. 数据探索（查看前几行数据、数据形状、缺失值检查）
2. 相关性分析及热力图可视化
3. 训练集/测试集划分
4. 线性回归建模及预测
5. 模型评估（MSE、R²）
6. 预测结果可视化（散点图）
7. 总结与改进建议

## 使用方法
下载 `diabetes_project.py` 文件，确保安装依赖：
pip install numpy pandas scikit-learn matplotlib seaborn

然后在命令行运行：

python diabetes_project.py

## 文件结构

diabetes_project/
├─ diabetes_project.py
└─ README.md

## 参考
- 数据集来源：scikit-learn `load_diabetes()`
