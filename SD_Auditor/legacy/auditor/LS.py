import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 生成模拟数据
#np.random.seed(42)
n_samples = 1000
n_features = 10
b = 10
n_outliers = 13
# 生成特征矩阵 X
X = np.random.rand(n_samples, n_features)
# 真实的回归系数 beta
true_beta = np.random.rand(n_features) * 10  # 不包括截距项
# 生成目标变量 Y，添加一些随机噪声
Y = X.dot(true_beta) + np.random.randn(n_samples) * 1.5

# 分割数据集为训练集和测试集
train_size = int(0.8 * n_samples)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# 使用最小二乘法计算训练集上的回归系数 (没有修改Y)
# beta = (X^T X)^-1 X^T Y
X_transpose = X_train.T
X_transpose_X = X_transpose.dot(X_train)
X_transpose_X_inv = np.linalg.inv(X_transpose_X)
X_transpose_Y = X_transpose.dot(Y_train)
beta_hat_original = X_transpose_X_inv.dot(X_transpose_Y)

# 计算训练数据的残差（使用没有调整前的Y进行预测得到的系数）
Y_train_pred_original = X_train.dot(beta_hat_original)
residuals_original = Y_train_pred_original - Y_train

# 输出前100个训练数据的残差
print("Residuals for the first 100 training data points:")
print(residuals_original[:100])

# 修改第一个数据点的目标变量，添加一个常数
Y_train_modified = Y_train.copy()
Y_train_modified[:n_outliers] += b

# 使用最小二乘法计算训练集上的回归系数
# beta = (X^T X)^-1 X^T Y
X_transpose_Y = X_transpose.dot(Y_train_modified)
beta_hat = X_transpose_X_inv.dot(X_transpose_Y)

# 计算训练数据的残差
Y_train_pred = X_train.dot(beta_hat)
residuals = Y_train_pred - Y_train_modified

# 输出前100个训练数据的残差
print("Residuals for the first 100 training data points:")
print(residuals[:100])

# 提取第一个样本
X_first_sample = X_train[0]

s = X_train.dot(X_transpose_X_inv.dot(b * X_first_sample))

r = residuals_original + s
#print(r[:100])

# 提取前10个样本
X_first_10_samples = X_train[:n_outliers]

# 计算前10个样本的调整影响
s10 = X_train.dot(X_transpose_X_inv.dot(b * X_first_10_samples.T).sum(axis=1))
print("Sum of the influence of the first 10 modified samples:")
print(s10)
r10 = residuals_original + s10
#print(r10[:100])


