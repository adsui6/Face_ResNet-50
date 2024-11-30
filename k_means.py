import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 设置环境变量以避免内存泄漏问题
os.environ["OMP_NUM_THREADS"] = "2"  # 你可以根据需要调整这个值

# 加载红酒质量数据集
data = pd.read_csv('winequality-red.csv', sep=';')

# 查看数据的基本信息
print(data.head())

# 提取特征数据
X = data.drop('quality', axis=1)  # 移除标签列
y = data['quality']  # 标签列

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 设置K均值聚类的数量
k = 6  # 根据酒的质量分数设置聚类数量
kmeans = KMeans(n_clusters=k, random_state=42)

# 拟合模型
kmeans.fit(X_scaled)

# 预测聚类
y_kmeans = kmeans.predict(X_scaled)

# 可视化聚类结果（使用前两个特征）
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis', marker='o', edgecolor='k', s=100)

# 标记聚类中心
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')

plt.title('K-Means Clustering on Wine Quality Dataset')
plt.xlabel('Standardized Feature 1')
plt.ylabel('Standardized Feature 2')
plt.legend()
plt.grid()
plt.show()
