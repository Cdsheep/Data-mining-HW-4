# 导入所需的库
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# 加载加利福尼亚房价数据集
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['target'] = california.target

# 标准化数据
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.drop('target', axis=1))

# 初始化变量
sil_scores = []
k_range = range(2, 7)  # 聚类数从2到6

# 计算每个k值的K-Means聚类和Silhouette得分
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    sil_score = silhouette_score(scaled_data, kmeans.labels_)
    sil_scores.append(sil_score)

# 找到最佳的k值
best_k = k_range[np.argmax(sil_scores)]
best_sil_score = max(sil_scores)

# 最优k值的K-Means聚类
kmeans_best = KMeans(n_clusters=best_k, random_state=42)
kmeans_best.fit(scaled_data)
labels = kmeans_best.labels_
centroids = kmeans_best.cluster_centers_

# 计算每个聚类的均值
cluster_means = pd.DataFrame(columns=df.columns[:-1])  # 去除目标列
for i in range(best_k):
    cluster_means.loc[i] = df.iloc[labels == i, :-1].mean()

# 输出结果
print(f"最佳聚类数 k = {best_k}")
print(f"Silhouette 得分 = {best_sil_score:.4f}")
print("\n每个聚类的特征均值：")
print(cluster_means)

print("\n最优聚类的质心坐标：")
print(centroids)
