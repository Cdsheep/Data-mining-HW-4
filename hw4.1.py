import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
           'acceleration', 'model_year', 'origin', 'car_name']
data = pd.read_csv(url, delim_whitespace=True, names=columns, na_values='?')

# 处理缺失值
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
data['horsepower'].fillna(data['horsepower'].mean(), inplace=True)

# 选取用于聚类的连续型特征
features = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']
X = data[features]

# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 层次聚类
model = AgglomerativeClustering(n_clusters=3, linkage='average')
data['cluster'] = model.fit_predict(X)

# 输出每个聚类的均值和方差
print("Cluster stats:")
print(data.groupby('cluster')[features].agg(['mean', 'var']))

# 按 origin 分组统计
print("\nOrigin stats:")
print(data.groupby('origin')[features].agg(['mean', 'var']))

# 聚类与 origin 的关系
print("\nCluster vs Origin:")
print(pd.crosstab(data['cluster'], data['origin']))

# 可视化
sns.heatmap(pd.crosstab(data['cluster'], data['origin']), annot=True, cmap='Blues')
plt.title("Cluster vs Origin")
plt.xlabel("Origin")
plt.ylabel("Cluster")
plt.show()
