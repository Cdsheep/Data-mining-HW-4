import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score

# 1. 加载葡萄酒数据集
wine_data = load_wine()
X = wine_data.data
y = wine_data.target

# 2. 将数据转换为 Pandas DataFrame
df = pd.DataFrame(X, columns=wine_data.feature_names)
df['actual_labels'] = y

# 3. 数据缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 执行 K-Means 聚类（设定聚类数为 3）
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
y_pred = kmeans.labels_

# 5. 计算同质性和完整性
homogeneity = homogeneity_score(y, y_pred)
completeness = completeness_score(y, y_pred)

# 6. 输出结果
print(f"Homogeneity: {homogeneity}")
print(f"Completeness: {completeness}")
