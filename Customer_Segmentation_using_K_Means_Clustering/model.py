import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load Data
data = pd.read_csv("P:/Machine_Learning_Models/Customer_Segmentation_using_K_Means_Clustering/Mall_Customers.csv")

# Drop unnecessary columns
data_encoded = pd.get_dummies(data, columns=['Gender'], drop_first=True)
data_cluster = data_encoded.drop('CustomerID', axis=1)

# Feature Scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_cluster)

# Elbow Method to determine optimal K
inertia = []

K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state = 42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.show()

# Silhouette Score for K = 2 to 10
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(data_scaled)
    score = silhouette_score(data_scaled, labels)
    print(f"Silhouette Score for k={k}: {score:.4f}")

# Fit KMeans with optimal K (say, 5 based on elbow/silhouette)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data_cluster['Cluster'] = kmeans.fit_predict(data_scaled)

# Visualize Clusters using 2 features (e.g., Annual Income vs Spending Score)
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=data['Annual Income (k$)'],
    y=data['Spending Score (1-100)'],
    hue=data_cluster['Cluster'],
    palette='Set2',
    s=100
)
plt.title('Customer Segments (K-Means Clustering)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True)
plt.show()
