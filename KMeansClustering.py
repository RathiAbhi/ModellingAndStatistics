import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Generate synthetic dataset
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.05, random_state=42)

# 📌 Step 1: Elbow Method to Find Best K
inertia = []  # Stores distortion (sum of squared distances)
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)  # Inertia = sum of squared distances

# 📌 Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Distortion)')
plt.title('Elbow Method for Optimal K')
plt.show()

# 📌 Step 2: Silhouette Score to Validate K
best_k = 2
best_score = -1

for k in range(2, 11):  # Silhouette needs at least 2 clusters
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)  # Measures cluster separation

    if score > best_score:
        best_score = score
        best_k = k

print(f"Best K using Silhouette Score: {best_k} (Score: {best_score:.2f})")

# 📌 Step 3: Train K-Means with Best K
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# 📌 Step 4: Plot Final Clusters
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label="Centroids")
plt.title(f"K-Means Clustering (K={best_k})")
plt.legend()
plt.show(block=True)