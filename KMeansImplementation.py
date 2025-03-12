import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Generate Synthetic Data
np.random.seed(42)
random_heights = np.random.normal(170, 6.5, 1000)  # Mean 170cm, Std Dev 6.5
noise = np.random.normal(0, 5, 1000)  # Random noise for realism
weights = random_heights - 100 + noise  # Correlation between height & weight

# Create a DataFrame
df = pd.DataFrame({'Height': random_heights, 'Weight': weights})

# Step 2: Feature Scaling (Important for K-Means)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)  # Standardizing features

# Step 3: Determine Optimal K (Elbow Method + Silhouette Score)
wcss = []  # Within-cluster sum of squares
sil_scores = []  # Silhouette scores

K_range = range(2, 10)  # Checking clusters from 2 to 10
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)  # Sum of squared distances
    sil_scores.append(silhouette_score(df_scaled, kmeans.labels_))  # Cluster quality

# Plot Elbow Method
plt.figure(figsize=(10, 4))
plt.plot(K_range, wcss, marker='o', linestyle='-', label="WCSS")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Within-Cluster Sum of Squares")
plt.title("Elbow Method for Optimal K")
plt.legend()
plt.show()

# Plot Silhouette Score
plt.figure(figsize=(10, 4))
plt.plot(K_range, sil_scores, marker='o', linestyle='-', color='green', label="Silhouette Score")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Analysis")
plt.legend()
plt.show(block=True)

# Step 4: Apply K-Means with Optimal K (Based on Elbow & Silhouette)
optimal_k = 4  # Assume from elbow method
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Step 5: Assign Size Labels (S, M, L, XL) Based on Cluster Centroids
cluster_means = df.groupby('Cluster')[['Height', 'Weight']].mean().sort_values('Height')
size_labels = ['S', 'M', 'L', 'XL']
size_mapping = {cluster: size for cluster, size in zip(cluster_means.index, size_labels)}
df['T-Shirt Size'] = df['Cluster'].map(size_mapping)

# Step 6: Visualize Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Height', y='Weight', hue='T-Shirt Size', palette='coolwarm', s=50)
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("T-Shirt Size Clustering")
plt.legend(title="Size")
plt.show(block=True)

# Display first few rows
df.head()