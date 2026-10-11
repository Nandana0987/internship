import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Load dataset
df = pd.read_csv("unsupervised/Mall_Customers (1).csv")

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Create dendrogram
plt.figure(figsize=(10, 5))
linkage_matrix = linkage(X, method='ward')

dendrogram(linkage_matrix)
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

# Apply Hierarchical Clustering
hc = AgglomerativeClustering(
    n_clusters=5,
    metric='euclidean',
    linkage='ward'
)

clusters = hc.fit_predict(X)

# Add cluster labels
df['Cluster'] = clusters

# Plot clusters
plt.figure(figsize=(8, 6))
plt.scatter(
    X['Annual Income (k$)'],
    X['Spending Score (1-100)'],
    c=clusters
)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Hierarchical Clustering')
plt.show()

print(df.head())