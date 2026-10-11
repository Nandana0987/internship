import pandas as pd                    # For reading dataset
import matplotlib.pyplot as plt        # For plotting graphs
from sklearn.cluster import KMeans     # K-Means clustering algorithm
from sklearn.metrics import silhouette_score
# -----------------------------
# STEP 1: Load Dataset
# -----------------------------

df = pd.read_csv("unsupervised/Mall_Customers (1).csv") 

# Select features for clustering
# We use Income and Spending Score
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values


# -----------------------------
# STEP 2: Elbow Method
# -----------------------------

wcss = []   # WCSS = Within Cluster Sum of Squares (measure of cluster error)

# Test different numbers of clusters
for i in range(1, 11):                      # Try cluster numbers from 1 to 10

    kmeans = KMeans(n_clusters=i, random_state=42)   # Create KMeans model

    kmeans.fit(X)                           # Train the model using dataset

    wcss.append(kmeans.inertia_)            # Store WCSS value (cluster error)

# Plot Elbow Graph
plt.plot(range(1, 11), wcss)                # X-axis = number of clusters, Y-axis = WCSS
plt.title("Elbow Method for Optimal K")     # Graph title
plt.xlabel("Number of Clusters (K)")        # X-axis label
plt.ylabel("WCSS")                          # Y-axis label
plt.show()                                  # Display graph


# -----------------------------
# STEP 3: Apply K-Means
# -----------------------------

# Assume elbow graph suggests K = 5
kmeans = KMeans(n_clusters=4, random_state=42)

y_kmeans = kmeans.fit_predict(X)            # Train model and assign clusters

# -----------------------------
# STEP 4: Silhouette Score
# -----------------------------

score = silhouette_score(X, y_kmeans)       # Calculate silhouette score

print("Silhouette Score:", score)           # Print clustering quality


# -----------------------------
# STEP 5: Visualize Clusters
# -----------------------------

# Plot clustered data
plt.scatter(X[:,0], X[:,1], c=y_kmeans)     # Color points by cluster

# Plot centroids (cluster centers)
plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1],
            s=200)                          # Larger size for visibility

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.show()