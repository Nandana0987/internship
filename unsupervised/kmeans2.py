# ==========================================
# CUSTOMER SEGMENTATION USING K-MEANS
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score

# ----------------------------------
# Load Dataset
# ----------------------------------

df = pd.read_csv(
    "unsupervised/marketing_campaign.csv",
    sep="\t"
)

print(df.head())
print(df.shape)

# ----------------------------------
# Missing Values
# ----------------------------------

df["Income"] = df["Income"].fillna(
    df["Income"].median()
)

# ----------------------------------
# Drop Unnecessary Columns
# ----------------------------------

df.drop(
    ["ID", "Dt_Customer"],
    axis=1,
    inplace=True
)

# ----------------------------------
# Encode Categorical Columns
# ----------------------------------

le = LabelEncoder()

df["Education"] = le.fit_transform(
    df["Education"]
)

df["Marital_Status"] = le.fit_transform(
    df["Marital_Status"]
)

# ----------------------------------
# Feature Scaling
# ----------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df)

# ----------------------------------
# Elbow Method
# ----------------------------------

wcss = []

for i in range(1,11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    wcss.append(
        kmeans.inertia_
    )

plt.figure(figsize=(8,5))

plt.plot(
    range(1,11),
    wcss,
    marker='o'
)

plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")

plt.show()

# ----------------------------------
# Train KMeans
# ----------------------------------

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(
    X_scaled
)

df["Cluster"] = clusters

# ----------------------------------
# Cluster Distribution
# ----------------------------------

print(
    df["Cluster"]
    .value_counts()
)

# ----------------------------------
# Silhouette Score
# ----------------------------------

score = silhouette_score(
    X_scaled,
    clusters
)

print(
    "\nSilhouette Score:",
    score
)

# ----------------------------------
# PCA Visualization
# ----------------------------------

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

plt.figure(figsize=(10,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters
)

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.title(
    "KMeans Customer Segmentation"
)

plt.show()

# ----------------------------------
# Cluster Summary
# ----------------------------------

cluster_summary = df.groupby(
    "Cluster"
).mean(
    numeric_only=True
)

print(
    "\nCluster Summary"
)

print(cluster_summary)

# ----------------------------------
# Save Results
# ----------------------------------

df.to_csv(
    "Marketing_KMeans_Output.csv",
    index=False
)

print(
    "\nOutput Saved Successfully"
)