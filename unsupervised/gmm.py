# =====================================================
# GAUSSIAN MIXTURE MODEL (GMM)
# MALL CUSTOMER SEGMENTATION
# =====================================================

# -----------------------------
# Step 1: Import Libraries
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# -----------------------------
# Step 2: Load Dataset
# -----------------------------

df = pd.read_csv("unsupervised/Mall_Customers (1).csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

# -----------------------------
# Step 3: Check Missing Values
# -----------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Step 4: Encode Gender
# -----------------------------

le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])

print("\nAfter Encoding Gender:")
print(df.head())

# -----------------------------
# Step 5: Select Features
# -----------------------------

X = df[[
    "Annual Income (k$)",
    "Spending Score (1-100)"
]]

print("\nSelected Features:")
print(X.head())

# -----------------------------
# Step 6: Feature Scaling
# -----------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled Data Shape:")
print(X_scaled.shape)

# -----------------------------
# Step 7: Find Best Number of Clusters
# Using BIC
# -----------------------------

bic_scores = []

for k in range(2, 11):

    gmm = GaussianMixture(
        n_components=k,
        random_state=42
    )

    gmm.fit(X_scaled)

    bic_scores.append(
        gmm.bic(X_scaled)
    )

# Plot BIC

plt.figure(figsize=(8,5))

plt.plot(
    range(2,11),
    bic_scores,
    marker='o'
)

plt.xlabel("Number of Clusters")
plt.ylabel("BIC Score")
plt.title("BIC Method for Optimal Clusters")

plt.show()

# -----------------------------
# Step 8: Train GMM
# -----------------------------

gmm = GaussianMixture(
    n_components=5,
    random_state=42
)

gmm.fit(X_scaled)

# -----------------------------
# Step 9: Predict Clusters
# -----------------------------

clusters = gmm.predict(X_scaled)

df["Cluster"] = clusters

print("\nCluster Labels:")
print(df["Cluster"].value_counts())

# -----------------------------
# Step 10: Cluster Probabilities
# -----------------------------

probabilities = gmm.predict_proba(X_scaled)

print("\nCluster Probabilities:")
print(probabilities[:10])

# -----------------------------
# Step 11: Cluster Summary
# -----------------------------

cluster_summary = df.groupby(
    "Cluster"
).mean(numeric_only=True)

print("\nCluster Summary:")
print(cluster_summary)

# -----------------------------
# Step 12: Visualize Clusters
# -----------------------------

plt.figure(figsize=(10,6))

scatter = plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"]
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Gaussian Mixture Model Clustering")

plt.colorbar(scatter)

plt.show()

# -----------------------------
# Step 13: Cluster Centers
# -----------------------------

centers = scaler.inverse_transform(
    gmm.means_
)

print("\nCluster Centers:")
print(centers)

# -----------------------------
# Step 14: Save Output
# -----------------------------

df.to_csv(
    "Mall_Customers_GMM_Output.csv",
    index=False
)

print(
    "\nClustered dataset saved as "
    "'Mall_Customers_GMM_Output.csv'"
)

# -----------------------------
# Step 15: Display Final Data
# -----------------------------

print("\nFinal Dataset:")
print(df.head(20))