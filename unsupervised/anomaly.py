# ==========================================================
# ANOMALY DETECTION USING ISOLATION FOREST
# CREDIT CARD FRAUD DETECTION DATASET
# ==========================================================

# -----------------------------
# Step 1: Import Libraries
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# -----------------------------
# Step 2: Load Dataset
# -----------------------------

df = pd.read_csv("unsupervised/creditcard.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# -----------------------------
# Step 3: Check Missing Values
# -----------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Step 4: Check Class Distribution
# -----------------------------

print("\nClass Distribution:")
print(df["Class"].value_counts())

# -----------------------------
# Step 5: Separate Features
# -----------------------------

X = df.drop("Class", axis=1)

y = df["Class"]

print("\nFeature Matrix Shape:")
print(X.shape)

# -----------------------------
# Step 6: Scale Features
# -----------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled Data Shape:")
print(X_scaled.shape)

# -----------------------------
# Step 7: Train Isolation Forest
# -----------------------------

model = IsolationForest(
    n_estimators=100,
    contamination=0.002,
    random_state=42
)

model.fit(X_scaled)

# -----------------------------
# Step 8: Predict Anomalies
# -----------------------------

predictions = model.predict(X_scaled)

# Isolation Forest Output:
# 1  = Normal
# -1 = Anomaly

print("\nPrediction Counts:")
print(pd.Series(predictions).value_counts())

# -----------------------------
# Step 9: Convert Labels
# -----------------------------

pred_binary = np.where(
    predictions == -1,
    1,   # Fraud
    0    # Normal
)

# Add prediction column
df["Predicted_Class"] = pred_binary

# -----------------------------
# Step 10: Evaluation
# -----------------------------

print("\nClassification Report:")
print(
    classification_report(
        y,
        pred_binary
    )
)

# -----------------------------
# Step 11: Confusion Matrix
# -----------------------------

cm = confusion_matrix(
    y,
    pred_binary
)

print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# Step 12: Count Detected Frauds
# -----------------------------

frauds = df[
    df["Predicted_Class"] == 1
]

print("\nDetected Fraud Transactions:")
print(frauds.shape[0])

# -----------------------------
# Step 13: Save Fraud Records
# -----------------------------

frauds.to_csv(
    "Detected_Fraud.csv",
    index=False
)

print(
    "\nFraud records saved as "
    "'Detected_Fraud.csv'"
)

# -----------------------------
# Step 14: Visualization
# -----------------------------

plt.figure(figsize=(10,6))

plt.scatter(
    df["Time"],
    df["Amount"],
    c=df["Predicted_Class"]
)

plt.xlabel("Time")
plt.ylabel("Transaction Amount")

plt.title(
    "Isolation Forest Anomaly Detection"
)

plt.show()

# -----------------------------
# Step 15: Compare Actual vs Predicted
# -----------------------------

comparison = pd.DataFrame({
    "Actual": y,
    "Predicted": pred_binary
})

print("\nActual vs Predicted:")
print(comparison.head(20))

# -----------------------------
# Step 16: Anomaly Percentage
# -----------------------------

anomaly_percentage = (
    len(frauds) / len(df)
) * 100

print(
    "\nPercentage of Detected Anomalies:",
    round(anomaly_percentage, 3),
    "%"
)

# -----------------------------
# Step 17: Top Anomalous Records
# -----------------------------

print("\nTop Detected Anomalies:")
print(frauds.head(10))

# -----------------------------
# Step 18: Save Full Dataset
# -----------------------------

df.to_csv(
    "CreditCard_With_Anomaly_Label.csv",
    index=False
)

print(
    "\nFull dataset with anomaly labels saved."
)

# ==========================================================
# END
# ==========================================================