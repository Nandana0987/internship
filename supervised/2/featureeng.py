import pandas as pd
import numpy as np

# --------------------------
# Load Dataset
# --------------------------
df = pd.read_csv("2/disease_diagnosis.csv")

# --------------------------
# FEATURE ENGINEERING
# --------------------------

# 1. Age Group
df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[0, 18, 40, 60, 100],
    labels=["Child", "Young Adult", "Adult", "Senior"]
)

# --------------------------
# 2. Fever Indicator
# Fever if temperature > 37.5°C
# --------------------------

df["Has_Fever"] = np.where(
    df["Body_Temperature_C"] > 37.5,
    1,
    0
)

# --------------------------
# 3. High Heart Rate Indicator
# --------------------------

df["High_Heart_Rate"] = np.where(
    df["Heart_Rate_bpm"] > 100,
    1,
    0
)

# --------------------------
# 4. Low Oxygen Indicator
# Oxygen below 95% considered low
# --------------------------

df["Low_Oxygen"] = np.where(
    df["Oxygen_Saturation_%"] < 95,
    1,
    0
)

# --------------------------
# 5. Symptom Count
# Count non-null symptoms
# --------------------------

symptom_columns = [
    "Symptom_1",
    "Symptom_2",
    "Symptom_3"
]

df["Symptom_Count"] = (
    df[symptom_columns]
    .notnull()
    .sum(axis=1)
)

# --------------------------
# 6. Severity Score
# Convert severity into numeric
# --------------------------

severity_mapping = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4
}

df["Severity_Score"] = (
    df["Severity"]
    .map(severity_mapping)
)



# --------------------------
# Save Feature Engineered Data
# --------------------------

df.to_csv(
    "feature_engineered_disease_data.csv",
    index=False
)

# --------------------------
# Output
# --------------------------

print("Feature Engineering Completed!")

print("\nNew Columns Added:")
print(df.columns)

print("\nPreview:")
print(df.head())