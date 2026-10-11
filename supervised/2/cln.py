import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --------------------------
# Load Dataset
# --------------------------
df = pd.read_csv("disease_diagnosis.csv")

# --------------------------
# 1. Handling Missing Values
# --------------------------

# Check missing values
print("Missing Values Before Cleaning:")
print(df.isnull().sum())

# Numeric columns
numeric_cols = [
    "Patient_ID",
    "Age",
    "Heart_Rate_bpm",
    "Body_Temperature_C",
    "Oxygen_Saturation_%"
]

# Categorical columns
categorical_cols = [
    "Gender",
    "Symptom_1",
    "Symptom_2",
    "Symptom_3",
    "Blood_Pressure_mmHg",
    "Diagnosis",
    "Severity",
    "Treatment_Plan"
]

# Fill numeric missing values with mean
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

# Fill categorical missing values with mode
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# --------------------------
# 2. Removing Duplicate Data
# --------------------------

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# --------------------------
# 3. Fixing Data Types
# --------------------------

print("\nData Types Before Conversion:")
print(df.dtypes)

# Convert numeric columns
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# --------------------------
# 4. Handling Outliers
# --------------------------

for col in numeric_cols:

    plt.figure(figsize=(5,3))
    plt.boxplot(df[col].dropna())
    plt.title(f"Boxplot of {col}")
    plt.show()

    # IQR Method
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - (1.5 * IQR)
    upper_limit = Q3 + (1.5 * IQR)

    # Remove outliers
    df = df[
        (df[col] >= lower_limit) &
        (df[col] <= upper_limit)
    ]

# --------------------------
# 5. Handling Inconsistent Data
# --------------------------

for col in categorical_cols:

    # Convert to lowercase
    df[col] = df[col].astype(str).str.lower()

    # Remove extra spaces
    df[col] = df[col].str.strip()

# --------------------------
# 6. Renaming Columns
# --------------------------

# Convert all column names to lowercase
df.columns = df.columns.str.lower()

# Example rename
if "body_temperature_c" in df.columns:
    df.rename(
        columns={
            "body_temperature_c":
            "temperature_c"
        },
        inplace=True
    )





# --------------------------
# Save Cleaned Dataset
# --------------------------

df.to_csv(
    "cleaned_disease_diagnosis.csv",
    index=False
)

# --------------------------
# Output
# --------------------------

print("\nData Cleaning Completed!")
print("Final Shape:", df.shape)

print("\nPreview of Cleaned Data:")
print(df.head())