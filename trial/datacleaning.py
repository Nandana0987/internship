# ============================================
# TITANIC DATA CLEANING COMPLETE PYTHON CODE
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("Titanic-Dataset (1).csv")

print("Original Dataset Shape:")
print(df.shape)

# -----------------------------
# 2. Check Missing Values
# -----------------------------
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# -----------------------------
# 3. Handle Missing Values
# -----------------------------

# Fill numeric columns with median
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
# ============================================
# 1. FORWARD FILL (ffill)
# ============================================

# Create a copy for forward fill
df_ffill = df.copy()

# Fill missing values using previous row values
df_ffill.ffill(inplace=True)

print("\nMissing Values After Forward Fill:\n")
print(df_ffill.isnull().sum())

# Save forward filled dataset
df_ffill.to_csv("titanic_forward_fill.csv", index=False)

print("\nForward Fill Dataset Saved!")

# ============================================
# 2. BACKWARD FILL (bfill)
# ============================================

# Create a copy for backward fill
df_bfill = df.copy()

# Fill missing values using next row values
df_bfill.bfill(inplace=True)

print("\nMissing Values After Backward Fill:\n")
print(df_bfill.isnull().sum())

# Save backward filled dataset
df_bfill.to_csv("titanic_backward_fill.csv", index=False)

print("\nBackward Fill Dataset Saved!")
# -----------------------------
# 4. Remove Duplicate Rows
# -----------------------------
print("\nDuplicate Rows Before:")
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("Duplicate Rows After:")
print(df.duplicated().sum())

# -----------------------------
# 5. Fix Data Types
# -----------------------------
print("\nData Types Before:")
print(df.dtypes)

# Convert Age to integer
df["Age"] = df["Age"].astype(int)

print("\nData Types After:")
print(df.dtypes)

# -----------------------------
# 6. Handle Outliers (IQR Method)
# -----------------------------

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    # Remove outliers
    df = df[
        (df[col] >= lower_limit) &
        (df[col] <= upper_limit)
    ]

# -----------------------------
# 7. Boxplot for Outlier Detection
# -----------------------------
plt.boxplot(df["Fare"])
plt.title("Boxplot of Fare")
plt.show()

# -----------------------------
# 8. Handle Inconsistent Data
# -----------------------------

# Convert text to lowercase
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.lower()

# Remove extra spaces
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()

# -----------------------------
# 9. Rename Columns
# -----------------------------
df.columns = df.columns.str.lower()

print("\nUpdated Column Names:")
print(df.columns)

# -----------------------------
# 10. Handle Date Format
# -----------------------------
# Example if date column exists

# df["date"] = pd.to_datetime(df["date"])

# -----------------------------
# 11. Encode Categorical Data
# --------------------------------
# LABEL ENCODING
# EXCLUDING 'name' COLUMN
# --------------------------------

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Loop through object columns
for col in df.select_dtypes(include=['object']).columns:

    # Exclude name and gender columns
    if col not in ["name", "sex"]:
        df[col] = le.fit_transform(df[col])
print(df.head())

print("\nDataset After Encoding:")
print(df.head())


# -----------------------------
# 13. Save Cleaned Dataset
# -----------------------------
df.to_csv("cleaned_titanic_dataset.csv", index=False)

print("\nData Cleaning Completed Successfully!")
print("Cleaned dataset saved as 'cleaned_titanic_dataset.csv'")