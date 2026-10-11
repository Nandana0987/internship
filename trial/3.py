# =====================================================
# COMPLETE DATA CLEANING - employees(1).csv
# USING EXACT SAME COLUMN NAMES
# =====================================================

import pandas as pd
import numpy as np

# =====================================================
# 1. LOAD DATASET
# =====================================================

df = pd.read_csv("employees.csv")

print("First 5 Rows:\n")
print(df.head())

# =====================================================
# 2. DATASET INFORMATION
# =====================================================

print("\nDataset Info:\n")
print(df.info())

print("\nDataset Shape:")
print(df.shape)

# =====================================================
# 3. CHECK MISSING VALUES
# =====================================================

print("\nMissing Values Before Cleaning:\n")
print(df.isnull().sum())

# =====================================================
# 4. REMOVE DUPLICATES
# =====================================================

duplicate_rows = df.duplicated().sum()

print("\nDuplicate Rows:", duplicate_rows)

df = df.drop_duplicates()

# =====================================================
# 5. HANDLE MISSING VALUES
# =====================================================

# ---------- Numerical Columns ----------

df['Salary'] = df['Salary'].fillna(
    df['Salary'].median()
)

df['Bonus %'] = df['Bonus %'].fillna(
    df['Bonus %'].median()
)

# ---------- Categorical Columns ----------

df['First Name'] = df['First Name'].fillna(
    df['First Name'].mode()[0]
)

df['Gender'] = df['Gender'].fillna(
    df['Gender'].mode()[0]
)

df['Team'] = df['Team'].fillna(
    df['Team'].mode()[0]
)

df['Senior Management'] = df['Senior Management'].fillna(
    df['Senior Management'].mode()[0]
)

# =====================================================
# 6. REMOVE EXTRA SPACES
# =====================================================

for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].astype(str).str.strip()
print("\nExtra spaces removed")

# =====================================================
# 7. CONVERT TEXT TO LOWERCASE
# =====================================================
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].astype(str).str.lower()
print("\nText converted to lowercase")

# =====================================================
# 8. CONVERT DATE COLUMNS
# =====================================================

df['Start Date'] = pd.to_datetime(
    df['Start Date'],
    errors='coerce'
)

print("\nStart Date converted to datetime format")

# =====================================================
# 9. REMOVE OUTLIERS USING IQR METHOD
# =====================================================

Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

before_rows = df.shape[0]

df = df[
    (df['Salary'] >= lower_bound) &
    (df['Salary'] <= upper_bound)
]

after_rows = df.shape[0]

print("\nOutliers Removed:", before_rows - after_rows)

# =====================================================
# 10. CHECK MISSING VALUES AFTER CLEANING
# =====================================================

print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())

# =====================================================
# 11. FINAL DATASET SHAPE
# =====================================================

print("\nFinal Dataset Shape:")
print(df.shape)

# =====================================================
# 12. DISPLAY CLEANED DATASET
# =====================================================

print("\nCleaned Dataset:\n")
print(df.head())

# =====================================================
# 13. SAVE CLEANED DATASET
# =====================================================

df.to_csv("cleaned_employees_dataset.csv", index=False)

print("\nDataset cleaned and saved successfully!")