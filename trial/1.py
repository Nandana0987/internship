# ================================
# DATA PREPROCESSING ON DATASET
# ================================

import pandas as pd

# Load dataset
df = pd.read_csv("sales_data_sample.csv", encoding='latin1')

# Display first 5 rows
print("Original Dataset:")
print(df.head())

# ================================
# 1. CHECK DATASET INFORMATION
# ================================

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Shape:", df.shape)

# ================================
# 2. REMOVE DUPLICATE RECORDS
# ================================

duplicates = df.duplicated().sum()
print("\nDuplicate Rows:", duplicates)

df = df.drop_duplicates()

# ================================
# 3. HANDLE MISSING VALUES
# ================================

# Fill numerical columns with median
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# ================================
# 4. CONVERT DATE COLUMN
# ================================

df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], errors='coerce')

print("\nORDERDATE converted to datetime format")

# ================================
# 5. REMOVE EXTRA SPACES
# ================================

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()

print("\nText formatting standardized")

# ================================
# 6. REMOVE OUTLIERS USING IQR
# ================================

Q1 = df['SALES'].quantile(0.25)
Q3 = df['SALES'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

before_rows = df.shape[0]

df = df[(df['SALES'] >= lower_bound) &
        (df['SALES'] <= upper_bound)]

after_rows = df.shape[0]

print("\nOutliers Removed:", before_rows - after_rows)

# ================================
# 7. FINAL DATASET
# ================================

print("\nFinal Dataset Shape:", df.shape)

print("\nCleaned Dataset Sample:")
print(df.head())

# ================================
# 8. SAVE CLEANED DATASET
# ================================

df.to_csv("cleaned_sales_data.csv", index=False)

print("\nCleaned dataset saved successfully!")