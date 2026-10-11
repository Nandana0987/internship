# # ==========================================
# # DATA PREPROCESSING - EMPLOYEE SALARY DATASET
# # ==========================================

# import pandas as pd

# # Load dataset
# df = pd.read_csv("Employee_Salary_Dataset.csv")

# # ==========================================
# # 1. DISPLAY DATASET
# # ==========================================

# print("First 5 Rows of Dataset:\n")
# print(df.head())

# print("\nDataset Shape:")
# print(df.shape)

# print("\nDataset Information:")
# print(df.info())

# # ==========================================
# # 2. CHECK MISSING VALUES
# # ==========================================

# print("\nMissing Values Before Cleaning:\n")
# print(df.isnull().sum())

# # ==========================================
# # 3. REMOVE DUPLICATE RECORDS
# # ==========================================

# duplicates = df.duplicated().sum()

# print("\nDuplicate Rows:", duplicates)

# df = df.drop_duplicates()

# # ==========================================
# # 4. HANDLE MISSING VALUES
# # ==========================================

# df['Age'] = df['Age'].fillna(df['Age'].median())

# df['Years_of_Experience'] = df['Years_of_Experience'].fillna(
#     df['Years_of_Experience'].median()
# )

# df['Salary'] = df['Salary'].fillna(df['Salary'].median())

# # Categorical Columns → Fill with Mode
# df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])

# df['Education_Level'] = df['Education_Level'].fillna(
#     df['Education_Level'].mode()[0]
# )

# df['Job_Title'] = df['Job_Title'].fillna(
#     df['Job_Title'].mode()[0]
# )

# print("\nMissing Values After Cleaning:\n")
# print(df.isnull().sum())

# # ==========================================
# # 5. REMOVE EXTRA SPACES
# # ==========================================

# for col in df.select_dtypes(include=['object']).columns:
#     df[col] = df[col].str.strip()

# print("\nExtra spaces removed from text columns")



# # Example: Joining_Date column
# # Uncomment if dataset contains date column

# # df['Joining_Date'] = pd.to_datetime(df['Joining_Date'], errors='coerce')

# # print("\nJoining_Date converted to datetime format")

# # ==========================================
# # 7. REMOVE OUTLIERS USING IQR METHOD
# # ==========================================

# # Replace 'Salary' with your salary column name if different

# Q1 = df['Salary'].quantile(0.25)
# Q3 = df['Salary'].quantile(0.75)

# IQR = Q3 - Q1

# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR

# before_rows = df.shape[0]

# df = df[(df['Salary'] >= lower_bound) &
#         (df['Salary'] <= upper_bound)]

# after_rows = df.shape[0]

# print("\nOutliers Removed:", before_rows - after_rows)

# # ==========================================
# # 8. FINAL DATASET
# # ==========================================

# print("\nFinal Dataset Shape:")
# print(df.shape)

# print("\nCleaned Dataset:\n")
# print(df.head())

# # ==========================================
# # 9. SAVE CLEANED DATASET
# # ==========================================

# df.to_csv("cleaned_employee_salary_dataset.csv", index=False)

# print("\nCleaned dataset saved successfully!")





# =====================================================
# COMPLETE DATA CLEANING - EMPLOYEE SALARY DATASET
# =====================================================

import pandas as pd
import numpy as np

# =====================================================
# 1. LOAD DATASET
# =====================================================

df = pd.read_csv("employee_salary_dataset (1).csv")

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

df['Age'] = df['Age'].fillna(df['Age'].median())

df['Experience_Years'] = df['Experience_Years'].fillna(
    df['Experience_Years'].median()
)

df['Monthly_Salary'] = df['Monthly_Salary'].fillna(
    df['Monthly_Salary'].median()
)

# ---------- Categorical Columns ----------

df['Department'] = df['Department'].fillna(
    df['Department'].mode()[0]
)

df['Education_Level'] = df['Education_Level'].fillna(
    df['Education_Level'].mode()[0]
)

df['Gender'] = df['Gender'].fillna(
    df['Gender'].mode()[0]
)

df['City'] = df['City'].fillna(
    df['City'].mode()[0]
)

# =====================================================
# 6. REMOVE EXTRA SPACES
# =====================================================

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

print("\nExtra spaces removed")

# =====================================================
# 7. CONVERT TEXT TO LOWERCASE
# =====================================================

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.lower()

print("\nText converted to lowercase")

# =====================================================
# 8. CHECK DATA TYPES
# =====================================================

print("\nData Types:\n")
print(df.dtypes)

# =====================================================
# 9. REMOVE OUTLIERS USING IQR METHOD
# =====================================================

Q1 = df['Monthly_Salary'].quantile(0.25)
Q3 = df['Monthly_Salary'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

before_rows = df.shape[0]

df = df[
    (df['Monthly_Salary'] >= lower_bound) &
    (df['Monthly_Salary'] <= upper_bound)
]

after_rows = df.shape[0]

print("\nOutliers Removed:", before_rows - after_rows)

# =====================================================
# 10. CHECK NULL VALUES AFTER CLEANING
# =====================================================

print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())

# =====================================================
# 11. RENAME COLUMN NAMES
# =====================================================

df.columns = df.columns.str.strip()

print("\nUpdated Column Names:\n")
print(df.columns)

# =====================================================
# 12. FINAL DATASET SHAPE
# =====================================================

print("\nFinal Dataset Shape:")
print(df.shape)

# =====================================================
# 13. DISPLAY CLEANED DATASET
# =====================================================

print("\nCleaned Dataset:\n")
print(df.head())

# =====================================================
# 14. SAVE CLEANED DATASET
# =====================================================

df.to_csv("cleaned_employee_salary_dataset.csv", index=False)

print("\nDataset cleaned and saved successfully!")