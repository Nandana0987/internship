import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --------------------------
# Load Dataset
# --------------------------
df = pd.read_csv("1/sales_data.csv")

# --------------------------
# 1. Handling Missing Values
# --------------------------

# Check missing values
print("Missing Values Before Cleaning:")
print(df.isnull().sum())

# Fill missing numeric values with mean
numeric_cols = df.select_dtypes(include=["number"]).columns

for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)

# Fill missing categorical values with mode
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# --------------------------
# 2. Removing Duplicate Data
# --------------------------

# Check duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# --------------------------
# 3. Fixing Data Types
# --------------------------

# Check datatypes
print("\nData Types Before Conversion:")
print(df.dtypes)

# Convert Sale_Date to datetime format
if "Sale_Date" in df.columns:
    df["Sale_Date"] = pd.to_datetime(df["Sale_Date"], errors="coerce")

# Convert numeric columns to proper datatype
possible_numeric_cols = [
    "Age",
    "Sales_Amount",
    "Quantity_Sold",
    "Unit_Cost",
    "Unit_Price",
    "Discount",
    "Product_ID"
]

for col in possible_numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# --------------------------
# 4. Handling Outliers
# --------------------------

for col in numeric_cols:

    # Draw boxplot
    plt.figure(figsize=(5, 3))
    plt.boxplot(df[col].dropna())
    plt.title(f"Boxplot of {col}")
    plt.show()

    # IQR Method
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

# --------------------------
# 5. Handling Inconsistent Data
# --------------------------

for col in categorical_cols:

    # Convert text to lowercase
    df[col] = df[col].astype(str).str.lower()

    # Remove extra spaces
    df[col] = df[col].str.strip()

# --------------------------
# 6. Renaming Columns
# --------------------------

# Convert column names to lowercase
df.columns = df.columns.str.lower()

# Example rename
if "emp_salary" in df.columns:
    df.rename(columns={"emp_salary": "salary"}, inplace=True)

# --------------------------
# 7. Handling Date Formats
# --------------------------

if "sale_date" in df.columns:
    df["sale_date"] = pd.to_datetime(df["sale_date"])

# --------------------------
# 8. Encoding Categorical Data
# --------------------------

le = LabelEncoder()

for col in categorical_cols:
    if col.lower() in df.columns:
        df[col.lower()] = le.fit_transform(df[col.lower()].astype(str))

# --------------------------
# 9. Feature Scaling
# --------------------------

scaler = StandardScaler()

numeric_columns_after = df.select_dtypes(include=["number"]).columns

df[numeric_columns_after] = scaler.fit_transform(
    df[numeric_columns_after]
)

# --------------------------
# Save Cleaned Dataset
# --------------------------

df.to_csv("cleaned_sales_data.csv", index=False)

print("\nData Cleaning Completed Successfully!")
print("Final Shape:", df.shape)

# Preview Data
print(df.head())