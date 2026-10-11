import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Load Dataset
# --------------------------
df = pd.read_csv("1/sales_data.csv")

# Check column names
print(df.columns)

# --------------------------
# 1. HEATMAP
# --------------------------

# Select numeric columns only
numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(10, 6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# --------------------------
# 2. COUNTPLOT
# --------------------------

# Count of products by category
plt.figure(figsize=(8, 5))

sns.countplot(
    x="Product_Category",
    data=df
)

plt.title("Count of Products by Category")
plt.xticks(rotation=45)
plt.show()

# --------------------------
# 3. BARPLOT
# --------------------------

# Sales amount by region
plt.figure(figsize=(8, 5))

sns.barplot(
    x="Region",
    y="Sales_Amount",
    data=df
)

plt.title("Sales Amount by Region")
plt.xticks(rotation=45)
plt.show()

# --------------------------
# 4. HISTOGRAM
# --------------------------

# Distribution of Sales Amount
plt.figure(figsize=(8, 5))

plt.hist(
    df["Sales_Amount"],
    bins=20
)

plt.title("Distribution of Sales Amount")
plt.xlabel("Sales Amount")
plt.ylabel("Frequency")
plt.show()