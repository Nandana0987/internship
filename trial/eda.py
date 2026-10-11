# # ============================================
# TITANIC DATASET - COMPLETE EDA
# INCLUDING HEATMAP & PAIRPLOT
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("cleaned_titanic_dataset.csv")



# ============================================
# 1. BASIC INFORMATION
# ============================================

print("================================")
print("FIRST 5 ROWS")
print("================================")
print(df.head())

print("\n================================")
print("DATASET SHAPE")
print("================================")
print(df.shape)

print("\n================================")
print("COLUMN NAMES")
print("================================")
print(df.columns)

print("\n================================")
print("DATA TYPES")
print("================================")
print(df.dtypes)

print("\n================================")
print("MISSING VALUES")
print("================================")
print(df.isnull().sum())

print("\n================================")
print("STATISTICAL SUMMARY")
print("================================")
print(df.describe())

# ============================================
# 2. SURVIVAL COUNT
# ============================================

print("\n================================")
print("SURVIVAL COUNT")
print("================================")
print(df["survived"].value_counts())

sns.countplot(x="survived", data=df)

plt.title("Survival Count")
plt.show()

# ============================================
# 3. GENDER DISTRIBUTION
# ============================================

print("\n================================")
print("GENDER DISTRIBUTION")
print("================================")
print(df["sex"].value_counts())

sns.countplot(x="sex", data=df)

plt.title("Gender Distribution")
plt.show()

# ============================================
# 4. PASSENGER CLASS DISTRIBUTION
# ============================================

print("\n================================")
print("PASSENGER CLASS DISTRIBUTION")
print("================================")
print(df["pclass"].value_counts())

sns.countplot(x="pclass", data=df)

plt.title("Passenger Class Distribution")
plt.show()

# ============================================
# 5. AGE DISTRIBUTION
# ============================================

plt.figure(figsize=(8,5))

sns.histplot(df["age"], bins=20, kde=True)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.show()

# ============================================
# 6. FARE DISTRIBUTION
# ============================================

plt.figure(figsize=(8,5))

sns.histplot(df["fare"], bins=20, kde=True)

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")

plt.show()

# ============================================
# 7. SURVIVAL BASED ON GENDER
# ============================================

print("\n================================")
print("SURVIVAL BASED ON GENDER")
print("================================")
print(pd.crosstab(df["sex"], df["survived"]))

sns.countplot(x="sex", hue="survived", data=df)

plt.title("Survival Based on Gender")

plt.show()

# ============================================
# 8. SURVIVAL BASED ON PASSENGER CLASS
# ============================================

print("\n================================")
print("SURVIVAL BASED ON PASSENGER CLASS")
print("================================")
print(pd.crosstab(df["pclass"], df["survived"]))

sns.countplot(x="pclass", hue="survived", data=df)

plt.title("Survival Based on Passenger Class")

plt.show()

# ============================================
# 9. BOXPLOT FOR OUTLIERS
# ============================================

plt.figure(figsize=(8,5))

sns.boxplot(y=df["fare"])

plt.title("Boxplot of Fare")

plt.show()

# ============================================
# 10. CORRELATION MATRIX
# ============================================

numeric_df = df.select_dtypes(include=["int64", "float64"])

correlation = numeric_df.corr()

print("\n================================")
print("CORRELATION MATRIX")
print("================================")
print(correlation)

# ============================================
# 11. HEATMAP
# ============================================

plt.figure(figsize=(12,8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Heatmap of Correlation Matrix")

plt.show()

# ============================================
# 12. PAIRPLOT
# ============================================

sns.pairplot(
    df[[
        "survived",
        "pclass",
        "age",
        "fare",
        "sex"
    ]],
    hue="survived"
)

plt.show()
# ============================================
# 2. PIE CHART FOR SURVIVED
# ============================================

survived_counts = df["survived"].value_counts()

plt.figure(figsize=(6,6))

plt.pie(
    survived_counts,
    labels=["Not Survived", "Survived"],
    autopct="%1.1f%%"
)

plt.title("Survival Distribution")

plt.show()

# ============================================
# 3. SCATTER PLOT
# AGE vs FARE
# ============================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    x=df["age"],
    y=df["fare"],
    hue=df["survived"]
)

plt.title("Scatter Plot: Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")

plt.show()





# ============================================
# 13. EDA COMPLETED
# ============================================

print("\n================================")
print("EDA COMPLETED SUCCESSFULLY")
print("================================")