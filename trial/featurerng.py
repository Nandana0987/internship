# ============================================
# FEATURE ENGINEERING - TITANIC DATASET
# ============================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv("cleaned_titanic_dataset.csv")

# Convert column names to lowercase
df.columns = df.columns.str.lower()

print("================================")
print("ORIGINAL DATASET")
print("================================")
print(df.head())


# ============================================
# 3. AGE GROUP FEATURE
# ============================================

# Create age categories

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 12, 19, 35, 60, 100],
    labels=[
        "Child",
        "Teen",
        "Young Adult",
        "Adult",
        "Senior"
    ]
)

print("\n================================")
print("AGE GROUP FEATURE ADDED")
print("================================")
print(df[["age", "age_group"]].head())

# ============================================
# 4. FARE CATEGORY FEATURE
# ============================================

# Create fare categories

df["fare_category"] = pd.cut(
    df["fare"],
    bins=[0, 10, 30, 100, 600],
    labels=[
        "Low",
        "Medium",
        "High",
        "Very High"
    ]
)

print("\n================================")
print("FARE CATEGORY FEATURE ADDED")
print("================================")
print(df[["fare", "fare_category"]].head())

# ============================================
# 5. TITLE EXTRACTION FROM NAME
# ============================================

# Extract titles like Mr, Mrs, Miss

if "name" in df.columns:

    df["title"] = df["name"].astype(str).str.extract(
        r'([A-Za-z]+)\.',
        expand=False
    )

    print("\n================================")
    print("TITLE FEATURE ADDED")
    print("================================")
    print(df[["name", "title"]].head())

# ============================================
# 6. ENCODE NEW CATEGORICAL FEATURES
# ============================================

le = LabelEncoder()

categorical_cols = [
    "age_group",
    "fare_category"
]

for col in categorical_cols:

    df[col] = le.fit_transform(
        df[col].astype(str)
    )

print("\n================================")
print("ENCODING COMPLETED")
print("================================")

# ============================================
# 7. SAVE FEATURE ENGINEERED DATASET
# ============================================

df.to_csv(
    "feature_engineered_titanic.csv",
    index=False
)

print("\n================================")
print("FEATURE ENGINEERING COMPLETED")
print("================================")

print("\nNew Columns Added:")
print([
    "family_size",
    "is_alone",
    "age_group",
    "fare_category",
    "title"
])

print("\nFeature Engineered Dataset Saved!")
print(df.head())

