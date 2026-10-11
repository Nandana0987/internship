# =====================================================
# APRIORI ALGORITHM ON ONLINE RETAIL DATASET
# =====================================================

# Step 1: Import Libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# -----------------------------------------------------
# Step 2: Load Dataset
# -----------------------------------------------------
df = pd.read_csv(
    "unsupervised/Assignment-1_Data.csv",
    sep=";",
    encoding="utf-8-sig",
    low_memory=False
)

# Display first few rows
print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# -----------------------------------------------------
# Step 3: Data Cleaning
# -----------------------------------------------------
df.columns = df.columns.str.replace('ï»¿', '')
# Remove missing product names
df = df.dropna(subset=['Itemname'])

# Remove cancelled transactions
df = df[~df['BillNo'].astype(str).str.startswith('C')]

# Convert Quantity to numeric
df['Quantity'] = pd.to_numeric(df['Quantity'])

# Keep only positive quantities
df = df[df['Quantity'] > 0]

# -----------------------------------------------------
# Step 4: Create Basket Matrix
# -----------------------------------------------------

# Create basket matrix
basket = (
    df.groupby(['BillNo', 'Itemname'])['Quantity']
      .sum()
      .unstack()
      .fillna(0)
)

# Convert to Boolean values
basket = basket > 0

print(basket.head())

# Apply Apriori
frequent_itemsets = apriori(
    basket,
    min_support=0.02,
    use_colnames=True
)
print("\nFrequent Itemsets:")
print(frequent_itemsets.head(10))

# -----------------------------------------------------
# Step 7: Generate Association Rules
# -----------------------------------------------------

rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1
)

# -----------------------------------------------------
# Step 8: Sort Rules by Lift
# -----------------------------------------------------

rules = rules.sort_values(
    by='lift',
    ascending=False
)

# -----------------------------------------------------
# Step 9: Display Top Rules
# -----------------------------------------------------

print("\nTop 20 Association Rules:")
print(
    rules[
        [
            'antecedents',
            'consequents',
            'support',
            'confidence',
            'lift'
        ]
    ].head(20)
)

# -----------------------------------------------------
# Step 10: Save Results
# -----------------------------------------------------

rules.to_csv(
    "apriori_rules.csv",
    index=False
)

print("\nAssociation Rules Saved Successfully!")

# -----------------------------------------------------
# Step 11: Top 10 Rules Only
# -----------------------------------------------------

top_rules = rules[
    [
        'antecedents',
        'consequents',
        'support',
        'confidence',
        'lift'
    ]
].head(10)

print("\nTop 10 Rules:")
print(top_rules)

# -----------------------------------------------------
# Step 12: Rule Interpretation Example
# -----------------------------------------------------

for index, row in top_rules.iterrows():
    print("\nRule:")
    print(f"{set(row['antecedents'])} --> {set(row['consequents'])}")
    print(f"Support   : {row['support']:.4f}")
    print(f"Confidence: {row['confidence']:.4f}")
    print(f"Lift      : {row['lift']:.4f}")