import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Load Dataset
# --------------------------
df = pd.read_csv("2/disease_diagnosis.csv")

# Style
sns.set_style("whitegrid")

# --------------------------
# 1. HEATMAP
# --------------------------

plt.figure(figsize=(10,6))

numeric_df = df.select_dtypes(include=["number"])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# --------------------------
# 2. HISTOGRAM
# --------------------------

plt.figure(figsize=(8,5))

plt.hist(
    df["Age"],
    bins=20
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# --------------------------
# 3. BARPLOT
# --------------------------

plt.figure(figsize=(8,5))

sns.barplot(
    x="Severity",
    y="Heart_Rate_bpm",
    data=df
)

plt.title("Average Heart Rate by Severity")
plt.xticks(rotation=45)
plt.show()

# --------------------------
# 4. SCATTERPLOT
# --------------------------

plt.figure(figsize=(8,5))

sns.scatterplot(
    x="Age",
    y="Heart_Rate_bpm",
    hue="Diagnosis",
    data=df
)

plt.title("Age vs Heart Rate")
plt.show()

# --------------------------
# 5. COUNTPLOT
# --------------------------

plt.figure(figsize=(8,5))

sns.countplot(
    x="Diagnosis",
    data=df
)

plt.title("Diagnosis Count")
plt.xticks(rotation=45)
plt.show()

# --------------------------
# 6. BOXPLOT
# --------------------------

plt.figure(figsize=(8,5))

sns.boxplot(
    x="Severity",
    y="Body_Temperature_C",
    data=df
)

plt.title("Body Temperature by Severity")
plt.xticks(rotation=45)
plt.show()