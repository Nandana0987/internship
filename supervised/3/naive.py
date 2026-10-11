import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("spam_ham_dataset.csv")

print("Dataset Shape:", df.shape)

# ==========================
# DATA CLEANING
# ==========================

if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# ==========================
# FEATURES AND TARGET
# ==========================

X = df["text"]
y = df["label"]

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# NAIVE BAYES MODEL
# ==========================

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("naive_bayes", MultinomialNB())
])

# ==========================
# TRAIN MODEL
# ==========================

model.fit(X_train, y_train)

# ==========================
# PREDICTIONS
# ==========================

y_pred = model.predict(X_test)

# ==========================
# EVALUATION
# ==========================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ==========================
# CONFUSION MATRIX PLOT
# ==========================

plt.figure(figsize=(6,5))
plt.imshow(cm)

plt.title("Confusion Matrix - Naive Bayes")
plt.colorbar()

plt.xticks([0,1], ["Ham","Spam"])
plt.yticks([0,1], ["Ham","Spam"])

plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i,j]),
                 ha='center',
                 va='center')

plt.show()

# ==========================
# SPAM vs HAM BAR CHART
# ==========================

plt.figure(figsize=(6,4))
df['label'].value_counts().plot(kind='bar')

plt.title("Spam vs Ham Messages")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# ==========================
# USER INPUT PREDICTION
# ==========================

while True:

    message = input("\nEnter Message (or type quit): ")

    if message.lower() == "quit":
        break

    prediction = model.predict([message])[0]

    print("\nPrediction:", prediction.upper())

    if prediction == "spam":
        print("⚠ This message is SPAM")
    else:
        print("✓ This message is HAM (Not Spam)")