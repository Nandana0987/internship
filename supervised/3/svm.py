import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
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
print(df.head())

# ==========================
# DATA CLEANING
# ==========================

# Remove unwanted index column
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Remove missing values
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
# SVM PIPELINE
# ==========================

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("svm", SVC(kernel="linear"))
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

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

plt.imshow(cm)

plt.title("Confusion Matrix")
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

plt.figure(figsize=(5,4))

plt.bar(["SVM Accuracy"], [accuracy*100])

plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy")

plt.ylim(0,100)

plt.show()
# ==========================
# USER INPUT PREDICTION
# ==========================

while True:

    message = input("\nEnter Email Message (or type quit): ")

    if message.lower() == "quit":
        break

    prediction = model.predict([message])[0]

    print("\nPrediction:", prediction.upper())

    if prediction == "spam":
        print("⚠ This message is SPAM")
    else:
        print("✓ This message is HAM (Not Spam)")

