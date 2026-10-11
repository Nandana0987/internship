from flask import Flask, render_template, request
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("work/WELFake_Dataset.csv")

# Remove unwanted column
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Remove missing values
df.dropna(inplace=True)

# Combine title and text
df["content"] = df["title"] + " " + df["text"]

X = df["content"]
y = df["label"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_df=0.7
    )),
    ("lr", LogisticRegression(
        max_iter=1000
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = round(
    accuracy_score(y_test, y_pred) * 100,
    2
)

# ==========================
# ROUTES
# ==========================

@app.route("/")
def home():
    return render_template(
        "index4.html",
        accuracy=accuracy
    )

@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    prediction = model.predict([news])[0]

    probabilities = model.predict_proba([news])[0]

    confidence = round(
        max(probabilities) * 100,
        2
    )

    if prediction == 1:
        result = "✅ REAL NEWS"
    else:
        result = "❌ FAKE NEWS"

    return render_template(
        "index4.html",
        prediction=result,
        confidence=confidence,
        accuracy=accuracy
    )

if __name__ == "__main__":
    app.run(debug=True)