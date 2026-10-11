from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    text = request.form["comment"]

    vec = vectorizer.transform([text])

    prediction = model.predict(vec)[0]

    if prediction == "positive":
        emoji = "😊"
        result = "Positive Sentiment"

    elif prediction == "negative":
        emoji = "😡"
        result = "Negative Sentiment"

    else:
        emoji = "😐"
        result = "Neutral Sentiment"

    return render_template(
        "index.html",
        result=result,
        emoji=emoji,
        text=text
    )

if __name__ == "__main__":
    app.run(debug=True)