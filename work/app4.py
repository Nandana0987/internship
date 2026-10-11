from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

app = Flask(__name__)

# =========================
# LOAD DATA + TRAIN MODEL
# =========================
df = pd.read_csv("work/Housing.csv")
df = pd.get_dummies(df, drop_first=True)

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return render_template("index1.html")

@app.route('/predict', methods=['POST'])
def predict():

    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    stories = int(request.form['stories'])
    parking = int(request.form['parking'])

    mainroad = request.form['mainroad']
    guestroom = request.form['guestroom']
    basement = request.form['basement']
    hotwaterheating = request.form['hotwaterheating']
    airconditioning = request.form['airconditioning']
    prefarea = request.form['prefarea']
    furnishingstatus = request.form['furnishingstatus']

    input_data = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'parking': parking,

        'mainroad_yes': 1 if mainroad == 'yes' else 0,
        'guestroom_yes': 1 if guestroom == 'yes' else 0,
        'basement_yes': 1 if basement == 'yes' else 0,
        'hotwaterheating_yes': 1 if hotwaterheating == 'yes' else 0,
        'airconditioning_yes': 1 if airconditioning == 'yes' else 0,
        'prefarea_yes': 1 if prefarea == 'yes' else 0,

        'furnishingstatus_semi-furnished': 1 if furnishingstatus == 'semi-furnished' else 0,
        'furnishingstatus_unfurnished': 1 if furnishingstatus == 'unfurnished' else 0
    }

    input_df = pd.DataFrame([input_data])

    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[X.columns]

    prediction = model.predict(input_df)[0]

    return render_template("index1.html", prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)