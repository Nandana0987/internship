from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

app = Flask(__name__)

# Load Dataset
df = pd.read_csv("work/StudentPerformanceFactors.csv")

X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]

categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(exclude=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = round(r2_score(y_test, preds) * 100, 2)

@app.route('/')
def home():
    return render_template("index2.html", accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():

    data = {
        'Hours_Studied':[int(request.form['Hours_Studied'])],
        'Attendance':[int(request.form['Attendance'])],
        'Parental_Involvement':[request.form['Parental_Involvement']],
        'Access_to_Resources':[request.form['Access_to_Resources']],
        'Extracurricular_Activities':[request.form['Extracurricular_Activities']],
        'Sleep_Hours':[int(request.form['Sleep_Hours'])],
        'Previous_Scores':[int(request.form['Previous_Scores'])],
        'Motivation_Level':[request.form['Motivation_Level']],
        'Internet_Access':[request.form['Internet_Access']],
        'Tutoring_Sessions':[int(request.form['Tutoring_Sessions'])],
        'Family_Income':[request.form['Family_Income']],
        'Teacher_Quality':[request.form['Teacher_Quality']],
        'School_Type':[request.form['School_Type']],
        'Peer_Influence':[request.form['Peer_Influence']],
        'Physical_Activity':[int(request.form['Physical_Activity'])],
        'Learning_Disabilities':[request.form['Learning_Disabilities']],
        'Parental_Education_Level':[request.form['Parental_Education_Level']],
        'Distance_from_Home':[request.form['Distance_from_Home']],
        'Gender':[request.form['Gender']]
    }

    input_df = pd.DataFrame(data)

    prediction = round(model.predict(input_df)[0],2)

    return render_template(
        "index2.html",
        prediction=prediction,
        accuracy=accuracy
    )

if __name__ == "__main__":
    app.run(debug=True)