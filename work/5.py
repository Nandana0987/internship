from flask import Flask, render_template, request
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

app = Flask(__name__)

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("work/Food_Delivery_Times.csv")

# Remove ID column if present
if "Order_ID" in df.columns:
    df.drop("Order_ID", axis=1, inplace=True)

# Handle missing values
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in df.select_dtypes(exclude='object').columns:
    df[col].fillna(df[col].median(), inplace=True)

# Target Column
y = df["Delivery_Time_min"]

X = df.drop("Delivery_Time_min", axis=1)

# Categorical Columns
cat_cols = X.select_dtypes(include='object').columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="passthrough"
)

# Model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

accuracy = round(r2_score(y_test, preds) * 100, 2)

# ==========================
# ROUTES
# ==========================

@app.route("/")
def home():
    return render_template(
        "index3.html",
        accuracy=accuracy
    )


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "Distance_km": [float(request.form["Distance_km"])],
        "Weather": [request.form["Weather"]],
        "Traffic_Level": [request.form["Traffic_Level"]],
        "Time_of_Day": [request.form["Time_of_Day"]],
        "Vehicle_Type": [request.form["Vehicle_Type"]],
        "Preparation_Time_min": [float(request.form["Preparation_Time_min"])],
        "Courier_Experience_yrs": [float(request.form["Courier_Experience_yrs"])]
    }

    input_df = pd.DataFrame(data)

    prediction = round(
        model.predict(input_df)[0],
        2
    )

    if prediction <= 30:
        status = "⚡ Fast Delivery"

    elif prediction <= 60:
        status = "🚚 Normal Delivery"

    else:
        status = "⏳ Delayed Delivery"

    return render_template(
        "index3.html",
        prediction=prediction,
        status=status,
        accuracy=accuracy
    )


if __name__ == "__main__":
    app.run(debug=True)