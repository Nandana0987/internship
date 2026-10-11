from flask import Flask, render_template, request
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("customer_churn_dataset-testing-master.csv")
df.drop("CustomerID", axis=1, inplace=True)

# =========================
# ENCODERS
# =========================
gender_encoder = LabelEncoder()
subscription_encoder = LabelEncoder()
contract_encoder = LabelEncoder()

df["Gender"] = gender_encoder.fit_transform(df["Gender"])
df["Subscription Type"] = subscription_encoder.fit_transform(df["Subscription Type"])
df["Contract Length"] = contract_encoder.fit_transform(df["Contract Length"])

# =========================
# FEATURES
# =========================
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# =========================
# MODEL
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =========================
# HOME
# =========================
@app.route("/")
def home():

    return render_template(
        "index.html",
        gender_options=gender_encoder.classes_,
        subscription_options=subscription_encoder.classes_,
        contract_options=contract_encoder.classes_
    )

# =========================
# PREDICT
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    gender = gender_encoder.transform([request.form["gender"]])[0]
    subscription = subscription_encoder.transform([request.form["subscription"]])[0]
    contract = contract_encoder.transform([request.form["contract"]])[0]

    age = int(request.form["age"])
    tenure = int(request.form["tenure"])
    usage = int(request.form["usage"])
    support = int(request.form["support"])
    delay = int(request.form["delay"])
    spend = float(request.form["spend"])
    interaction = int(request.form["interaction"])

    user_data = pd.DataFrame([[
        age, gender, tenure, usage, support,
        delay, subscription, contract,
        spend, interaction
    ]], columns=X.columns)

    user_data = scaler.transform(user_data)

    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1] * 100

    result = "Customer Likely To Churn" if prediction == 1 else "Customer Likely To Stay"

    return render_template(
        "index.html",
        prediction=result,
        probability=round(probability, 2),
        gender_options=gender_encoder.classes_,
        subscription_options=subscription_encoder.classes_,
        contract_options=contract_encoder.classes_
    )

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)