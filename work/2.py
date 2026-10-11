# ==========================================
# CUSTOMER CHURN PREDICTION
# USER INPUT VERSION
# ==========================================

import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("work/customer_churn_dataset-testing-master.csv")

# Drop CustomerID
df.drop("CustomerID", axis=1, inplace=True)

# ==========================================
# ENCODE CATEGORICAL DATA
# ==========================================

gender_encoder = LabelEncoder()
subscription_encoder = LabelEncoder()
contract_encoder = LabelEncoder()

df["Gender"] = gender_encoder.fit_transform(df["Gender"])
df["Subscription Type"] = subscription_encoder.fit_transform(df["Subscription Type"])
df["Contract Length"] = contract_encoder.fit_transform(df["Contract Length"])

# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df.drop("Churn", axis=1)
y = df["Churn"]
feature_names = X.columns.tolist()

print(feature_names)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# TRAIN MODEL
# ==========================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ==========================================
# USER INPUT
# ==========================================

print("\n===== ENTER CUSTOMER DETAILS =====")

gender = input("Gender (Male/Female): ").strip().title()


age = int(input("Age: "))
tenure = int(input("Tenure: "))
usage_frequency = int(input("Usage Frequency: "))
support_calls = int(input("Support Calls: "))
payment_delay = int(input("Payment Delay: "))

subscription = input(
    "Subscription Type (Basic/Standard/Premium): "
).strip().title()

contract = input(
    "Contract Length (Monthly/Quarterly/Annual): "
).strip().title()

total_spend = float(input("Total Spend: "))
last_interaction = int(input("Last Interaction: "))

# ==========================================
# ENCODE USER INPUT
# ==========================================

gender = gender_encoder.transform([gender])[0]

subscription = subscription_encoder.transform(
    [subscription]
)[0]

contract = contract_encoder.transform(
    [contract]
)[0]

# ==========================================
# CREATE INPUT DATAFRAME
# ==========================================

user_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Tenure": [tenure],
    "Usage Frequency": [usage_frequency],
    "Support Calls": [support_calls],
    "Payment Delay": [payment_delay],
    "Subscription Type": [subscription],
    "Contract Length": [contract],
    "Total Spend": [total_spend],
    "Last Interaction": [last_interaction]
})

# ==========================================
# SCALE INPUT
# ==========================================

user_data = scaler.transform(user_data)

# ==========================================
# PREDICT
# ==========================================

prediction = model.predict(user_data)

probability = model.predict_proba(user_data)

# ==========================================
# RESULT
# ==========================================

print("\n===== PREDICTION RESULT =====")

if prediction[0] == 1:
    print("⚠ Customer Will Churn")
else:
    print("✅ Customer Will Stay")

print(
    f"Churn Probability: {probability[0][1]*100:.2f}%"
)