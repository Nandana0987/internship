import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ------------------------------------
# PAGE CONFIG
# ------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------
# LOAD DATA
# ------------------------------------

df = pd.read_csv("customer_churn_dataset-testing-master.csv")

df.drop("CustomerID", axis=1, inplace=True)

# ------------------------------------
# ENCODING
# ------------------------------------

gender_encoder = LabelEncoder()
subscription_encoder = LabelEncoder()
contract_encoder = LabelEncoder()

df["Gender"] = gender_encoder.fit_transform(df["Gender"])
df["Subscription Type"] = subscription_encoder.fit_transform(df["Subscription Type"])
df["Contract Length"] = contract_encoder.fit_transform(df["Contract Length"])

# ------------------------------------
# TRAIN MODEL
# ------------------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ------------------------------------
# TITLE
# ------------------------------------

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("---")

# ------------------------------------
# SIDEBAR INPUTS
# ------------------------------------

st.sidebar.header("Customer Details")

gender_text = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

tenure = st.sidebar.number_input(
    "Tenure",
    min_value=0,
    value=12
)

usage_frequency = st.sidebar.number_input(
    "Usage Frequency",
    min_value=0,
    value=15
)

support_calls = st.sidebar.number_input(
    "Support Calls",
    min_value=0,
    value=2
)

payment_delay = st.sidebar.number_input(
    "Payment Delay",
    min_value=0,
    value=5
)

subscription_text = st.sidebar.selectbox(
    "Subscription Type",
    ["Basic", "Standard", "Premium"]
)

contract_text = st.sidebar.selectbox(
    "Contract Length",
    ["Monthly", "Quarterly", "Annual"]
)

total_spend = st.sidebar.number_input(
    "Total Spend",
    min_value=0.0,
    value=500.0
)

last_interaction = st.sidebar.number_input(
    "Last Interaction",
    min_value=0,
    value=10
)

# ------------------------------------
# ENCODE INPUTS
# ------------------------------------

gender = gender_encoder.transform([gender_text])[0]

subscription = subscription_encoder.transform(
    [subscription_text]
)[0]

contract = contract_encoder.transform(
    [contract_text]
)[0]

# ------------------------------------
# PREDICT BUTTON
# ------------------------------------

if st.button("Predict Churn"):

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

    user_data = user_data[X.columns]

    user_data = scaler.transform(user_data)

    prediction = model.predict(user_data)

    probability = model.predict_proba(user_data)

    churn_prob = probability[0][1] * 100

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Information")

        st.write(f"**Gender:** {gender_text}")
        st.write(f"**Age:** {age}")
        st.write(f"**Tenure:** {tenure}")
        st.write(f"**Total Spend:** ${total_spend}")

    with col2:

        st.subheader("Prediction Result")

        if prediction[0] == 1:

            st.error("⚠ Customer Likely To Churn")

        else:

            st.success("✅ Customer Likely To Stay")

        st.metric(
            "Churn Probability",
            f"{churn_prob:.2f}%"
        )

# ------------------------------------
# DATASET OVERVIEW
# ------------------------------------

st.markdown("---")

st.subheader("Dataset Preview")

st.dataframe(df.head())

st.subheader("Churn Distribution")

st.bar_chart(df["Churn"].value_counts())