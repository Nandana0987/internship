import streamlit as st
import pandas as pd
import pickle

# Page Config
st.set_page_config(
    page_title="Car Purchase Prediction",
    page_icon="🚗",
    layout="wide"
)

# Load Model
model = pickle.load(open("car_model.pkl", "rb"))

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:35px !important;
    font-weight:bold;
    text-align:center;
    color:#4CAF50;
}

.prediction-box {
    background-color:#1E1E1E;
    padding:20px;
    border-radius:15px;
    text-align:center;
}

.metric-card {
    background:#262730;
    padding:20px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    "<p class='big-font'>🚗 Car Purchase Amount Prediction Dashboard</p>",
    unsafe_allow_html=True
)

st.write("---")

# Sidebar
st.sidebar.header("Enter Customer Details")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

gender_value = 1 if gender == "Male" else 0

age = st.sidebar.slider(
    "Age",
    18,
    80,
    35
)

salary = st.sidebar.number_input(
    "Annual Salary",
    min_value=1000.0,
    value=50000.0
)

debt = st.sidebar.number_input(
    "Credit Card Debt",
    min_value=0.0,
    value=5000.0
)

net_worth = st.sidebar.number_input(
    "Net Worth",
    min_value=0.0,
    value=100000.0
)

# Layout
col1, col2 = st.columns(2)

with col1:
    st.metric("Age", age)

with col2:
    st.metric("Annual Salary", f"${salary:,.0f}")

st.write("---")

if st.button("Predict Purchase Amount"):

    input_data = pd.DataFrame({
        "gender":[gender_value],
        "age":[age],
        "annual Salary":[salary],
        "credit card debt":[debt],
        "net worth":[net_worth]
    })

    prediction = model.predict(input_data)

    st.markdown(
        f"""
        <div class='prediction-box'>
        <h2>Predicted Car Purchase Amount</h2>
        <h1 style='color:#4CAF50'>
        ${prediction[0]:,.2f}
        </h1>
        </div>
        """,
        unsafe_allow_html=True
    )