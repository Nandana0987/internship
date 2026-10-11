import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv("processes2.csv")

print("Dataset Shape:", df.shape)

# =====================================
# DATA CLEANING
# =====================================

# Drop unwanted index column
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values if any
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])
# Standardize column names
df.columns = (
    df.columns
      .str.lower()
      .str.replace(" ", "_")
      .str.replace("(", "", regex=False)
      .str.replace(")", "", regex=False)
)

print("\nColumns:")
print(df.columns)

# =====================================
# FEATURES AND TARGET
# =====================================

target = "selling_price"

X = df.drop(target, axis=1)
y = df[target]

# =====================================
# IDENTIFY CATEGORICAL COLUMNS
# =====================================

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("\nCategorical Columns:")
print(categorical_cols)

# =====================================
# PREPROCESSOR
# =====================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        )
    ],
    remainder="passthrough"
)

# =====================================
# MODEL PIPELINE
# =====================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =====================================
# TRAIN MODEL
# =====================================

model.fit(X_train, y_train)

# =====================================
# PREDICTIONS
# =====================================

y_pred = model.predict(X_test)

# =====================================
# EVALUATION
# =====================================

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance")
print("-" * 30)
print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}")


plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()

# =====================================
# SAMPLE PREDICTIONS
# =====================================

# =====================================
# USER INPUT FOR PREDICTION
# =====================================

print("\nEnter Car Details:")

name = input("Car Name: ")
year = int(input("Year: "))
km_driven = int(input("KM Driven: "))
fuel = input("Fuel Type (Petrol/Diesel/CNG/LPG/Electric): ")
seller_type = input("Seller Type (Dealer/Individual/Trustmark Dealer): ")
transmission = input("Transmission (Manual/Automatic): ")
owner = input("Owner (First Owner/Second Owner/Third Owner/Fourth & Above Owner/Test Drive Car): ")
seats = int(input("Seats: "))
max_power_in_bph = float(input("Max Power (BHP): "))
mileage_unit = input("Mileage Unit (kmpl/kmkg): ")
mileage = float(input("Mileage: "))
engine_cc = int(input("Engine CC: "))

# Create DataFrame
user_data = pd.DataFrame({
    'name': [name],
    'year': [year],
    'km_driven': [km_driven],
    'fuel': [fuel],
    'seller_type': [seller_type],
    'transmission': [transmission],
    'owner': [owner],
    'seats': [seats],
    'max_power_in_bph': [max_power_in_bph],
    'mileage_unit': [mileage_unit],
    'mileage': [mileage],
    'engine_cc': [engine_cc]
})

# Predict
predicted_price = model.predict(user_data)

print("\nPredicted Selling Price:")
print(f"₹ {predicted_price[0]:,.2f}")


# # =====================================
# # RESIDUAL PLOT
# # =====================================

# residuals = y_test - y_pred

# plt.figure(figsize=(8,6))
# plt.scatter(y_pred, residuals)
# plt.axhline(y=0)
# plt.xlabel("Predicted Values")
# plt.ylabel("Residuals")
# plt.title("Residual Plot")
# plt.grid(True)
# plt.show()

# =====================================
