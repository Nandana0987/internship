# ==========================================
# HOUSE PRICE PREDICTION USING USER INPUT
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("work/Housing.csv")

# ==========================================
# ENCODE CATEGORICAL COLUMNS
# ==========================================

df = pd.get_dummies(df, drop_first=True)

# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df.drop("price", axis=1)
y = df["price"]

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
# TRAIN MODEL
# ==========================================

model = LinearRegression()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-" * 30)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)


print("\n===== HOUSE PRICE PREDICTION =====")

# ==========================================
# USER INPUT
# ==========================================

area = float(input("Enter Area (sq ft): "))
bedrooms = int(input("Enter Number of Bedrooms: "))
bathrooms = int(input("Enter Number of Bathrooms: "))
stories = int(input("Enter Number of Stories: "))
parking = int(input("Enter Parking Spaces: "))

mainroad = input("Main Road Access (yes/no): ").lower()
guestroom = input("Guest Room (yes/no): ").lower()
basement = input("Basement (yes/no): ").lower()
hotwaterheating = input("Hot Water Heating (yes/no): ").lower()
airconditioning = input("Air Conditioning (yes/no): ").lower()
prefarea = input("Preferred Area (yes/no): ").lower()

furnishingstatus = input(
    "Furnishing Status (furnished/semi-furnished/unfurnished): "
).lower()

# ==========================================
# CREATE INPUT DATA
# ==========================================

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

# ==========================================
# MATCH TRAINING COLUMNS
# ==========================================

input_df = pd.DataFrame([input_data])

# Add missing columns
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns
input_df = input_df[X.columns]

# ==========================================
# PREDICT PRICE
# ==========================================

predicted_price = model.predict(input_df)

print("\n===================================")
print("Predicted House Price:", round(predicted_price[0], 2))
print("===================================")