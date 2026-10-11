import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("sales_data.csv")

# Convert date column to datetime
df['Sale_Date'] = pd.to_datetime(df['Sale_Date'])

# Aggregate daily sales
daily_sales = df.groupby('Sale_Date')['Sales_Amount'].sum().reset_index()

# Convert dates to numerical values
daily_sales['Days'] = (
    daily_sales['Sale_Date'] - daily_sales['Sale_Date'].min()
).dt.days

# Features and target
X = daily_sales[['Days']]
y = daily_sales['Sales_Amount']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=False
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# Plot Actual vs Predicted
plt.figure(figsize=(12,6))
plt.plot(daily_sales['Sale_Date'].iloc[len(X_train):],
         y_test,
         label='Actual Sales')

plt.plot(daily_sales['Sale_Date'].iloc[len(X_train):],
         y_pred,
         label='Predicted Sales')

plt.title("Sales Trend Analysis Using Linear Regression")
plt.xlabel("Date")
plt.ylabel("Sales Amount")
plt.legend()
plt.grid(True)
plt.show()