import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("work/car_purchasing.csv", encoding="latin1")

# Drop unnecessary columns
df.drop(['customer name',
         'customer e-mail',
         'country'], axis=1, inplace=True)

# Features and target
X = df.drop('car purchase amount', axis=1)
y = df['car purchase amount']

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

# Save model
pickle.dump(model, open("car_model.pkl", "wb"))

print("Model Saved!")

