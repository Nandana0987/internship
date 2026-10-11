import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("1/sales_data.csv")



# Drop target column
x = df.drop("Sales_Amount", axis=1)



y = df["Sales_Amount"]



x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


print("================================")
print("TRAIN TEST SPLIT COMPLETED")
print("================================")

print("\nX Train Shape:")
print(x_train.shape)

print("\nX Test Shape:")
print(x_test.shape)

print("\nY Train Shape:")
print(y_train.shape)

print("\nY Test Shape:")
print(y_test.shape)


print("\n================================")
print("X TRAIN")
print("================================")
print(x_train.head())

print("\n================================")
print("Y TRAIN")
print("================================")
print(y_train.head())