import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("logistic regression/disease_diagnosis.csv")

# =====================================
# SPLIT BLOOD PRESSURE
# =====================================

bp = df["Blood_Pressure_mmHg"].str.split("/", expand=True)

df["Systolic_BP"] = bp[0].astype(int)
df["Diastolic_BP"] = bp[1].astype(int)

df.drop(["Blood_Pressure_mmHg"], axis=1, inplace=True)

# =====================================
# FEATURES & TARGET
# =====================================

X = df.drop(
    ["Patient_ID", "Diagnosis", "Severity", "Treatment_Plan"],
    axis=1
)

y = df["Diagnosis"]

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# =====================================
# CATEGORICAL & NUMERIC COLUMNS
# =====================================

categorical_cols = [
    "Gender",
    "Symptom_1",
    "Symptom_2",
    "Symptom_3"
]

numeric_cols = [
    "Age",
    "Heart_Rate_bpm",
    "Body_Temperature_C",
    "Oxygen_Saturation_%",
    "Systolic_BP",
    "Diastolic_BP"
]

# =====================================
# PREPROCESSING
# =====================================

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# =====================================
# MODEL PIPELINE
# =====================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=5000))
])

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# TRAIN MODEL
# =====================================

model.fit(X_train, y_train)

# =====================================
# EVALUATION
# =====================================

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

# =====================================
# USER INPUT
# =====================================

print("\n===== DISEASE PREDICTION =====")

age = int(input("Age: "))
gender = input("Gender (Male/Female): ")

symptom1 = input("Symptom 1: ")
symptom2 = input("Symptom 2: ")
symptom3 = input("Symptom 3: ")

heart_rate = int(input("Heart Rate: "))
temperature = float(input("Body Temperature: "))
oxygen = int(input("Oxygen Saturation: "))

systolic = int(input("Systolic BP: "))
diastolic = int(input("Diastolic BP: "))

user_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Symptom_1": [symptom1],
    "Symptom_2": [symptom2],
    "Symptom_3": [symptom3],
    "Heart_Rate_bpm": [heart_rate],
    "Body_Temperature_C": [temperature],
    "Oxygen_Saturation_%": [oxygen],
    "Systolic_BP": [systolic],
    "Diastolic_BP": [diastolic]
})

prediction = model.predict(user_data)

predicted_disease = label_encoder.inverse_transform(prediction)

print("\nPredicted Disease:", predicted_disease[0])