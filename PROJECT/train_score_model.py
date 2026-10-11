import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("student_performance.csv")

# Encode categorical columns
categorical_cols = [
    "Resources",
    "Extracurricular",
    "Motivation",
    "Internet",
    "Gender",
    "LearningStyle",
    "EduTech",
    "FinalGrade"
]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features
X = df[
[
    "StudyHours",
    "Attendance",
    "Resources",
    "Extracurricular",
    "Motivation",
    "Internet",
    "Gender",
    "Age",
    "LearningStyle",
    "OnlineCourses",
    "Discussions",
    "AssignmentCompletion",
    "EduTech",
    "StressLevel"
]
]

# Target
y = df["ExamScore"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open("score_model.pkl", "wb"))

# Save encoders
pickle.dump(encoders, open("encoders.pkl", "wb"))

print("Model trained successfully.")