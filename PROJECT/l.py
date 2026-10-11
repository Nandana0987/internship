import pandas as pd

df = pd.read_csv("student_performance.csv")

print("Resources:", df["Resources"].unique())
print("Extracurricular:", df["Extracurricular"].unique())
print("Motivation:", df["Motivation"].unique())
print("Internet:", df["Internet"].unique())
print("Gender:", df["Gender"].unique())
print("LearningStyle:", df["LearningStyle"].unique())
print("EduTech:", df["EduTech"].unique())
print("FinalGrade:", df["FinalGrade"].unique())