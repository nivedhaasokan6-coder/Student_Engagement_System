import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("student_engagement_dataset.csv")

# Create Low, Medium, High labels
score = (
    df["Login_Frequency"] +
    df["Participation_Forums"] +
    df["Assignment_Submissions"] +
    df["Resource_Access_Frequency"]
)

df["Engagement_Level"] = pd.qcut(
    score,
    q=3,
    labels=["Low", "Medium", "High"]
)

# Features
X = df[[
    "Login_Frequency",
    "Participation_Forums",
    "Assignment_Submissions",
    "Resource_Access_Frequency",
    "Quiz_Performance_Average"
]]

# Target
y = df["Engagement_Level"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save files
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model Trained Successfully")