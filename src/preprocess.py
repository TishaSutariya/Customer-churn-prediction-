# ============================================
# preprocess.py - Clean the data
# ============================================
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"

print("Loading from:", DATA_PATH)

df = pd.read_csv(DATA_PATH)

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.loc[(df["tenure"] == 0) & (df["TotalCharges"].isnull()), "TotalCharges"] = df["MonthlyCharges"]
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Convert Churn to 0 and 1
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
print(f"Churn rate: {df['Churn'].mean():.2%}")

# Convert all Yes/No to 1/0
df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
df["Partner"] = df["Partner"].map({"No": 0, "Yes": 1})
df["Dependents"] = df["Dependents"].map({"No": 0, "Yes": 1})
df["PhoneService"] = df["PhoneService"].map({"No": 0, "Yes": 1})
df["PaperlessBilling"] = df["PaperlessBilling"].map({"No": 0, "Yes": 1})

# Service columns
df["MultipleLines"] = df["MultipleLines"].map({"Yes": 1, "No": 0, "No phone service": 0})
df["OnlineSecurity"] = df["OnlineSecurity"].map({"Yes": 1, "No": 0, "No internet service": 0})
df["OnlineBackup"] = df["OnlineBackup"].map({"Yes": 1, "No": 0, "No internet service": 0})
df["DeviceProtection"] = df["DeviceProtection"].map({"Yes": 1, "No": 0, "No internet service": 0})
df["TechSupport"] = df["TechSupport"].map({"Yes": 1, "No": 0, "No internet service": 0})
df["StreamingTV"] = df["StreamingTV"].map({"Yes": 1, "No": 0, "No internet service": 0})
df["StreamingMovies"] = df["StreamingMovies"].map({"Yes": 1, "No": 0, "No internet service": 0})

# Other categories
df["InternetService"] = df["InternetService"].map({"No": 0, "DSL": 1, "Fiber optic": 2})
df["Contract"] = df["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2})

# PaymentMethod (make separate columns)
df = pd.get_dummies(df, columns=["PaymentMethod"], drop_first=True)

# Remove customerID
df = df.drop(columns=["customerID"])

# Save cleaned data
import os
processed_dir = BASE_DIR / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(processed_dir / "cleaned_churn_data.csv", index=False)
print("Saved cleaned data!")
print(f"Final shape: {df.shape}")