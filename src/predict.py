# ============================================
# predict.py - Predict churn for a customer
# ============================================

from pathlib import Path
import pandas as pd
import joblib

# ============================================
# Project Paths
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models" / "trained"
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_churn_data.csv"

print("=" * 60)
print("CUSTOMER CHURN PREDICTION")
print("=" * 60)

# ============================================
# Load Model & Scaler
# ============================================

print("Loading model...")

model = joblib.load(MODEL_DIR / "model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

print("Model loaded successfully!")

# ============================================
# Load Feature Names
# ============================================

df_clean = pd.read_csv(DATA_PATH)

feature_cols = df_clean.drop(columns=["Churn"]).columns.tolist()

print(f"Model has {len(feature_cols)} features")

# ============================================
# Example Customer
# ============================================

customer = {
    "gender": 0,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 1,
    "PhoneService": 0,
    "MultipleLines": 0,
    "InternetService": 1,
    "OnlineSecurity": 0,
    "OnlineBackup": 1,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 0,
    "StreamingMovies": 0,
    "Contract": 0,
    "PaperlessBilling": 1,
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
    "PaymentMethod_Credit card (automatic)": 0,
    "PaymentMethod_Electronic check": 1,
    "PaymentMethod_Mailed check": 0,
    "PaymentMethod_Bank transfer (automatic)": 0
}

# ============================================
# Create DataFrame
# ============================================

df = pd.DataFrame([customer])

# Add any missing columns
for col in feature_cols:
    if col not in df.columns:
        df[col] = 0

# Keep only training columns in correct order
df = df[feature_cols]

# ============================================
# Scale Features
# ============================================

df_scaled = scaler.transform(df)

# ============================================
# Prediction
# ============================================

prediction = model.predict(df_scaled)[0]
probability = model.predict_proba(df_scaled)[0][1]

# ============================================
# Risk Level
# ============================================

if probability < 0.30:
    risk = "Low"
elif probability < 0.60:
    risk = "Medium"
else:
    risk = "High"

# ============================================
# Display Result
# ============================================

print("\n" + "=" * 50)
print("PREDICTION RESULT")
print("=" * 50)

print(f"Churn Probability : {probability:.2%}")
print(f"Risk Level        : {risk}")
print(f"Prediction        : {'Will Churn' if prediction == 1 else 'Will Stay'}")

print("=" * 50)