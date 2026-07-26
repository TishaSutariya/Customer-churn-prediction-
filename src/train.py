# ============================================
# train.py - Train the model
# ============================================

from pathlib import Path

print("=" * 60)
print("THIS TRAIN.PY IS RUNNING")
print(Path(__file__).resolve())
print("=" * 60)


from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ============================================
# Project Paths
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_churn_data.csv"
MODEL_DIR = BASE_DIR / "models" / "trained"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# Load Data
# ============================================

print("=" * 60)
print("TRAINING MODEL")
print("=" * 60)

print("\n1. Loading cleaned data...")

print(f"Loading from: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print(f"Loaded {len(df)} customers")

# ============================================
# Features & Target
# ============================================

X = df.drop(columns=["Churn"])
y = df["Churn"]

print(f"Features: {X.shape[1]}")
print(f"Customers: {X.shape[0]}")
print(f"Churn Rate: {y.mean():.2%}")

# ============================================
# Train Test Split
# ============================================

print("\n2. Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ============================================
# Feature Scaling
# ============================================

print("\n3. Scaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# Train Model
# ============================================

print("\n4. Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train_scaled, y_train)

print("Model trained successfully!")

# ============================================
# Predictions
# ============================================

print("\n5. Evaluating model...")

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall   : {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 Score : {f1:.4f} ({f1*100:.2f}%)")
print(f"ROC AUC  : {roc_auc:.4f} ({roc_auc*100:.2f}%)")

# ============================================
# Save Model
# ============================================

print("\n6. Saving model...")

joblib.dump(model, MODEL_DIR / "model.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

print(f"Model saved to: {MODEL_DIR}")

# ============================================
# Feature Importance
# ============================================

importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": model.feature_importances_,
    }
).sort_values(by="Importance", ascending=False)

importance.to_csv(
    REPORT_DIR / "feature_importance.csv",
    index=False,
)

print("\nTop 10 Important Features\n")
print(importance.head(10).to_string(index=False))

print("\nFeature importance saved.")

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)