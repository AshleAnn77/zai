import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from xgboost import XGBClassifier

# -------------------------------
# Define Project Paths
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed_paysim.csv"
MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_columns.pkl"

# -------------------------------
# Load Dataset
# -------------------------------

df = pd.read_csv(DATA_PATH)

# Use sample while developing
df = df.sample(n=500000, random_state=42)

# -------------------------------
# Create Features and Target
# -------------------------------

y = df["isFraud"]

X = df.drop(
    columns=[
        "isFraud",
        "isFlaggedFraud",
        "nameOrig",
        "nameDest"
    ]
)

# Save feature order
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, FEATURES_PATH)

# -------------------------------
# Split Dataset
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# Train XGBoost Model
# -------------------------------

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

print("Model training completed!")

# -------------------------------
# Evaluate Model
# -------------------------------

y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_pred))

# -------------------------------
# Save Model
# -------------------------------

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")