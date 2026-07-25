import pandas as pd
import joblib
from pathlib import Path

# -------------------------------
# Define Project Paths
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_columns.pkl"

# -------------------------------
# Load Model and Supporting Files
# -------------------------------

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
feature_columns = joblib.load(FEATURES_PATH)

# -------------------------------
# Sample Transaction
# -------------------------------

sample = {
    "step": 300,
    "type": "TRANSFER",
    "amount": 1000000,
    "nameOrig": "C111111111",
    "oldbalanceOrg": 1000000,
    "newbalanceOrig": 0,
    "nameDest": "C222222222",
    "oldbalanceDest": 0,
    "newbalanceDest": 1000000
}
# -------------------------------
# Feature Engineering
# -------------------------------

sample["balance_change_sender"] = (
    sample["oldbalanceOrg"] - sample["newbalanceOrig"]
)

sample["balance_change_receiver"] = (
    sample["newbalanceDest"] - sample["oldbalanceDest"]
)

sample["high_amount_flag"] = int(sample["amount"] > 200000)

sample["is_cashout_or_transfer"] = int(
    sample["type"] in ["CASH_OUT", "TRANSFER"]
)

sample["zero_sender_balance_after"] = int(
    sample["newbalanceOrig"] == 0
)

sample["zero_receiver_balance_before"] = int(
    sample["oldbalanceDest"] == 0
)

# -------------------------------
# Encode Transaction Type
# -------------------------------

sample["type"] = encoder.transform([sample["type"]])[0]

# -------------------------------
# Remove Unused Columns
# -------------------------------

sample.pop("nameOrig")
sample.pop("nameDest")

# -------------------------------
# Convert to DataFrame
# -------------------------------

sample_df = pd.DataFrame([sample])

# Arrange columns in same order as training
sample_df = sample_df[feature_columns]

# -------------------------------
# Prediction
# -------------------------------

prediction = model.predict(sample_df)[0]
probability = model.predict_proba(sample_df)[0][1]

# -------------------------------
# Display Result
# -------------------------------

print("\nPrediction Result")
print("----------------------------")

if prediction == 1:
    print("Transaction Status : FRAUD")
else:
    print("Transaction Status : LEGITIMATE")

print(f"Fraud Probability : {probability:.2%}")