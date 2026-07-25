from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / "models" / "xgboost_model.pkl")
encoder = joblib.load(BASE_DIR / "models" / "label_encoder.pkl")
feature_columns = joblib.load(BASE_DIR / "models" / "feature_columns.pkl")

app = FastAPI(title="AI-FraudShield API")

# allow your dashboard (running on a different port) to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float

@app.post("/score-transaction")
def score_transaction(txn: Transaction):
    data = txn.dict()

    data["balance_change_sender"] = data["oldbalanceOrg"] - data["newbalanceOrig"]
    data["balance_change_receiver"] = data["newbalanceDest"] - data["oldbalanceDest"]
    data["high_amount_flag"] = int(data["amount"] > 200000)
    data["is_cashout_or_transfer"] = int(data["type"] in ["CASH_OUT", "TRANSFER"])
    data["zero_sender_balance_after"] = int(data["newbalanceOrig"] == 0)
    data["zero_receiver_balance_before"] = int(data["oldbalanceDest"] == 0)

    data["type"] = encoder.transform([data["type"]])[0]

    orig_id, dest_id = data.pop("nameOrig"), data.pop("nameDest")

    df = pd.DataFrame([data])[feature_columns]

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return {
        "nameOrig": orig_id,
        "nameDest": dest_id,
        "status": "FRAUD" if prediction == 1 else "LEGITIMATE",
        "fraud_probability": round(probability, 4),
        "risk_score": round(probability * 100, 1),
    }

@app.get("/health")
def health():
    return {"status": "ok"}