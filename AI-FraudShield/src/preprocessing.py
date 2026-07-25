import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Define Project Paths
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "PaySim" / "PS_20174392719_1491204439457_log.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed_paysim.csv"

LABEL_ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"

# -------------------------------
# Load Dataset
# -------------------------------

df = pd.read_csv(DATA_PATH)

# -------------------------------
# Feature Engineering
# -------------------------------

# Feature 1: Sender Balance Change
df["balance_change_sender"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)

# Feature 2: Receiver Balance Change
df["balance_change_receiver"] = (
    df["newbalanceDest"] - df["oldbalanceDest"]
)

# Feature 3: High Amount Flag
df["high_amount_flag"] = (
    df["amount"] > 200000
).astype(int)

# Feature 4: Cash Out or Transfer
# Fraud mainly occurs in CASH_OUT and TRANSFER transactions.
df["is_cashout_or_transfer"] = (
    df["type"].isin(["CASH_OUT", "TRANSFER"])
).astype(int)

# Feature 5: Sender Balance becomes Zero
# Fraudulent transactions often empty the sender's account.
df["zero_sender_balance_after"] = (
    df["newbalanceOrig"] == 0
).astype(int)

# Feature 6: Receiver Initially Had Zero Balance
# Fraud may involve newly created or inactive destination accounts.
df["zero_receiver_balance_before"] = (
    df["oldbalanceDest"] == 0
).astype(int)

# -------------------------------
# Encode Categorical Column
# -------------------------------

encoder = LabelEncoder()

df["type"] = encoder.fit_transform(df["type"])

# Save Label Encoder
joblib.dump(encoder, LABEL_ENCODER_PATH)

# -------------------------------
# Save Feature List
# -------------------------------

feature_columns = df.drop(
    columns=["isFraud"]
).columns.tolist()

joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)

# -------------------------------
# Save Processed Dataset
# -------------------------------

df.to_csv(PROCESSED_DATA_PATH, index=False)

print("Preprocessing completed successfully!")