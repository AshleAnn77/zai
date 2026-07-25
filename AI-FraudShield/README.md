# AI-FraudShield

## Overview

AI-FraudShield is a machine learning-based fraud detection system developed to identify suspicious financial transactions. The project uses the PaySim dataset and an XGBoost classifier to distinguish fraudulent transactions from legitimate ones through feature engineering and supervised learning.

---

## Project Structure

```
AI-FraudShield/
│
├── data/
│   ├── PaySim/                  # Original dataset (not tracked by Git)
│   └── processed_paysim.csv     # Processed dataset (ignored by Git)
│
├── models/
│   ├── xgboost_model.pkl
│   ├── feature_columns.pkl
│   └── label_encoder.pkl
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict_model.py
│
├── README.md
└── requirements.txt
```

---

## Dataset

Dataset Used: **PaySim Mobile Money Transaction Dataset**

The dataset contains simulated mobile money transactions with both legitimate and fraudulent transactions.

Original Dataset Columns:

- step
- type
- amount
- nameOrig
- oldbalanceOrg
- newbalanceOrig
- nameDest
- oldbalanceDest
- newbalanceDest
- isFraud
- isFlaggedFraud

---

## Feature Engineering

The following features were created to improve fraud detection:

| Feature | Description |
|---------|-------------|
| balance_change_sender | Difference between sender's old and new balance |
| balance_change_receiver | Difference between receiver's new and old balance |
| high_amount_flag | Indicates transactions greater than ₹200,000 |
| is_cashout_or_transfer | Flags CASH_OUT and TRANSFER transactions |
| zero_sender_balance_after | Checks if sender's balance becomes zero |
| zero_receiver_balance_before | Checks if receiver initially had zero balance |

---

## Model

Algorithm Used:

- XGBoost Classifier

The trained model is stored in:

```
models/xgboost_model.pkl
```

---

## Project Workflow

1. Load PaySim dataset
2. Perform Exploratory Data Analysis (EDA)
3. Clean and preprocess the data
4. Perform feature engineering
5. Encode categorical variables
6. Train the XGBoost model
7. Evaluate model performance
8. Save trained model
9. Predict fraud on new transactions

---

## Files

### preprocessing.py

- Loads dataset
- Performs feature engineering
- Encodes transaction type
- Saves processed dataset

### train_model.py

- Loads processed dataset
- Splits training and testing data
- Trains XGBoost model
- Saves trained model and feature information

### evaluate_model.py

- Loads trained model
- Evaluates model performance
- Displays classification report
- Displays confusion matrix
- Calculates ROC-AUC score

### predict_model.py

- Loads saved model
- Accepts transaction details
- Predicts fraud probability
- Displays transaction status

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Step 1

```bash
python src/preprocessing.py
```

### Step 2

```bash
python src/train_model.py
```

### Step 3

```bash
python src/evaluate_model.py
```

### Step 4

```bash
python src/predict_model.py
```

---

## Sample Output

```
Prediction Result

Transaction Status : LEGITIMATE

Fraud Probability : 22.65%
```

---

## Future Improvements

- SHAP-based model explainability
- Real-time fraud detection using FastAPI
- Interactive Streamlit dashboard
- Ensemble model comparison
- API integration for deployment

---

## Team

Developed as part of the AI-FraudShield project.