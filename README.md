# AI-FRAUDSHIELD

An enterprise-grade, real-time **AI-Powered UPI Fraud Detection and Risk Intelligence Auditing Terminal**. The UI design features a premium, sleek charcoal theme inspired by Stripe, Ramp, and high-end fintech dashboards, powered by an optimized machine learning pipeline (XGBoost) to evaluate transaction risk markers dynamically.

---

## Technical Features

1. **Intelligent Forensic Dashboard**: 
   - Dynamic KPI metric counters (Scored Ledger Rows, Fraud Flagged Count, Average Risk Level, Protected Capital).
   - Natural language risk summaries generated dynamically.
   - Registry Search bar to instantly filter the transaction ledger by customer account, transaction ID, bank, or location.
   
2. **Transaction Scanner**:
   - A real-time transaction testing suite displaying Sender parameters, Receiver parameters, and transaction details concurrently.
   - Quick Preset Templates to simulate specific threat scenarios (Low-Risk P2P, Account Takeover Drain, and Rapid Cash-Out).
   - Balance auto-calculation verification and parameter validation checklist.

3. **Active Fraud Monitoring**:
   - Dynamic alert queues. Clicking "Approve" or "Block" performs active database overrides, logs actions to the history ledger, and triggers success notifications.

4. **Forensic Analytics**:
   - Plotly timeline trends, a weekly fraud density activity heatmap, and transaction method risk distributions.

5. **Risk Assessment Map**:
   - An interactive Plotly `Scattergeo` projection centered on India coordinates displaying metro hotspot bubbles, alert severity levels, and regional recovery summaries.

6. **Investigation Center (Bulk Upload)**:
   - Spreadsheet auditing tool allowing users to upload transaction CSVs, map column headers, and execute batch model predictions with exportable scored outputs.

7. **Exportable Reports**:
   - Executive TXT summaries and detailed CSV ledgers generated dynamically in-memory for download.

---

## Project Structure

```text
AI-FraudShield/
├── .streamlit/
│   └── config.toml             # Native Streamlit dark-mode configuration
├── data/
│   └── archive/
│       └── PS_20174392719_1491204439457_log.csv # Kaggle PaySim dataset (470MB)
├── models/
│   ├── xgboost_model.pkl       # Trained XGBoost classifier
│   ├── label_encoder.pkl       # Encoders for payment categories
│   └── feature_columns.pkl     # Feature array baselines
├── src/
│   ├── api.py                  # FastAPI high-performance inference engine
│   └── train.py                # Model training/evaluation scripts
├── dashboard.py                # Auditing Terminal main entry point
├── bulk_test_transactions.csv  # Pre-generated template for testing bulk uploads
└── README.md                   # System documentation
```

---

## Setup & Local Installation

### Prerequisites
- Python 3.10 or higher
- Streamlit and FastAPI installed in a virtual environment

### 1. Set Up Virtual Environment & Dependencies
Navigate to your project root folder and execute:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
*(If a requirements.txt is not yet generated, install packages manually)*:
```bash
pip install streamlit fastapi uvicorn pandas numpy scikit-learn xgboost plotly joblib requests
```

### 2. Start the FastAPI Backend
Start the high-performance inference API:
```bash
uvicorn src.api:app --reload
```
- API will start at **http://127.0.0.1:8000**
- View automatic API docs at **http://127.0.0.1:8000/docs**

### 3. Start the Streamlit Auditing Terminal
In a separate terminal window with your virtual environment active, run:
```bash
streamlit run dashboard.py
```
- The dashboard will open in your web browser at **http://localhost:8501**

---

## System Verification & Testing Guide

### 1. Seeding Data
On startup, the system searches the `data/` directory for the Kaggle PaySim dataset, reads the first 50 rows, and scores them using the local XGBoost model. This dynamic data populates all statistics and charts immediately on boot.

### 2. Testing the Scanner Presets
- Navigate to the **Transaction Scanner** page.
- Click **Scenario 2: Account Takeover Drain** to load the preset.
- All columns (Sender, Receiver, Details) will fill with coordinates.
- Click **Execute AI Fraud Scan** to evaluate parameters against the model and view risk scores and baselines.
- Modify any value in the form; the dirty indicator badge will display a warning indicating parameters are modified. Re-run to update the verdict.

### 3. Testing Bulk spreadsheet scoring
- Navigate to the **Investigation Center**.
- Click **Browse files** and upload the pre-generated [bulk_test_transactions.csv](bulk_test_transactions.csv) file from the project root.
- Match variables in the column mapping panel and click **Score Spreadsheet Rows**.
- Confirm that distributions, scored values, and the flagged list update dynamically, and download the final scored output sheet.