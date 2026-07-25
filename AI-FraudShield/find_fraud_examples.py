import pandas as pd

df = pd.read_csv("data/archive/PS_20174392719_1491204439457_log.csv")

fraud_rows = df[df["isFraud"] == 1].head(5)
print(fraud_rows.to_dict(orient="records"))
