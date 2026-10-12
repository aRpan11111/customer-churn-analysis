
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "telco_churn_cleaned.csv"
OUTPUT_DIR = ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# Add dashboard-friendly fields
df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 60, 72],
    labels=["0-12 months", "13-24 months", "25-48 months",
            "49-60 months", "61-72 months"],
)

df["MonthlyChargeBand"] = pd.cut(
    df["MonthlyCharges"],
    bins=[0, 35, 70, 100, float("inf")],
    labels=["Low (<=35)", "Medium (35-70)",
            "High (70-100)", "Very high (>100)"],
    include_lowest=True,
)

df["CustomerStatus"] = df["Churn"].map(
    {"Yes": "Churned", "No": "Retained"}
)

output_file = OUTPUT_DIR / "tableau_churn_dashboard.csv"
df.to_csv(output_file, index=False)

print("Tableau dataset saved:", output_file)
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nChurn counts:")
print(df["CustomerStatus"].value_counts().to_string())

