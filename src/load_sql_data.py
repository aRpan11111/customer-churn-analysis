
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_FILE = ROOT / "reports" / "churn_analysis_data.csv"
DB_FILE = ROOT / "data" / "processed" / "churn_analysis.db"

df = pd.read_csv(CSV_FILE)

with sqlite3.connect(DB_FILE) as connection:
    df.to_sql(
        "customers",
        connection,
        if_exists="replace",
        index=False
    )

    count = pd.read_sql_query(
        "SELECT COUNT(*) AS total_customers FROM customers",
        connection
    )

print("Database created:", DB_FILE)
print(count.to_string(index=False))

