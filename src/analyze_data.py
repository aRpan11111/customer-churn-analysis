
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "telco_churn_cleaned.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    sns.set_theme(style="whitegrid")

    # 1. Overall churn summary
    churn_counts = df["Churn"].value_counts()
    churn_rate = (df["Churn"].eq("Yes").mean()) * 100

    print("=" * 45)
    print("CUSTOMER CHURN ANALYSIS")
    print("=" * 45)
    print(f"Total customers: {len(df):,}")
    print(f"Customers who churned: {churn_counts.get('Yes', 0):,}")
    print(f"Customers retained: {churn_counts.get('No', 0):,}")
    print(f"Overall churn rate: {churn_rate:.2f}%")

    # 2. Churn rate by category
    categories = ["Contract", "PaymentMethod", "InternetService"]

    for column in categories:
        summary = (
            df.groupby(column, observed=True)["Churn"]
            .apply(lambda values: values.eq("Yes").mean() * 100)
            .sort_values(ascending=False)
            .round(2)
        )

        print(f"\nChurn rate by {column} (%):")
        print(summary.to_string())

        summary.to_csv(REPORTS / f"churn_by_{column.lower()}.csv")

        plt.figure(figsize=(9, 5))
        sns.barplot(x=summary.values, y=summary.index)
        plt.title(f"Churn Rate by {column}")
        plt.xlabel("Churn rate (%)")
        plt.ylabel(column)
        plt.tight_layout()
        plt.savefig(FIGURES / f"churn_by_{column.lower()}.png", dpi=150)
        plt.close()

    # 3. Churn by tenure groups
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 72],
        labels=["0-12", "13-24", "25-48", "49-60", "61-72"],
    )

    tenure_summary = (
        df.groupby("TenureGroup", observed=False)["Churn"]
        .apply(lambda values: values.eq("Yes").mean() * 100)
        .round(2)
    )

    print("\nChurn rate by tenure group (%):")
    print(tenure_summary.to_string())
    tenure_summary.to_csv(REPORTS / "churn_by_tenure.csv")

    plt.figure(figsize=(9, 5))
    sns.barplot(x=tenure_summary.index, y=tenure_summary.values)
    plt.title("Churn Rate by Customer Tenure")
    plt.xlabel("Tenure (months)")
    plt.ylabel("Churn rate (%)")
    plt.tight_layout()
    plt.savefig(FIGURES / "churn_by_tenure.png", dpi=150)
    plt.close()

    # 4. Monthly charges by churn status
    print("\nMonthly charges by churn status:")
    print(
        df.groupby("Churn")["MonthlyCharges"]
        .agg(["count", "mean", "median"])
        .round(2)
        .to_string()
    )

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
    plt.title("Monthly Charges: Churned vs Retained Customers")
    plt.xlabel("Churn")
    plt.ylabel("Monthly charges")
    plt.tight_layout()
    plt.savefig(FIGURES / "monthly_charges_by_churn.png", dpi=150)
    plt.close()

    # 5. Export a reusable analysis dataset for SQL/Tableau
    df.to_csv(REPORTS / "churn_analysis_data.csv", index=False)

    print("\nCharts saved in:", FIGURES)
    print("Analysis dataset saved in:", REPORTS / "churn_analysis_data.csv")


if __name__ == "__main__":
    main()

