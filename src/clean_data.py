
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "telco_churn_cleaned.csv"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the original dataset
    df = pd.read_csv(RAW_FILE)

    print("Original dataset shape:", df.shape)
    print("\nMissing values before cleaning:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # Remove accidental whitespace from column names and text values
    df.columns = df.columns.str.strip()

    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip()

    # Convert TotalCharges to numeric.
    # Blank strings in this column become NaN.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    # Check and remove duplicate customer records
    duplicate_count = df.duplicated().sum()
    print("\nDuplicate rows found:", duplicate_count)
    df = df.drop_duplicates()

    # Check missing values after conversion
    print("\nMissing values after conversion:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # Remove rows with missing essential values
    df = df.dropna(subset=["customerID", "TotalCharges"])

    # Validate customer IDs
    df = df.drop_duplicates(subset=["customerID"], keep="first")

    # Save cleaned dataset
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nCleaned dataset shape:", df.shape)
    print("Cleaned file saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()

