
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "telco_churn_cleaned.csv"
REPORTS_DIR = ROOT / "reports"


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_FILE)

    # Customer ID is an identifier, not a useful predictor
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print("=" * 55)
    print("CUSTOMER CHURN PREDICTION")
    print("=" * 55)
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Retained", "Churned"],
        zero_division=0,
    ))

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    # Save test predictions for further analysis
    predictions_df = pd.DataFrame({
        "actual_churn": y_test.map({0: "No", 1: "Yes"}).to_numpy(),
        "predicted_churn": pd.Series(
            predictions, index=y_test.index
        ).map({0: "No", 1: "Yes"}).to_numpy(),
        "churn_probability": probabilities,
    })

    predictions_df.to_csv(
        REPORTS_DIR / "churn_model_predictions.csv",
        index=False,
    )

    metrics_df = pd.DataFrame([{
        "accuracy": accuracy,
        "roc_auc": auc,
        "train_records": len(X_train),
        "test_records": len(X_test),
    }])

    metrics_df.to_csv(
        REPORTS_DIR / "model_metrics.csv",
        index=False,
    )

    print("\nSaved reports/model_metrics.csv")
    print("Saved reports/churn_model_predictions.csv")


if __name__ == "__main__":
    main()


