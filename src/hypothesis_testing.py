
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
from math import sqrt

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "telco_churn_cleaned.csv"
REPORTS_DIR = ROOT / "reports"


def main():
    df = pd.read_csv(DATA_FILE)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    alpha = 0.05
    results = []

    # Test 1: Contract type vs customer churn
    contingency = pd.crosstab(df["Contract"], df["Churn"])
    chi2, p_value, dof, expected = chi2_contingency(contingency)

    n = contingency.to_numpy().sum()
    rows, cols = contingency.shape
    cramers_v = sqrt(chi2 / (n * min(rows - 1, cols - 1)))

    print("=" * 55)
    print("TEST 1: CONTRACT TYPE VS CHURN")
    print("=" * 55)
    print("\nObserved counts:")
    print(contingency)
    print(f"\nChi-square statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p_value:.8g}")
    print(f"Cramer's V: {cramers_v:.4f}")

    if p_value < alpha:
        print("Conclusion: Statistically significant association.")
    else:
        print("Conclusion: No statistically significant association.")

    results.append({
        "test": "Contract type vs churn (chi-square)",
        "statistic": chi2,
        "p_value": p_value,
        "effect_size": cramers_v,
        "alpha": alpha,
        "significant": p_value < alpha,
    })

    # Test 2: Monthly charges, churned vs retained
    churned = df.loc[df["Churn"] == "Yes", "MonthlyCharges"]
    retained = df.loc[df["Churn"] == "No", "MonthlyCharges"]

    t_stat, p_value = ttest_ind(
        churned,
        retained,
        equal_var=False,
        alternative="two-sided",
    )

    difference = churned.mean() - retained.mean()

    # Approximate 95% confidence interval using Welch's standard error
    se = sqrt(
        churned.var(ddof=1) / len(churned)
        + retained.var(ddof=1) / len(retained)
    )

    # Welch-Satterthwaite degrees of freedom
    a = churned.var(ddof=1) / len(churned)
    b = retained.var(ddof=1) / len(retained)
    df_welch = (a + b) ** 2 / (
        a**2 / (len(churned) - 1)
        + b**2 / (len(retained) - 1)
    )

    from scipy.stats import t
    critical_value = t.ppf(0.975, df_welch)
    ci_low = difference - critical_value * se
    ci_high = difference + critical_value * se

    print("\n" + "=" * 55)
    print("TEST 2: MONTHLY CHARGES VS CHURN")
    print("=" * 55)
    print(f"Churned customers: {len(churned)}")
    print(f"Retained customers: {len(retained)}")
    print(f"Churned mean: {churned.mean():.2f}")
    print(f"Retained mean: {retained.mean():.2f}")
    print(f"Mean difference (churned - retained): {difference:.2f}")
    print(f"95% confidence interval: [{ci_low:.2f}, {ci_high:.2f}]")
    print(f"Welch t-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.8g}")

    if p_value < alpha:
        print("Conclusion: Statistically significant difference.")
    else:
        print("Conclusion: No statistically significant difference.")

    results.append({
        "test": "Monthly charges by churn (Welch t-test)",
        "statistic": t_stat,
        "p_value": p_value,
        "effect_size": None,
        "alpha": alpha,
        "significant": p_value < alpha,
    })

    # Save results for reporting
    pd.DataFrame(results).to_csv(
        REPORTS_DIR / "hypothesis_test_results.csv",
        index=False,
    )

    print("\nResults saved to reports/hypothesis_test_results.csv")


if __name__ == "__main__":
    main()


