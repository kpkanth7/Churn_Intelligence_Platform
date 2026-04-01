from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_data.csv"
EDA_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def save_plot(filename: str) -> None:
    # i save every plot into one place so the visuals are easy to reuse in the app or README later.
    EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(EDA_OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def plot_churn_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    churn_counts = df["Churn"].map({0: "No Churn", 1: "Churn"}).value_counts()
    sns.barplot(x=churn_counts.index, y=churn_counts.values, palette="Blues")
    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn Status")
    plt.ylabel("Customers")
    save_plot("churn_distribution.png")


def plot_monthly_charges_by_churn(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette="Set2")
    plt.title("Monthly Charges by Churn")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")
    save_plot("monthly_charges_by_churn.png")


def plot_tenure_by_churn(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="Churn", y="tenure", palette="Set3")
    plt.title("Tenure by Churn")
    plt.xlabel("Churn")
    plt.ylabel("Tenure (Months)")
    save_plot("tenure_by_churn.png")


def plot_contract_churn_rate(df: pd.DataFrame) -> None:
    contract_rates = (
        df.groupby("Contract", as_index=False)["Churn"]
        .mean()
        .sort_values("Churn", ascending=False)
    )
    plt.figure(figsize=(7, 4))
    sns.barplot(data=contract_rates, x="Contract", y="Churn", palette="crest")
    plt.title("Churn Rate by Contract Type")
    plt.xlabel("Contract")
    plt.ylabel("Churn Rate")
    save_plot("contract_churn_rate.png")


def plot_payment_method_churn_rate(df: pd.DataFrame) -> None:
    payment_rates = (
        df.groupby("PaymentMethod", as_index=False)["Churn"]
        .mean()
        .sort_values("Churn", ascending=False)
    )
    plt.figure(figsize=(8, 4))
    sns.barplot(data=payment_rates, x="PaymentMethod", y="Churn", palette="flare")
    plt.title("Churn Rate by Payment Method")
    plt.xlabel("Payment Method")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=20, ha="right")
    save_plot("payment_method_churn_rate.png")


def plot_complaints_by_churn(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="Churn", y="num_complaints", palette="rocket")
    plt.title("Complaint Count by Churn")
    plt.xlabel("Churn")
    plt.ylabel("Number of Complaints")
    save_plot("complaints_by_churn.png")


def plot_support_calls_by_churn(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="Churn", y="support_calls", palette="mako")
    plt.title("Support Calls by Churn")
    plt.xlabel("Churn")
    plt.ylabel("Support Calls")
    save_plot("support_calls_by_churn.png")


def plot_feature_heatmap(df: pd.DataFrame) -> None:
    # i keep this heatmap focused on the main numeric story instead of dumping every column into it.
    numeric_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "num_complaints",
        "support_calls",
        "payment_risk",
        "low_engagement",
        "high_value_customer",
        "complaint_negative_score",
        "Churn",
    ]
    plt.figure(figsize=(9, 6))
    correlation = df[numeric_columns].corr(numeric_only=True)
    sns.heatmap(correlation, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Correlation Heatmap of Key Numeric Features")
    save_plot("feature_correlation_heatmap.png")


def main() -> None:
    # i generate the whole EDA pack in one run so i do not have to babysit notebook exports.
    sns.set_theme(style="whitegrid")
    df = load_dataset()
    plot_churn_distribution(df)
    plot_monthly_charges_by_churn(df)
    plot_tenure_by_churn(df)
    plot_contract_churn_rate(df)
    plot_payment_method_churn_rate(df)
    plot_complaints_by_churn(df)
    plot_support_calls_by_churn(df)
    plot_feature_heatmap(df)
    print(f"EDA plots saved to {EDA_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
