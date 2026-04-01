import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_TELCO_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RAW_COMPLAINTS_PATH = PROJECT_ROOT / "data" / "raw" / "twcs.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_data.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATASET_AUDIT_PATH = OUTPUTS_DIR / "dataset_audit.json"

RANDOM_STATE = 42
MAX_COMPLAINT_USERS = 5000
NEGATIVE_KEYWORDS = {
    "bad",
    "broken",
    "cancel",
    "complaint",
    "delay",
    "disappointed",
    "issue",
    "problem",
    "refund",
    "slow",
    "terrible",
    "worst",
}
THEME_KEYWORDS = {
    "billing": ["bill", "billing", "charge", "charged", "payment", "refund", "price"],
    "technical": ["app", "connection", "internet", "network", "phone", "screen", "wifi"],
    "service": ["agent", "customer service", "help", "reply", "response", "support"],
}
CANONICAL_COLUMN_ORDER = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Dependents",
    "tenure",
    "PhoneService",
    "DeviceProtection",
    "TechSupport",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
    "complaints",
    "num_complaints",
    "payment_risk",
    "low_engagement",
    "support_calls",
    "tenure_group",
    "high_value_customer",
    "has_complaint",
    "has_internet",
    "fiber_user",
    "complaint_text_length",
    "complaint_negative_score",
    "complaint_negative_flag",
    "billing_issue_flag",
    "technical_issue_flag",
    "service_issue_flag",
]


def clean_text(text: str) -> str:
    # i clean the raw complaint text just enough to keep the downstream features readable.
    text = re.sub(r"@\w+", "", str(text))
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def score_negative_language(text: str) -> int:
    tokens = re.findall(r"[a-z']+", str(text).lower())
    return sum(token in NEGATIVE_KEYWORDS for token in tokens)


def theme_flag(text: str, keywords: list[str]) -> int:
    lowered = str(text).lower()
    return int(any(keyword in lowered for keyword in keywords))


def build_dataset(save: bool = True) -> pd.DataFrame:
    telco = pd.read_csv(RAW_TELCO_PATH)
    twcs = pd.read_csv(RAW_COMPLAINTS_PATH)

    # i only keep numeric author ids here because i am using them as a simple stand-in pool for complaint text.
    twcs = twcs[twcs["author_id"].astype(str).str.isnumeric()].copy()
    grouped = twcs.groupby("author_id")["text"].apply(list).reset_index()
    complaint_users = grouped.sample(
        min(MAX_COMPLAINT_USERS, len(grouped), len(telco)),
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    telco["complaints"] = ""
    telco["num_complaints"] = 0

    # i assign by position on purpose so i never create a stray extra row again.
    sampled_complaints = complaint_users["text"].apply(lambda texts: " ".join(texts))
    assignment_index = telco.index[: len(complaint_users)]
    telco.loc[assignment_index, "complaints"] = sampled_complaints.to_numpy()
    telco.loc[assignment_index, "num_complaints"] = complaint_users["text"].apply(len).to_numpy()

    telco["complaints"] = telco["complaints"].fillna("").map(clean_text)
    telco["num_complaints"] = telco["num_complaints"].clip(upper=5).astype(int)

    telco["TotalCharges"] = pd.to_numeric(telco["TotalCharges"], errors="coerce")
    telco["Churn"] = telco["Churn"].replace({"Yes": 1, "No": 0}).astype(int)

    telco["payment_risk"] = (
        (telco["MonthlyCharges"] > telco["MonthlyCharges"].median())
        | (telco["Contract"] == "Month-to-month")
    ).astype(int)
    telco["low_engagement"] = (telco["tenure"] < 12).astype(int)

    # i keep this synthetic support signal simple and deterministic so the project stays reproducible.
    rng = np.random.default_rng(RANDOM_STATE)
    telco["support_calls"] = telco["num_complaints"] + rng.integers(0, 2, len(telco))

    telco["tenure_group"] = pd.cut(
        telco["tenure"],
        bins=[-1, 12, 24, 60, 100],
        labels=["new", "mid", "loyal", "long_term"],
        include_lowest=True,
    )

    telco["high_value_customer"] = (telco["MonthlyCharges"] > 80).astype(int)
    telco["has_complaint"] = (telco["num_complaints"] > 0).astype(int)

    binary_cols = [
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling",
    ]
    for col in binary_cols:
        telco[col] = telco[col].replace(
            {
                "Yes": 1,
                "No": 0,
                "No phone service": 0,
                "No internet service": 0,
            }
        ).astype(int)

    telco["has_internet"] = (telco["InternetService"] != "No").astype(int)
    telco["fiber_user"] = (telco["InternetService"] == "Fiber optic").astype(int)

    telco["complaint_text_length"] = telco["complaints"].str.len().astype(int)
    telco["complaint_negative_score"] = telco["complaints"].map(score_negative_language).astype(int)
    telco["complaint_negative_flag"] = (telco["complaint_negative_score"] > 0).astype(int)
    telco["billing_issue_flag"] = telco["complaints"].map(
        lambda text: theme_flag(text, THEME_KEYWORDS["billing"])
    ).astype(int)
    telco["technical_issue_flag"] = telco["complaints"].map(
        lambda text: theme_flag(text, THEME_KEYWORDS["technical"])
    ).astype(int)
    telco["service_issue_flag"] = telco["complaints"].map(
        lambda text: theme_flag(text, THEME_KEYWORDS["service"])
    ).astype(int)

    telco["TotalCharges"] = telco["TotalCharges"].fillna(
        telco["MonthlyCharges"] * telco["tenure"]
    )

    telco = telco.drop(
        columns=[
            "Partner",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "StreamingTV",
            "StreamingMovies",
            "InternetService",
        ],
        errors="ignore",
    )

    # i replace empty complaint text with a plain placeholder so csv reloads stay clean.
    empty_complaints = telco["complaints"].astype(str).str.strip().eq("")
    telco.loc[empty_complaints, "complaints"] = "No complaint recorded."
    telco = telco.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    telco = telco[CANONICAL_COLUMN_ORDER].copy()

    validate_dataset(telco)

    if save:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        telco.to_csv(PROCESSED_DATA_PATH, index=False)
        write_dataset_audit(telco)

    return telco


def validate_dataset(df: pd.DataFrame) -> None:
    # i want the processed dataset to fail loudly if the shape or target looks wrong.
    if len(df) != 7043:
        raise ValueError(f"Expected 7043 rows, found {len(df)}.")
    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customerID values found in processed dataset.")
    if df["Churn"].isna().any():
        raise ValueError("Target column contains null values.")
    if set(df["Churn"].unique()) != {0, 1}:
        raise ValueError("Target column must be binary with values {0, 1}.")
    unexpected_nulls = df.drop(columns=["complaints"]).isna().sum()
    unexpected_nulls = unexpected_nulls[unexpected_nulls > 0]
    if not unexpected_nulls.empty:
        raise ValueError(f"Unexpected nulls found: {unexpected_nulls.to_dict()}")


def write_dataset_audit(df: pd.DataFrame) -> None:
    # i save a lightweight audit so i can sanity-check the processed data without reopening the notebook.
    audit = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": df.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "null_counts": {column: int(count) for column, count in df.isna().sum().items()},
        "target_distribution": {
            str(key): int(value) for key, value in df["Churn"].value_counts().sort_index().items()
        },
        "synthetic_complaint_enrichment": True,
        "notes": [
            "Complaint text is a lightweight enrichment sampled from the TWCS dataset.",
            "Complaint features are intended for project storytelling, not as a true customer-level join.",
        ],
    }
    DATASET_AUDIT_PATH.write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    dataset = build_dataset(save=True)
    print(f"Processed dataset saved to {PROCESSED_DATA_PATH}")
    print(f"Dataset shape: {dataset.shape}")
