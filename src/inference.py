import json
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BEST_MODEL_PATH = MODELS_DIR / "best_model_pipeline.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
TEST_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "test_predictions.csv"
SAMPLE_PREDICTION_PATH = OUTPUTS_DIR / "sample_prediction.json"
CURATED_CASES_PATH = OUTPUTS_DIR / "curated_cases.json"

REASON_LABELS = {
    "Contract": "month-to-month contract increases churn risk",
    "tenure": "short tenure suggests the customer is still early in the lifecycle",
    "MonthlyCharges": "higher monthly charges can increase price sensitivity",
    "payment_risk": "billing profile signals elevated payment friction",
    "low_engagement": "low engagement suggests weak product stickiness",
    "PhoneService": "phone service profile aligns with a more churn-prone segment",
    "num_complaints": "multiple complaints point to unresolved friction",
    "support_calls": "high support contact volume signals service friction",
    "has_complaint": "a recorded complaint raises churn concern",
    "complaint_negative_flag": "complaint language shows negative sentiment",
    "billing_issue_flag": "complaints mention billing or refund issues",
    "technical_issue_flag": "complaints mention technical or connectivity issues",
    "service_issue_flag": "complaints suggest poor service experience",
    "fiber_user": "fiber users often show higher churn in this dataset",
    "TechSupport": "lack of tech support coverage can increase churn risk",
    "DeviceProtection": "missing device protection can reduce stickiness",
    "PaperlessBilling": "paperless billing is associated with a more churn-prone segment",
    "PaymentMethod": "payment method falls into a riskier segment",
}
PROTECTIVE_REASON_LABELS = {
    "Contract": "contract structure is helping retention",
    "tenure": "longer tenure is helping retention",
    "MonthlyCharges": "pricing profile is not heavily pressuring retention",
    "TotalCharges": "overall account value profile is consistent with retained customers",
    "payment_risk": "billing profile does not show major payment friction",
    "low_engagement": "engagement signals are healthier than high-risk cases",
    "PhoneService": "phone service profile is not adding churn pressure",
    "num_complaints": "complaint volume is limited",
    "support_calls": "support contact volume is manageable",
    "has_complaint": "lack of a strong complaint signal is helping retention",
    "complaint_negative_flag": "complaint language is not strongly negative",
    "billing_issue_flag": "no major billing issue signal is present",
    "technical_issue_flag": "no major technical issue signal is present",
    "service_issue_flag": "no major service issue signal is present",
    "fiber_user": "internet profile is not creating unusual churn pressure",
    "TechSupport": "tech support coverage is helping retention",
    "DeviceProtection": "device protection coverage is supporting stickiness",
    "PaperlessBilling": "billing setup is not the main churn driver here",
    "PaymentMethod": "payment method is not the main churn driver here",
    "Dependents": "household stability signals are helping retention",
}


def load_model_bundle():
    # i load the saved pipeline and metadata together so inference always follows the trained schema.
    model = joblib.load(BEST_MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    return model, metadata


def risk_band(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    if probability >= 0.4:
        return "medium"
    return "low"


def validate_input_frame(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    required_columns = metadata["feature_columns"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for inference: {missing}")
    # i reorder the columns here because sklearn expects the same training-time layout back.
    return df[required_columns].copy()


def _base_feature_name(transformed_name: str) -> str:
    if "__" in transformed_name:
        transformed_name = transformed_name.split("__", 1)[1]
    for feature_name in REASON_LABELS:
        if transformed_name == feature_name or transformed_name.startswith(f"{feature_name}_"):
            return feature_name
    return transformed_name.split("_", 1)[0]


def explain_row(model, row: pd.DataFrame, probability: float) -> list[str]:
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    # i use coefficient-style explanations when i can because they are easier to talk through in a demo.
    if hasattr(classifier, "coef_") and hasattr(preprocessor, "get_feature_names_out"):
        transformed = preprocessor.transform(row)
        feature_names = preprocessor.get_feature_names_out()
        coefficients = classifier.coef_[0]
        transformed_row = transformed.toarray()[0] if hasattr(transformed, "toarray") else transformed[0]
        contributions = transformed_row * coefficients

        grouped = {}
        for feature_name, contribution in zip(feature_names, contributions):
            base_name = _base_feature_name(feature_name)
            grouped[base_name] = grouped.get(base_name, 0.0) + float(contribution)

        if probability >= 0.5:
            ranked = sorted(grouped.items(), key=lambda item: item[1], reverse=True)
            filtered = [name for name, value in ranked if value > 0][:3]
        else:
            ranked = sorted(grouped.items(), key=lambda item: item[1])
            filtered = [name for name, value in ranked if value < 0][:3]

        if filtered:
            label_source = REASON_LABELS if probability >= 0.5 else PROTECTIVE_REASON_LABELS
            return [label_source.get(name, f"{name} influenced the prediction") for name in filtered]

    return heuristic_reasons(row.iloc[0], probability)


def get_global_feature_insights(model, top_n: int = 8) -> pd.DataFrame:
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    # i keep this limited to the strongest drivers so the insight table stays readable.
    if not (hasattr(classifier, "coef_") and hasattr(preprocessor, "get_feature_names_out")):
        return pd.DataFrame(columns=["feature", "impact", "direction", "magnitude"])

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]
    grouped = {}
    for feature_name, coefficient in zip(feature_names, coefficients):
        base_name = _base_feature_name(feature_name)
        grouped[base_name] = grouped.get(base_name, 0.0) + float(coefficient)

    rows = []
    for feature_name, impact in grouped.items():
        rows.append(
            {
                "feature": feature_name,
                "impact": impact,
                "direction": "raises churn risk" if impact > 0 else "reduces churn risk",
                "magnitude": abs(impact),
                "label": REASON_LABELS.get(feature_name, feature_name),
            }
        )

    insight_frame = pd.DataFrame(rows).sort_values("magnitude", ascending=False)
    return insight_frame.head(top_n).reset_index(drop=True)


def heuristic_reasons(row: pd.Series, probability: float) -> list[str]:
    # i fall back to simple business rules here so every prediction still has a usable explanation.
    reasons = []
    if probability >= 0.5:
        if row.get("Contract") == "Month-to-month":
            reasons.append(REASON_LABELS["Contract"])
        if row.get("tenure", 0) < 12:
            reasons.append(REASON_LABELS["tenure"])
        if row.get("payment_risk", 0) == 1:
            reasons.append(REASON_LABELS["payment_risk"])
        if row.get("num_complaints", 0) >= 2:
            reasons.append(REASON_LABELS["num_complaints"])
        if row.get("technical_issue_flag", 0) == 1:
            reasons.append(REASON_LABELS["technical_issue_flag"])
        if row.get("service_issue_flag", 0) == 1:
            reasons.append(REASON_LABELS["service_issue_flag"])
    else:
        if row.get("Contract") in {"One year", "Two year"}:
            reasons.append(PROTECTIVE_REASON_LABELS["Contract"])
        if row.get("tenure", 0) >= 24:
            reasons.append(PROTECTIVE_REASON_LABELS["tenure"])
        if row.get("num_complaints", 0) == 0:
            reasons.append(PROTECTIVE_REASON_LABELS["num_complaints"])
        if row.get("TechSupport", 0) == 1:
            reasons.append(PROTECTIVE_REASON_LABELS["TechSupport"])
        if row.get("DeviceProtection", 0) == 1:
            reasons.append(PROTECTIVE_REASON_LABELS["DeviceProtection"])
        if row.get("technical_issue_flag", 0) == 0:
            reasons.append(PROTECTIVE_REASON_LABELS["technical_issue_flag"])

    if not reasons:
        if probability >= 0.5:
            reasons.append("combined behavioral and pricing signals are pushing risk upward")
        else:
            reasons.append("contract stability and lower friction signals are keeping risk down")
    return reasons[:3]


def recommend_retention_action(row: pd.Series, probability: float, reasons: list[str]) -> str:
    lowered = " ".join(reasons).lower()
    if probability < 0.4:
        return "Maintain standard engagement with a routine health-check and renewal reminder."
    if "billing" in lowered or row.get("payment_risk", 0) == 1:
        return "Offer a billing review, clarify charges, and test an autopay or plan-fit incentive."
    if "technical" in lowered or row.get("technical_issue_flag", 0) == 1:
        return "Trigger proactive technical support outreach and resolve service reliability issues."
    if "service" in lowered or row.get("num_complaints", 0) >= 2:
        return "Escalate to customer success for complaint recovery and close the loop on open issues."
    if row.get("tenure", 0) < 12 or row.get("low_engagement", 0) == 1:
        return "Launch an onboarding and adoption campaign with targeted feature education."
    return "Offer a tailored retention touchpoint with value reinforcement and contract review."


def build_business_summary(probability: float, reasons: list[str], action: str) -> str:
    band = risk_band(probability)
    reason_text = "; ".join(reasons[:2])
    return (
        f"Customer is in the {band} churn-risk band at {probability:.1%}. "
        f"Primary signals: {reason_text}. Recommended action: {action}"
    )


def predict_dataframe(df: pd.DataFrame, model=None, metadata=None) -> pd.DataFrame:
    if model is None or metadata is None:
        model, metadata = load_model_bundle()

    features = validate_input_frame(df, metadata)
    probabilities = model.predict_proba(features)[:, 1]
    predictions = model.predict(features)

    results = features.copy()
    results["predicted_churn"] = predictions.astype(int)
    results["churn_probability"] = probabilities
    results["risk_band"] = [risk_band(probability) for probability in probabilities]

    top_reasons = []
    retention_actions = []
    business_summaries = []
    # i build these row by row because the explanation and action text are customer-specific.
    for idx, probability in enumerate(probabilities):
        row = features.iloc[[idx]]
        reasons = explain_row(model, row, float(probability))
        action = recommend_retention_action(row.iloc[0], float(probability), reasons)
        top_reasons.append(" | ".join(reasons))
        retention_actions.append(action)
        business_summaries.append(build_business_summary(float(probability), reasons, action))

    results["top_reasons"] = top_reasons
    results["retention_action"] = retention_actions
    results["business_summary"] = business_summaries
    return results


def save_sample_prediction(df: pd.DataFrame) -> None:
    record = df.iloc[0].to_dict()
    SAMPLE_PREDICTION_PATH.write_text(json.dumps(record, indent=2, default=str))


def build_curated_cases() -> list[dict]:
    # i save a few representative stories so the demo has ready-made customer examples.
    predictions = pd.read_csv(TEST_PREDICTIONS_PATH)
    high_risk = predictions.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    medium_risk = predictions[
        (predictions["churn_probability"] >= 0.35) & (predictions["churn_probability"] < 0.65)
    ].sort_values("churn_probability", ascending=False).reset_index(drop=True)
    low_risk = predictions.sort_values("churn_probability", ascending=True).reset_index(drop=True)

    case_sources = [
        ("High-risk complaint-heavy customer", high_risk, 0),
        ("Medium-risk onboarding customer", medium_risk if not medium_risk.empty else high_risk, 0),
        ("Low-risk stable customer", low_risk, 0),
    ]

    curated = []
    for title, frame, idx in case_sources:
        row = frame.iloc[idx]
        curated.append(
            {
                "title": title,
                "risk_band": row["risk_band"],
                "churn_probability": float(row["churn_probability"]),
                "top_reasons": [reason.strip() for reason in str(row["top_reasons"]).split("|") if reason.strip()],
                "retention_action": row["retention_action"],
                "business_summary": row["business_summary"],
            }
        )

    CURATED_CASES_PATH.write_text(json.dumps(curated, indent=2))
    return curated


if __name__ == "__main__":
    model, metadata = load_model_bundle()
    example = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "final_data.csv")
    feature_frame = example.drop(columns=["customerID", "Churn", "complaints"], errors="ignore").head(1)
    prediction = predict_dataframe(feature_frame, model=model, metadata=metadata)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    save_sample_prediction(prediction)
    if TEST_PREDICTIONS_PATH.exists():
        build_curated_cases()
    print(prediction.to_dict(orient="records")[0])
