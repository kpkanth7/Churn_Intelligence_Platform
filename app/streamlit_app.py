from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    # i push the project root onto sys.path so streamlit can import src cleanly from the app folder.
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import build_curated_cases, get_global_feature_insights, load_model_bundle, predict_dataframe


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_data.csv"
METRICS_PATH = PROJECT_ROOT / "outputs" / "training_metrics.json"
CURATED_CASES_PATH = PROJECT_ROOT / "outputs" / "curated_cases.json"


@st.cache_data
def load_processed_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_artifacts():
    return load_model_bundle()


@st.cache_data
def load_metrics() -> pd.DataFrame:
    metrics = json.loads(METRICS_PATH.read_text())
    return pd.DataFrame.from_dict(metrics["models"], orient="index")


def format_reason_block(reasons: str) -> list[str]:
    return [reason.strip() for reason in str(reasons).split("|") if reason.strip()]


@st.cache_data
def load_curated_cases() -> list[dict]:
    if CURATED_CASES_PATH.exists():
        return json.loads(CURATED_CASES_PATH.read_text())
    return build_curated_cases()


def build_manual_input_form() -> pd.DataFrame:
    # i keep the manual form opinionated so the demo feels guided instead of overwhelming.
    st.subheader("Score A Custom Customer")
    with st.form("manual_customer_input"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            dependents = st.selectbox("Dependents", [0, 1], format_func=lambda x: "Yes" if x else "No")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        with col2:
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly_charges = st.slider("Monthly Charges", 15.0, 130.0, 70.0, 0.5)
            total_charges = st.slider("Total Charges", 0.0, 9000.0, 1500.0, 0.5)
            phone_service = st.selectbox("Phone Service", [0, 1], format_func=lambda x: "Yes" if x else "No")
            paperless = st.selectbox("Paperless Billing", [0, 1], format_func=lambda x: "Yes" if x else "No")

        with col3:
            device_protection = st.selectbox("Device Protection", [0, 1], format_func=lambda x: "Yes" if x else "No")
            tech_support = st.selectbox("Tech Support", [0, 1], format_func=lambda x: "Yes" if x else "No")
            has_internet = st.selectbox("Has Internet", [0, 1], format_func=lambda x: "Yes" if x else "No")
            fiber_user = st.selectbox("Fiber User", [0, 1], format_func=lambda x: "Yes" if x else "No")
            num_complaints = st.slider("Complaint Count", 0, 5, 1)

        complaint_negative_score = st.slider("Negative Complaint Keyword Score", 0, 10, min(num_complaints, 3))
        complaint_text_length = st.slider("Complaint Text Length", 0, 1200, 120)
        billing_issue_flag = st.checkbox("Billing Issue Mentioned")
        technical_issue_flag = st.checkbox("Technical Issue Mentioned")
        service_issue_flag = st.checkbox("Service Issue Mentioned")

        submitted = st.form_submit_button("Score Customer")

    if not submitted:
        return pd.DataFrame()

    # i recreate the engineered fields here so custom demo inputs match the training schema.
    if tenure <= 12:
        tenure_group = "new"
    elif tenure <= 24:
        tenure_group = "mid"
    elif tenure <= 60:
        tenure_group = "loyal"
    else:
        tenure_group = "long_term"

    payment_risk = int(monthly_charges > 70 or contract == "Month-to-month")
    low_engagement = int(tenure < 12)
    support_calls = min(6, num_complaints + (1 if service_issue_flag else 0))
    high_value_customer = int(monthly_charges > 80)
    has_complaint = int(num_complaints > 0)
    complaint_negative_flag = int(complaint_negative_score > 0)

    manual_record = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "num_complaints": num_complaints,
        "payment_risk": payment_risk,
        "low_engagement": low_engagement,
        "support_calls": support_calls,
        "tenure_group": tenure_group,
        "high_value_customer": high_value_customer,
        "has_complaint": has_complaint,
        "has_internet": has_internet,
        "fiber_user": fiber_user,
        "complaint_text_length": complaint_text_length,
        "complaint_negative_score": complaint_negative_score,
        "complaint_negative_flag": complaint_negative_flag,
        "billing_issue_flag": int(billing_issue_flag),
        "technical_issue_flag": int(technical_issue_flag),
        "service_issue_flag": int(service_issue_flag),
    }
    return pd.DataFrame([manual_record])


def render_prediction_card(prediction: pd.Series, actual_churn=None, customer_id=None, complaint_text=None):
    # i use one shared card renderer so dataset rows and manual scenarios look the same in the app.
    top_left, top_mid, top_right = st.columns(3)
    top_left.metric("Customer ID", customer_id or "Custom Input")
    top_mid.metric("Actual Churn", "-" if actual_churn is None else int(actual_churn))
    top_right.metric("Predicted Churn", int(prediction["predicted_churn"]))

    risk_col, prob_col, band_col = st.columns(3)
    risk_col.metric("Churn Probability", f"{prediction['churn_probability']:.1%}")
    prob_col.metric("Risk Band", prediction["risk_band"].title())
    band_col.metric("Priority", "Immediate" if prediction["risk_band"] == "high" else "Monitor")

    st.subheader("Business Summary")
    st.info(prediction["business_summary"])

    st.subheader("Top Reasons")
    for reason in format_reason_block(prediction["top_reasons"]):
        st.write(f"- {reason}")

    st.subheader("Recommended Retention Action")
    st.success(prediction["retention_action"])

    if complaint_text is not None:
        st.subheader("Complaint Context")
        st.write(complaint_text)


def main():
    # i want the first screen to feel like a polished demo, not just a dataframe dump.
    st.set_page_config(page_title="Churn Intelligence Platform", layout="wide")

    st.title("Churn Intelligence Platform")
    st.caption(
        "Recruiter-friendly churn intelligence demo with structured features, complaint enrichment, "
        "explanations, and retention guidance."
    )

    dataset = load_processed_data()
    model, metadata = load_artifacts()

    st.sidebar.header("Demo Controls")
    view_mode = st.sidebar.radio("View Mode", ["Dataset Customer", "Custom Customer"])

    overview_left, overview_mid, overview_right = st.columns(3)
    overview_left.metric("Dataset Rows", len(dataset))
    overview_mid.metric("Selected Model", metadata["selected_model"].replace("_", " ").title())
    overview_right.metric("Feature Count", len(metadata["feature_columns"]))

    st.divider()

    tabs = st.tabs(["Prediction Workspace", "Model Insights", "Curated Customer Stories"])

    with tabs[0]:
        if view_mode == "Dataset Customer":
            customer_index = st.sidebar.slider(
                "Select a customer record",
                min_value=0,
                max_value=len(dataset) - 1,
                value=0,
                step=1,
            )
            selected_row = dataset.iloc[[customer_index]].copy()
            customer_record = selected_row.iloc[0]
            feature_frame = selected_row.drop(columns=["customerID", "Churn", "complaints"], errors="ignore")
            prediction = predict_dataframe(feature_frame, model=model, metadata=metadata).iloc[0]

            left_col, right_col = st.columns([1.2, 1])
            with left_col:
                st.subheader("Customer Profile")
                profile_columns = [
                    "gender",
                    "SeniorCitizen",
                    "Dependents",
                    "tenure",
                    "Contract",
                    "PaymentMethod",
                    "MonthlyCharges",
                    "TotalCharges",
                    "num_complaints",
                    "support_calls",
                    "tenure_group",
                    "has_internet",
                    "fiber_user",
                ]
                profile_data = {column: customer_record[column] for column in profile_columns}
                st.dataframe(pd.DataFrame(profile_data.items(), columns=["Field", "Value"]), use_container_width=True)

            with right_col:
                render_prediction_card(
                    prediction,
                    actual_churn=customer_record["Churn"],
                    customer_id=customer_record["customerID"],
                    complaint_text=customer_record["complaints"],
                )
        else:
            manual_df = build_manual_input_form()
            if not manual_df.empty:
                prediction = predict_dataframe(manual_df, model=model, metadata=metadata).iloc[0]
                render_prediction_card(prediction, customer_id="Manual Scenario")
            else:
                st.info("Fill out the form and click `Score Customer` to generate a live churn intelligence summary.")

    with tabs[1]:
        st.subheader("Current Model Metrics")
        st.dataframe(load_metrics(), use_container_width=True)

        st.subheader("Top Global Model Drivers")
        insights = get_global_feature_insights(model)
        if insights.empty:
            st.warning("Global coefficient insights are not available for the selected model.")
        else:
            st.dataframe(
                insights[["feature", "label", "direction", "impact"]].rename(
                    columns={"label": "business_label"}
                ),
                use_container_width=True,
            )
            st.caption(
                "These are global model tendencies from the selected model, not guaranteed per-customer explanations."
            )

    with tabs[2]:
        st.subheader("Curated Customer Stories")
        for case in load_curated_cases():
            with st.container(border=True):
                st.markdown(f"**{case['title']}**")
                st.write(
                    f"Risk band: {case['risk_band'].title()} | "
                    f"Churn probability: {case['churn_probability']:.1%}"
                )
                for reason in case["top_reasons"]:
                    st.write(f"- {reason}")
                st.write(case["retention_action"])
                st.caption(case["business_summary"])

    st.caption(
        "Complaint enrichment is lightweight synthetic augmentation using TWCS text. "
        "It improves the project narrative but is not a true customer-level support join."
    )


if __name__ == "__main__":
    main()
