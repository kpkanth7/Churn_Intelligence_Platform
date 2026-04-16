from pathlib import Path
import html
import json
import sys

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (  # noqa: E402
    build_curated_cases,
    get_global_feature_insights,
    load_model_bundle,
    predict_dataframe,
    risk_band,
)


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_data.csv"
METRICS_PATH = PROJECT_ROOT / "outputs" / "training_metrics.json"
CURATED_CASES_PATH = PROJECT_ROOT / "outputs" / "curated_cases.json"
EDA_DIR = PROJECT_ROOT / "outputs" / "eda"

RISK_ORDER = ["low", "medium", "high"]
RISK_COLORS = {"low": "#0F9F7E", "medium": "#F0B429", "high": "#E85F5C"}
RISK_COPY = {
    "low": "Stable account. Keep the relationship warm.",
    "medium": "Watch closely. A focused touchpoint can change the path.",
    "high": "Move now. The customer needs a clear save motion.",
}
PROFILE_GROUPS = [
    (
        "Account",
        [
            ("Customer ID", "customerID"),
            ("Contract", "Contract"),
            ("Tenure", "tenure"),
            ("Monthly Charges", "MonthlyCharges"),
        ],
    ),
    (
        "Relationship",
        [
            ("Gender", "gender"),
            ("Senior Citizen", "SeniorCitizen"),
            ("Dependents", "Dependents"),
            ("Payment Method", "PaymentMethod"),
        ],
    ),
    (
        "Service Signals",
        [
            ("Tech Support", "TechSupport"),
            ("Device Protection", "DeviceProtection"),
            ("Complaint Count", "num_complaints"),
            ("Support Calls", "support_calls"),
        ],
    ),
]


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


@st.cache_data(show_spinner=False)
def score_dataset_quick(dataset: pd.DataFrame, metadata: dict, _model) -> pd.DataFrame:
    feature_frame = dataset.drop(columns=["customerID", "Churn", "complaints"], errors="ignore")
    feature_frame = feature_frame[metadata["feature_columns"]].copy()
    probabilities = _model.predict_proba(feature_frame)[:, 1]
    predictions = _model.predict(feature_frame).astype(int)

    scored = dataset[
        [
            "customerID",
            "Churn",
            "Contract",
            "tenure",
            "MonthlyCharges",
            "PaymentMethod",
            "num_complaints",
            "support_calls",
            "complaints",
        ]
    ].copy()
    scored["churn_probability"] = probabilities
    scored["predicted_churn"] = predictions
    scored["risk_band"] = [risk_band(float(probability)) for probability in probabilities]
    return scored


@st.cache_data
def load_curated_cases() -> list[dict]:
    if CURATED_CASES_PATH.exists():
        return json.loads(CURATED_CASES_PATH.read_text())
    return build_curated_cases()


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #15171A;
            --muted: #596169;
            --line: #DDE5E0;
            --paper: #FFFFFF;
            --wash: #F7F9F6;
            --mint: #11836F;
            --coral: #E85F5C;
            --sky: #2A8FB8;
            --sun: #F0B429;
        }

        .stApp {
            background:
                linear-gradient(180deg, #F7F9F6 0%, #FFFFFF 36%, #F8FBFB 100%),
                radial-gradient(circle at top left, rgba(17, 131, 111, 0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(232, 95, 92, 0.08), transparent 26%);
            color: var(--ink);
        }

        section[data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid var(--line);
            box-shadow: 10px 0 30px rgba(21, 23, 26, 0.05);
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {
            color: var(--ink);
        }

        div[data-baseweb="select"] > div,
        div[data-testid="stSelectbox"] div,
        div[role="radiogroup"] label,
        div[data-testid="stCheckbox"] label,
        div[data-testid="stSlider"] label {
            color: var(--ink);
        }

        div[data-baseweb="select"] > div {
            background: #FFFFFF;
            border-color: var(--line);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1220px;
        }

        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ink);
        }

        div[data-testid="stMetric"] {
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.9rem 0.95rem;
            box-shadow: 0 12px 28px rgba(21, 23, 26, 0.06);
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: var(--muted);
        }

        div[data-testid="stTabs"] button {
            border-radius: 8px 8px 0 0;
        }

        .stButton > button, .stFormSubmitButton > button {
            border-radius: 8px;
            border: 1px solid #111714;
            background: #111714;
            color: #FFFFFF;
            font-weight: 700;
        }

        .stButton > button:hover, .stFormSubmitButton > button:hover {
            border-color: var(--mint);
            color: #FFFFFF;
            background: var(--mint);
        }

        .c-topline {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            margin-bottom: 1.1rem;
        }

        .c-eyebrow {
            color: var(--sky);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .c-title {
            font-size: clamp(2rem, 4.6vw, 3.8rem);
            line-height: 1.04;
            letter-spacing: 0;
            margin: 0;
            max-width: 900px;
        }

        .c-subtitle {
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.55;
            max-width: 760px;
            margin: 0.8rem 0 1.1rem;
        }

        .c-status {
            background: #15171A;
            color: #FFFFFF !important;
            border-radius: 8px;
            padding: 0.7rem 0.85rem;
            min-width: 165px;
            text-align: center;
            box-shadow: 0 14px 32px rgba(17, 23, 20, 0.20);
        }

        .c-status span,
        .c-status strong {
            color: #FFFFFF !important;
        }

        .c-status strong {
            display: block;
            font-size: 1.25rem;
        }

        .c-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 12px 30px rgba(21, 23, 26, 0.06);
            min-height: 100%;
        }

        .c-card h3 {
            margin: 0 0 0.35rem;
            font-size: 1rem;
        }

        .c-small {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.45;
        }

        .risk-card {
            background: linear-gradient(135deg, #15171A 0%, #18332D 100%);
            color: #FFFFFF;
            border-radius: 8px;
            padding: 1.15rem;
            box-shadow: 0 20px 40px rgba(17, 23, 20, 0.20);
        }

        .risk-card,
        .risk-card h2,
        .risk-card h3,
        .risk-card p,
        .risk-card strong,
        .risk-card div {
            color: #FFFFFF !important;
        }

        .risk-score {
            font-size: clamp(2.4rem, 8vw, 4.3rem);
            line-height: 1;
            font-weight: 900;
            letter-spacing: 0;
            margin: 0.25rem 0;
        }

        .risk-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 8px;
            padding: 0.33rem 0.58rem;
            color: #111714;
            font-size: 0.8rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .risk-card .risk-pill {
            color: #111714 !important;
        }

        .meter {
            width: 100%;
            height: 14px;
            border-radius: 8px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.16);
            margin: 0.85rem 0;
        }

        .meter > span {
            display: block;
            height: 100%;
            border-radius: 8px;
        }

        .reason-list {
            display: grid;
            gap: 0.55rem;
            margin-top: 0.7rem;
        }

        .reason-chip {
            border: 1px solid var(--line);
            background: #FFFFFF;
            border-radius: 8px;
            padding: 0.7rem 0.8rem;
            line-height: 1.42;
            color: var(--ink);
        }

        .action-box {
            border-left: 5px solid var(--mint);
            background: #F4FBF8;
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            color: #13241F;
            font-weight: 650;
        }

        .case-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #FFFFFF;
            padding: 1rem;
            min-height: 100%;
            box-shadow: 0 14px 32px rgba(21, 23, 26, 0.06);
        }

        .case-card h3,
        .case-card p,
        .case-card div {
            color: var(--ink);
        }

        .eda-caption {
            color: var(--muted);
            font-size: 0.86rem;
            margin-top: 0.25rem;
        }

        @media (max-width: 720px) {
            .c-topline {
                display: block;
            }

            .c-status {
                margin-top: 1rem;
                text-align: left;
            }

        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def esc(value) -> str:
    return html.escape(str(value))


def format_reason_block(reasons: str) -> list[str]:
    return [reason.strip() for reason in str(reasons).split("|") if reason.strip()]


def format_probability(value: float) -> str:
    return f"{value:.1%}"


def yes_no(value) -> str:
    return "Yes" if int(value) == 1 else "No"


def format_profile_value(label: str, value) -> str:
    if label in {"Monthly Charges", "Total Charges"}:
        return f"${float(value):,.2f}"
    if label == "Tenure":
        return f"{int(value)} months"
    if label in {"Senior Citizen", "Dependents", "Tech Support", "Device Protection"}:
        return yes_no(value)
    if label in {"Complaint Count", "Support Calls"}:
        return f"{int(value)}"
    return str(value).replace("_", " ").title() if label == "Tenure Segment" else str(value)


def card(title: str, value: str, detail: str, accent: str = "#0F9F7E") -> None:
    st.markdown(
        f"""
        <div class="c-card" style="border-top: 4px solid {accent};">
            <h3>{esc(title)}</h3>
            <div style="font-size: 1.75rem; font-weight: 900; letter-spacing: 0;">{esc(value)}</div>
            <div class="c-small">{esc(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(dataset: pd.DataFrame, metadata: dict, metrics: pd.DataFrame) -> None:
    selected_model = metadata["selected_model"].replace("_", " ").title()
    selected_metrics = metrics.loc[metadata["selected_model"]]
    churn_rate = dataset["Churn"].mean()

    st.markdown(
        f"""
        <div class="c-topline">
            <div>
                <div class="c-eyebrow">Retention intelligence cockpit</div>
                <h1 class="c-title">Find the customers most likely to leave, then choose the next move.</h1>
                <p class="c-subtitle">
                    Score live customer profiles, explain the risk drivers, and turn model output into a focused
                    retention action without leaving the dashboard.
                </p>
            </div>
            <div class="c-status">
                <span>Selected model</span>
                <strong>{esc(selected_model)}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        card("Customer records", f"{len(dataset):,}", "Processed Telco customer base", "#2A8FB8")
    with col2:
        card("Observed churn", format_probability(churn_rate), "Share of customers marked churned", "#E85F5C")
    with col3:
        card("ROC AUC", f"{selected_metrics['roc_auc']:.3f}", "Holdout ranking quality", "#0F9F7E")
    with col4:
        card("F1 score", f"{selected_metrics['f1']:.3f}", "Balanced precision and recall", "#F0B429")


def render_risk_card(prediction: pd.Series, actual_churn=None, customer_id=None) -> None:
    probability = float(prediction["churn_probability"])
    band = str(prediction["risk_band"])
    color = RISK_COLORS.get(band, "#2A8FB8")
    label_color = "#FFFFFF" if band == "high" else "#111714"
    meter_width = max(3, min(100, probability * 100))

    st.markdown(
        f"""
        <div class="risk-card">
            <div class="c-eyebrow" style="color: #9EE7D3;">Customer risk readout</div>
            <div style="display: flex; justify-content: space-between; gap: 1rem; align-items: flex-start;">
                <div>
                    <div class="risk-score">{format_probability(probability)}</div>
                    <span class="risk-pill" style="background: {color}; color: {label_color};">
                        {esc(band)} risk
                    </span>
                </div>
                <div style="text-align: right;">
                    <div class="c-small" style="color: rgba(255,255,255,0.72);">Customer</div>
                    <strong>{esc(customer_id or "Custom input")}</strong>
                    <div class="c-small" style="color: rgba(255,255,255,0.72); margin-top: 0.55rem;">Actual churn</div>
                    <strong>{esc("-" if actual_churn is None else int(actual_churn))}</strong>
                </div>
            </div>
            <div class="meter"><span style="width: {meter_width:.1f}%; background: {color};"></span></div>
            <p style="margin-bottom: 0;">{esc(RISK_COPY.get(band, "Review this account closely."))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_card(prediction: pd.Series, actual_churn=None, customer_id=None, complaint_text=None) -> None:
    render_risk_card(prediction, actual_churn=actual_churn, customer_id=customer_id)

    st.markdown("### Risk Drivers")
    reasons_html = "".join(
        f'<div class="reason-chip">{esc(reason)}</div>' for reason in format_reason_block(prediction["top_reasons"])
    )
    st.markdown(f'<div class="reason-list">{reasons_html}</div>', unsafe_allow_html=True)

    st.markdown("### Retention Action")
    st.markdown(f'<div class="action-box">{esc(prediction["retention_action"])}</div>', unsafe_allow_html=True)

    st.markdown("### Business Summary")
    st.info(prediction["business_summary"])

    if complaint_text is not None:
        with st.expander("Complaint Context"):
            st.write(complaint_text)


def build_manual_input_form() -> pd.DataFrame:
    st.markdown("### Build A Customer Scenario")
    st.caption("Change the account profile, then score the scenario against the saved model.")

    with st.form("manual_customer_input"):
        profile_col, billing_col, service_col = st.columns(3)

        with profile_col:
            st.markdown("**Profile**")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            dependents = st.selectbox("Dependents", [0, 1], format_func=lambda x: "Yes" if x else "No")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        with billing_col:
            st.markdown("**Billing**")
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
            paperless = st.selectbox("Paperless Billing", [0, 1], format_func=lambda x: "Yes" if x else "No")
            high_value_customer = int(monthly_charges > 80)

        with service_col:
            st.markdown("**Service Signals**")
            phone_service = st.selectbox("Phone Service", [0, 1], format_func=lambda x: "Yes" if x else "No")
            device_protection = st.selectbox("Device Protection", [0, 1], format_func=lambda x: "Yes" if x else "No")
            tech_support = st.selectbox("Tech Support", [0, 1], format_func=lambda x: "Yes" if x else "No")
            has_internet = st.selectbox("Has Internet", [0, 1], format_func=lambda x: "Yes" if x else "No")
            fiber_user = st.selectbox("Fiber User", [0, 1], format_func=lambda x: "Yes" if x else "No")

        st.markdown("**Complaint Context**")
        complaint_col1, complaint_col2, complaint_col3 = st.columns(3)
        with complaint_col1:
            num_complaints = st.slider("Complaint Count", 0, 5, 1)
            complaint_negative_score = st.slider("Negative Keyword Score", 0, 10, min(num_complaints, 3))
        with complaint_col2:
            complaint_text_length = st.slider("Complaint Text Length", 0, 1200, 120)
            billing_issue_flag = st.checkbox("Billing Issue Mentioned")
        with complaint_col3:
            technical_issue_flag = st.checkbox("Technical Issue Mentioned")
            service_issue_flag = st.checkbox("Service Issue Mentioned")

        submitted = st.form_submit_button("Score Customer")

    if not submitted:
        return pd.DataFrame()

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


def render_profile(customer_record: pd.Series) -> None:
    for title, fields in PROFILE_GROUPS:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            cols = st.columns(2)
            for idx, (label, column) in enumerate(fields):
                with cols[idx % 2]:
                    st.metric(label, format_profile_value(label, customer_record[column]))


def risk_distribution_chart(scored: pd.DataFrame):
    chart_data = scored["risk_band"].value_counts().reindex(RISK_ORDER, fill_value=0).reset_index()
    chart_data.columns = ["risk_band", "customers"]
    return (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("risk_band:N", sort=RISK_ORDER, title="Risk band"),
            y=alt.Y("customers:Q", title="Customers"),
            color=alt.Color(
                "risk_band:N",
                scale=alt.Scale(domain=RISK_ORDER, range=[RISK_COLORS[item] for item in RISK_ORDER]),
                legend=None,
            ),
            tooltip=["risk_band", "customers"],
        )
        .properties(height=280)
    )


def contract_risk_chart(scored: pd.DataFrame):
    chart_data = (
        scored.groupby("Contract", as_index=False)
        .agg(avg_probability=("churn_probability", "mean"), customers=("customerID", "count"))
        .sort_values("avg_probability", ascending=False)
    )
    return (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, color="#2A8FB8")
        .encode(
            y=alt.Y("Contract:N", sort="-x", title="Contract"),
            x=alt.X("avg_probability:Q", title="Average churn probability", axis=alt.Axis(format="%")),
            tooltip=[
                "Contract",
                alt.Tooltip("avg_probability:Q", format=".1%"),
                "customers:Q",
            ],
        )
        .properties(height=260)
    )


def render_portfolio_insights(scored: pd.DataFrame) -> None:
    high_risk = scored[scored["risk_band"] == "high"]
    medium_or_high = scored[scored["risk_band"].isin(["medium", "high"])]
    top_10_risk = scored.sort_values("churn_probability", ascending=False).head(10)["churn_probability"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("High-risk customers", f"{len(high_risk):,}")
    col2.metric("Medium + high watchlist", f"{len(medium_or_high):,}")
    col3.metric("Top 10 average risk", format_probability(top_10_risk))

    chart_col1, chart_col2 = st.columns([1, 1])
    with chart_col1:
        st.markdown("### Risk Distribution")
        st.altair_chart(risk_distribution_chart(scored), width="stretch")
    with chart_col2:
        st.markdown("### Contract Risk")
        st.altair_chart(contract_risk_chart(scored), width="stretch")

    st.markdown("### Highest Priority Accounts")
    display = scored.sort_values("churn_probability", ascending=False).head(12).copy()
    display["churn_probability"] = display["churn_probability"].map(format_probability)
    st.dataframe(
        display[
            [
                "customerID",
                "risk_band",
                "churn_probability",
                "Contract",
                "tenure",
                "MonthlyCharges",
                "num_complaints",
                "support_calls",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Visual Signals")
    image_specs = [
        ("Churn distribution", EDA_DIR / "churn_distribution.png"),
        ("Contract churn rate", EDA_DIR / "contract_churn_rate.png"),
        ("Payment method risk", EDA_DIR / "payment_method_churn_rate.png"),
        ("Complaint signal", EDA_DIR / "complaints_by_churn.png"),
    ]
    image_cols = st.columns(2)
    for idx, (caption, path) in enumerate(image_specs):
        with image_cols[idx % 2]:
            if path.exists():
                st.image(str(path), width="stretch")
                st.markdown(f'<div class="eda-caption">{esc(caption)}</div>', unsafe_allow_html=True)


def render_curated_cases() -> None:
    st.markdown("### Ready-Made Customer Stories")
    st.caption("Use these to walk through the model narrative quickly in a demo.")
    case_cols = st.columns(3)
    for idx, case in enumerate(load_curated_cases()):
        color = RISK_COLORS.get(case["risk_band"], "#2A8FB8")
        with case_cols[idx % 3]:
            reasons_html = "".join(
                f'<div class="reason-chip">{esc(reason)}</div>' for reason in case["top_reasons"]
            )
            st.markdown(
                f"""
                <div class="case-card" style="border-top: 4px solid {color};">
                    <div class="c-eyebrow" style="color: {color};">{esc(case["risk_band"])} risk</div>
                    <h3>{esc(case["title"])}</h3>
                    <div style="font-size: 2rem; font-weight: 900;">{format_probability(case["churn_probability"])}</div>
                    <p class="c-small">{esc(case["business_summary"])}</p>
                    <div class="reason-list">{reasons_html}</div>
                    <div style="margin-top: 0.8rem;" class="action-box">{esc(case["retention_action"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_model_notes(model, metadata: dict, metrics: pd.DataFrame) -> None:
    st.markdown("### Current Model Metrics")
    metric_table = metrics.copy()
    for column in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        metric_table[column] = metric_table[column].map(lambda value: f"{value:.3f}")
    st.dataframe(metric_table, width="stretch")

    st.markdown("### Global Model Drivers")
    insights = get_global_feature_insights(model)
    if insights.empty:
        st.warning("Global coefficient insights are not available for the selected model.")
    else:
        insight_display = insights[["feature", "label", "direction", "impact"]].rename(
            columns={"label": "business_label"}
        )
        st.dataframe(insight_display, width="stretch", hide_index=True)
        st.caption("These are global model tendencies from the selected model, not guaranteed per-customer reasons.")

    st.markdown("### Deployment Footprint")
    dep_cols = st.columns(3)
    dep_cols[0].metric("Features", len(metadata["feature_columns"]))
    dep_cols[1].metric("Train rows", f"{metadata['train_shape'][0]:,}")
    dep_cols[2].metric("Test rows", f"{metadata['test_shape'][0]:,}")


def main():
    st.set_page_config(page_title="Churn Intelligence Platform", layout="wide", page_icon="CI")
    inject_css()

    dataset = load_processed_data()
    model, metadata = load_artifacts()
    metrics = load_metrics()
    scored = score_dataset_quick(dataset, metadata, model)

    render_header(dataset, metadata, metrics)

    st.sidebar.markdown("## Demo Controls")
    view_mode = st.sidebar.radio("Scoring Mode", ["Dataset Customer", "Custom Customer"])
    risk_lens = st.sidebar.selectbox("Dataset Risk Lens", ["All", "High", "Medium", "Low"])
    sort_mode = st.sidebar.selectbox("Sort Dataset By", ["Highest risk", "Lowest risk", "Original order"])

    tabs = st.tabs(["Score Customer", "Portfolio Insights", "Customer Stories", "Model Notes"])

    with tabs[0]:
        if view_mode == "Dataset Customer":
            filtered = scored.copy()
            if risk_lens != "All":
                filtered = filtered[filtered["risk_band"] == risk_lens.lower()]

            if sort_mode == "Highest risk":
                filtered = filtered.sort_values("churn_probability", ascending=False)
            elif sort_mode == "Lowest risk":
                filtered = filtered.sort_values("churn_probability", ascending=True)

            filtered = filtered.reset_index()
            if filtered.empty:
                st.warning("No customers match that risk lens.")
                return

            choice_label = st.sidebar.selectbox(
                "Select Customer",
                filtered.index,
                format_func=lambda idx: (
                    f"{filtered.loc[idx, 'customerID']} - "
                    f"{filtered.loc[idx, 'risk_band'].title()} - "
                    f"{format_probability(filtered.loc[idx, 'churn_probability'])}"
                ),
            )
            source_index = int(filtered.loc[choice_label, "index"])
            selected_row = dataset.iloc[[source_index]].copy()
            customer_record = selected_row.iloc[0]
            feature_frame = selected_row.drop(columns=["customerID", "Churn", "complaints"], errors="ignore")
            prediction = predict_dataframe(feature_frame, model=model, metadata=metadata).iloc[0]

            left_col, right_col = st.columns([0.95, 1.05])
            with left_col:
                st.markdown("### Customer Profile")
                render_profile(customer_record)
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
                st.markdown("### Scenario Result")
                render_prediction_card(prediction, customer_id="Manual Scenario")
            else:
                st.info("Create a profile and score it to generate a live churn intelligence summary.")

    with tabs[1]:
        render_portfolio_insights(scored)

    with tabs[2]:
        render_curated_cases()

    with tabs[3]:
        render_model_notes(model, metadata, metrics)

    st.caption(
        "Complaint enrichment is lightweight synthetic augmentation. The demo is designed for portfolio storytelling, "
        "not production customer-level support matching."
    )


if __name__ == "__main__":
    main()
