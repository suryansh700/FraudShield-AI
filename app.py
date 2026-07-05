# ============================================================
# FRAUDSHIELD AI
# FINANCIAL FRAUD INTELLIGENCE PLATFORM
# ============================================================

import os
import json
import textwrap
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.case_explainer import FraudCaseExplainer


warnings.filterwarnings("ignore")


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "fraudshield_transactions.csv",
)

METADATA_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "model_metadata.json",
)

RESULTS_PATH = os.path.join(
    PROJECT_DIR,
    "reports",
    "model_results.csv",
)

PREDICTIONS_PATH = os.path.join(
    PROJECT_DIR,
    "reports",
    "evaluation_predictions.csv",
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    PROJECT_DIR,
    "reports",
    "feature_importance.csv",
)

ROC_PATH = os.path.join(
    PROJECT_DIR,
    "reports",
    "roc_curve.csv",
)

PR_PATH = os.path.join(
    PROJECT_DIR,
    "reports",
    "precision_recall_curve.csv",
)


# ============================================================
# HTML RENDERER
# ============================================================

def render_html(html):
    cleaned_html = textwrap.dedent(html).strip()

    st.markdown(
        cleaned_html,
        unsafe_allow_html=True,
    )


# ============================================================
# GLOBAL CSS
# ============================================================

render_html(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 88% 14%,
                rgba(82, 57, 190, 0.11),
                transparent 27%
            ),
            #050914;
        color: #f8fafc;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 5rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    header[data-testid="stHeader"] {
        background: #0b0e15;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    section[data-testid="stSidebar"] {
        background: #08101f;
        border-right: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    .sidebar-brand {
        padding: 18px 2px 20px 2px;
    }

    .sidebar-logo {
        color: #ffffff;
        font-size: 29px;
        font-weight: 850;
        letter-spacing: -1px;
    }

    .sidebar-logo span {
        color: #ff7a78;
    }

    .sidebar-subtitle {
        margin-top: 7px;
        color: #6680a9;
        font-size: 13px;
    }

    .engine-status {
        margin-top: 22px;
        margin-bottom: 30px;
        padding: 17px 16px;
        border-radius: 12px;
        background: rgba(40, 180, 110, 0.13);
        border: 1px solid rgba(72, 211, 139, 0.25);
        color: #57e389;
        font-size: 13px;
        font-weight: 800;
    }

    .sidebar-section-label {
        margin-top: 16px;
        margin-bottom: 12px;
        color: #8aa4cf;
        font-size: 12px;
        letter-spacing: 0.7px;
        font-weight: 700;
    }

    .sidebar-info-card {
        background: #0d1628;
        border: 1px solid #202d45;
        border-radius: 13px;
        padding: 17px;
        margin-top: 18px;
    }

    .sidebar-info-label {
        color: #7089b3;
        font-size: 10px;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .sidebar-info-value {
        color: #ffffff;
        font-size: 16px;
        font-weight: 800;
    }

    .sidebar-info-value.green {
        color: #55e586;
    }

    div[data-testid="stRadio"] label {
        color: #a9bddc !important;
        font-size: 14px;
    }

    .hero {
        padding: 35px 38px 37px 38px;
        border-radius: 0 0 27px 27px;
        background:
            linear-gradient(
                115deg,
                rgba(28, 62, 145, 0.86),
                rgba(43, 23, 102, 0.88)
            );
        border: 1px solid rgba(103, 112, 255, 0.32);
        margin-bottom: 45px;
    }

    .hero-label {
        color: #67a6ff;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        color: #ffffff;
        font-size: 42px;
        line-height: 1.18;
        font-weight: 850;
        letter-spacing: -1.5px;
        margin-bottom: 16px;
    }

    .hero-description {
        color: #9eb8df;
        font-size: 16px;
        line-height: 1.9;
        max-width: 1100px;
    }

    .section-title {
        color: #ffffff;
        font-size: 34px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-top: 12px;
        margin-bottom: 24px;
    }

    .subsection-title {
        color: #ffffff;
        font-size: 22px;
        font-weight: 800;
        margin-top: 24px;
        margin-bottom: 16px;
    }

    .metric-card {
        min-height: 172px;
        padding: 27px 25px;
        border-radius: 17px;
        background: #0d1628;
        border: 1px solid #202d45;
        margin-bottom: 20px;
    }

    .metric-icon {
        font-size: 25px;
        margin-bottom: 17px;
    }

    .metric-label {
        color: #7189b1;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        margin-bottom: 13px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 30px;
        font-weight: 850;
        margin-bottom: 11px;
        overflow-wrap: anywhere;
    }

    .metric-description {
        color: #637ca4;
        font-size: 12px;
        line-height: 1.6;
    }

    .information-box {
        background: #0c1629;
        border: 1px solid #223252;
        border-radius: 15px;
        padding: 20px 22px;
        color: #9ab4da;
        font-size: 14px;
        line-height: 1.8;
        margin-bottom: 24px;
    }

    .case-header {
        background:
            linear-gradient(
                110deg,
                rgba(34, 64, 139, 0.45),
                rgba(68, 37, 139, 0.40)
            );
        border: 1px solid #314472;
        border-radius: 18px;
        padding: 27px;
        margin-top: 25px;
        margin-bottom: 27px;
    }

    .case-id-label {
        color: #75a8ff;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.4px;
    }

    .case-id-value {
        color: #ffffff;
        font-size: 29px;
        font-weight: 850;
        margin-top: 7px;
    }

    .interpretation-box {
        background:
            linear-gradient(
                115deg,
                rgba(24, 53, 115, 0.32),
                rgba(65, 35, 125, 0.30)
            );
        border: 1px solid #304677;
        border-radius: 17px;
        padding: 27px;
        color: #b5c8e7;
        font-size: 15px;
        line-height: 1.9;
        margin-top: 15px;
        margin-bottom: 25px;
    }

    .architecture-box {
        background: #0c1629;
        border: 1px solid #223252;
        border-radius: 17px;
        padding: 27px;
        color: #9ab4da;
        font-size: 14px;
        line-height: 1.8;
        margin-bottom: 25px;
    }

    .architecture-title {
        color: #ffffff;
        font-size: 16px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #202d45;
        border-radius: 14px;
        overflow: hidden;
    }

    div[data-testid="stSelectbox"] > div > div {
        background: #0d1628;
        border-color: #293853;
        color: white;
    }

    .stButton > button {
        background: linear-gradient(90deg, #376fe8, #7958e8);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.3rem;
        font-weight: 700;
    }

    .stDownloadButton > button {
        width: 100%;
        background: linear-gradient(90deg, #2563eb, #6d4bd8);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 750;
        padding: 0.8rem 1.2rem;
    }

    hr {
        border-color: #233047 !important;
    }

    </style>
    """
)


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"fraudshield_transactions.csv was not found. Expected path: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return {}

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


@st.cache_data
def load_model_results():
    if not os.path.exists(RESULTS_PATH):
        return pd.DataFrame()

    return pd.read_csv(RESULTS_PATH)


@st.cache_data
def load_predictions():
    if not os.path.exists(PREDICTIONS_PATH):
        return pd.DataFrame()

    return pd.read_csv(PREDICTIONS_PATH)


@st.cache_data
def load_feature_importance():
    if not os.path.exists(FEATURE_IMPORTANCE_PATH):
        return pd.DataFrame()

    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


@st.cache_data
def load_roc_data():
    if not os.path.exists(ROC_PATH):
        return pd.DataFrame()

    return pd.read_csv(ROC_PATH)


@st.cache_data
def load_pr_data():
    if not os.path.exists(PR_PATH):
        return pd.DataFrame()

    return pd.read_csv(PR_PATH)


@st.cache_resource
def load_case_explainer():
    return FraudCaseExplainer()


# ============================================================
# UI COMPONENTS
# ============================================================

def hero(label, title, description):
    render_html(
        f"""
        <div class="hero">
            <div class="hero-label">{label}</div>
            <div class="hero-title">{title}</div>
            <div class="hero-description">{description}</div>
        </div>
        """
    )


def section_title(title):
    render_html(
        f"""
        <div class="section-title">{title}</div>
        """
    )


def subsection_title(title):
    render_html(
        f"""
        <div class="subsection-title">{title}</div>
        """
    )


def metric_card(icon, label, value, description):
    render_html(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-description">{description}</div>
        </div>
        """
    )


def information_box(text):
    render_html(
        f"""
        <div class="information-box">{text}</div>
        """
    )


def get_risk_level(probability):
    if probability >= 0.90:
        return "Critical"

    if probability >= 0.70:
        return "High"

    if probability >= 0.40:
        return "Medium"

    return "Low"


def transparent_chart(figure, height=430):
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(
            l=20,
            r=20,
            t=25,
            b=20,
        ),
    )

    return figure


# ============================================================
# LOAD DATA
# ============================================================

try:
    data = load_dataset()
    metadata = load_metadata()
    model_results = load_model_results()
    predictions = load_predictions()
    feature_importance = load_feature_importance()
    roc_data = load_roc_data()
    pr_data = load_pr_data()

except Exception as error:
    st.error(
        f"FraudShield initialization failed: {error}"
    )

    st.stop()


# ============================================================
# STANDARDIZE PREDICTION DATA
# ============================================================

if not predictions.empty:
    if "Fraud_Probability" in predictions.columns:
        predictions["Fraud_Probability"] = pd.to_numeric(
            predictions["Fraud_Probability"],
            errors="coerce",
        ).fillna(0)

    if "Risk_Level" not in predictions.columns:
        predictions["Risk_Level"] = predictions[
            "Fraud_Probability"
        ].apply(get_risk_level)


# ============================================================
# MODEL INFORMATION
# ============================================================

model_name = metadata.get(
    "model_name",
    metadata.get(
        "model",
        "Gradient Boosting",
    ),
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    render_html(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">
                🛡️ FraudShield<span>.</span>
            </div>
            <div class="sidebar-subtitle">
                Financial Fraud Intelligence
            </div>
        </div>
        """
    )

    render_html(
        """
        <div class="engine-status">
            ● FRAUD ENGINE ONLINE
        </div>
        """
    )

    render_html(
        """
        <div class="sidebar-section-label">
            INTELLIGENCE MODULES
        </div>
        """
    )

    module = st.radio(
        "FraudShield Navigation",
        [
            "Command Center",
            "Fraud Investigation",
            "Risk Intelligence",
            "Explainable AI",
            "Model Performance",
            "About System",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    render_html(
        f"""
        <div class="sidebar-info-card">
            <div class="sidebar-info-label">
                Active Detection Model
            </div>
            <div class="sidebar-info-value">
                {model_name}
            </div>
        </div>

        <div class="sidebar-info-card">
            <div class="sidebar-info-label">
                Model Status
            </div>
            <div class="sidebar-info-value green">
                Operational
            </div>
        </div>
        """
    )


# ============================================================
# COMMAND CENTER
# ============================================================

def command_center():
    hero(
        "FRAUD OPERATIONS",
        "Financial Fraud Command Center",
        (
            "Continuous transaction intelligence powered by machine "
            "learning. FraudShield automatically evaluates transaction "
            "behaviour, calculates fraud probability and prioritizes "
            "suspicious financial activity for investigation."
        ),
    )

    total_transactions = len(data)

    fraud_signals = int(
        (data["Class"] == 1).sum()
    )

    if not predictions.empty:
        critical_threats = int(
            (
                predictions["Fraud_Probability"] >= 0.90
            ).sum()
        )

        average_risk = (
            predictions["Fraud_Probability"].mean()
            * 100
        )

        peak_risk = (
            predictions["Fraud_Probability"].max()
            * 100
        )

    else:
        critical_threats = 0
        average_risk = 0
        peak_risk = 0

    columns = st.columns(4)

    with columns[0]:
        metric_card(
            "💳",
            "Transactions Monitored",
            f"{total_transactions:,}",
            "Complete transaction population",
        )

    with columns[1]:
        metric_card(
            "🚨",
            "Fraud Signals",
            f"{fraud_signals:,}",
            (
                f"{fraud_signals / total_transactions * 100:.2f}% "
                "historically classified"
            ),
        )

    with columns[2]:
        metric_card(
            "🔴",
            "Critical Threats",
            f"{critical_threats:,}",
            "Immediate investigation priority",
        )

    with columns[3]:
        metric_card(
            "🧠",
            "Average Risk",
            f"{average_risk:.2f}%",
            f"Peak risk {peak_risk:.1f}%",
        )

    section_title(
        "Threat Intelligence Overview"
    )

    chart_columns = st.columns(2)

    with chart_columns[0]:
        subsection_title(
            "Transaction Classification"
        )

        class_counts = (
            data["Class"]
            .value_counts()
            .rename_axis("Class")
            .reset_index(name="Transactions")
        )

        class_counts["Classification"] = (
            class_counts["Class"].map(
                {
                    0: "Legitimate",
                    1: "Fraudulent",
                }
            )
        )

        figure = px.pie(
            class_counts,
            values="Transactions",
            names="Classification",
            hole=0.67,
        )

        figure.update_layout(
            legend_title_text="",
        )

        transparent_chart(
            figure,
            430,
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    with chart_columns[1]:
        subsection_title(
            "Threat Severity Distribution"
        )

        if not predictions.empty:
            risk_distribution = (
                predictions["Risk_Level"]
                .value_counts()
                .reindex(
                    [
                        "Low",
                        "Medium",
                        "High",
                        "Critical",
                    ],
                    fill_value=0,
                )
                .rename_axis("Risk Level")
                .reset_index(name="Transactions")
            )

            figure = px.bar(
                risk_distribution,
                x="Risk Level",
                y="Transactions",
                text="Transactions",
            )

            figure.update_layout(
                showlegend=False,
                xaxis_title="Risk Classification",
                yaxis_title="Transactions",
            )

            transparent_chart(
                figure,
                430,
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

        else:
            information_box(
                "Prediction intelligence is currently unavailable."
            )


# ============================================================
# FRAUD INVESTIGATION
# ============================================================

def fraud_investigation():
    hero(
        "FRAUD INVESTIGATION",
        "Investigate High-Risk Financial Activity",
        (
            "Review prioritized fraud cases, inspect transaction-level "
            "risk signals and understand why the active detection model "
            "classified individual transactions as suspicious."
        ),
    )

    if predictions.empty:
        st.error(
            "evaluation_predictions.csv was not found. "
            "Run python src/train_model.py first."
        )

        return

    required_columns = [
        "Fraud_Probability",
        "Predicted_Class",
        "Actual_Class",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in predictions.columns
    ]

    if missing_columns:
        st.error(
            "Prediction report is missing columns: "
            + ", ".join(missing_columns)
        )

        return

    case_data = predictions[
        predictions["Fraud_Probability"] >= 0.50
    ].copy()

    case_data = case_data.sort_values(
        "Fraud_Probability",
        ascending=False,
    ).reset_index(drop=True)

    case_data["Case_ID"] = [
        f"CASE-{index + 1:06d}"
        for index in range(len(case_data))
    ]

    case_data["Risk_Level"] = case_data[
        "Fraud_Probability"
    ].apply(get_risk_level)

    open_cases = len(case_data)

    critical_cases = int(
        (
            case_data["Risk_Level"] == "Critical"
        ).sum()
    )

    high_cases = int(
        (
            case_data["Risk_Level"] == "High"
        ).sum()
    )

    average_case_risk = (
        case_data["Fraud_Probability"].mean() * 100
        if open_cases > 0
        else 0
    )

    columns = st.columns(4)

    with columns[0]:
        metric_card(
            "📂",
            "Open Cases",
            f"{open_cases:,}",
            "Cases requiring analyst review",
        )

    with columns[1]:
        metric_card(
            "🔴",
            "Critical Cases",
            f"{critical_cases:,}",
            "Immediate investigation priority",
        )

    with columns[2]:
        metric_card(
            "⚠️",
            "High Risk",
            f"{high_cases:,}",
            "Elevated fraud probability",
        )

    with columns[3]:
        metric_card(
            "🧠",
            "Average Case Risk",
            f"{average_case_risk:.2f}%",
            "Mean probability across open cases",
        )

    section_title(
        "Case Investigation Queue"
    )

    information_box(
        """
        The queue contains transactions assigned a fraud probability
        of at least 50% by the active detection model. Cases are
        automatically prioritized by model risk score.
        """
    )

    if case_data.empty:
        st.info(
            "No transactions currently exceed the investigation threshold."
        )

        return

    queue_display = case_data[
        [
            "Case_ID",
            "Fraud_Probability",
            "Risk_Level",
            "Predicted_Class",
            "Actual_Class",
        ]
    ].copy()

    queue_display["Fraud Probability (%)"] = (
        queue_display["Fraud_Probability"]
        * 100
    ).round(4)

    queue_display["Model Decision"] = (
        queue_display["Predicted_Class"].map(
            {
                0: "Legitimate",
                1: "Fraudulent",
            }
        )
    )

    queue_display["Evaluation Label"] = (
        queue_display["Actual_Class"].map(
            {
                0: "Legitimate",
                1: "Fraudulent",
            }
        )
    )

    queue_display = queue_display[
        [
            "Case_ID",
            "Fraud Probability (%)",
            "Risk_Level",
            "Model Decision",
            "Evaluation Label",
        ]
    ]

    queue_display.columns = [
        "Case ID",
        "Fraud Probability (%)",
        "Risk Level",
        "Model Decision",
        "Evaluation Label",
    ]

    st.dataframe(
        queue_display.head(100),
        use_container_width=True,
        hide_index=True,
    )

    section_title(
        "Case Intelligence View"
    )

    selected_case_id = st.selectbox(
        "Select a case for detailed investigation",
        case_data["Case_ID"].tolist(),
    )

    selected_case = case_data[
        case_data["Case_ID"] == selected_case_id
    ].iloc[0]

    render_html(
        f"""
        <div class="case-header">
            <div class="case-id-label">
                ACTIVE FRAUD INVESTIGATION
            </div>
            <div class="case-id-value">
                {selected_case_id}
            </div>
        </div>
        """
    )

    try:
        explainer = load_case_explainer()

        transaction = pd.Series(
            {
                feature: selected_case[feature]
                for feature in explainer.feature_columns
            }
        )

        summary = explainer.generate_case_summary(
            transaction,
            top_n=5,
        )

        explanation = explainer.explain_transaction(
            transaction,
            top_n=10,
        )

        interpretation = (
            explainer.generate_risk_interpretation(
                transaction
            )
        )

    except Exception as error:
        st.error(
            f"Case explanation failed: {error}"
        )

        return

    columns = st.columns(4)

    with columns[0]:
        metric_card(
            "🎯",
            "Fraud Probability",
            f"{summary['fraud_probability']:.2f}%",
            "Active model probability",
        )

    with columns[1]:
        metric_card(
            "🚨",
            "Risk Classification",
            str(summary["risk_level"]).upper(),
            "FraudShield severity assessment",
        )

    with columns[2]:
        metric_card(
            "🧠",
            "Model Decision",
            str(summary["model_decision"]).upper(),
            "Binary model classification",
        )

    with columns[3]:
        metric_card(
            "🔬",
            "Strongest Risk Signal",
            summary["strongest_risk_feature"],
            (
                f"{summary['strongest_contribution']:.2f} "
                "percentage-point influence"
            ),
        )

    section_title(
        "Why Was This Transaction Flagged?"
    )

    information_box(
        """
        FraudShield performs transaction-level perturbation analysis.
        Each feature is replaced with its reference baseline and the
        resulting change in fraud probability is measured. Positive
        values indicate features that increase the model's fraud-risk
        assessment for this specific transaction.
        """
    )

    chart_data = explanation.copy()

    chart_data["Contribution Direction"] = np.where(
        chart_data["Risk_Contribution"] >= 0,
        "Risk Increasing",
        "Risk Reducing",
    )

    chart_data = chart_data.sort_values(
        "Risk_Contribution"
    )

    figure = px.bar(
        chart_data,
        x="Risk_Contribution",
        y="Feature",
        orientation="h",
        text="Risk_Contribution",
        color="Contribution Direction",
    )

    figure.update_traces(
        texttemplate="%{text:.2f} pp",
        textposition="outside",
    )

    figure.update_layout(
        xaxis_title=(
            "Fraud Probability Influence "
            "(Percentage Points)"
        ),
        yaxis_title="Transaction Feature",
        legend_title_text="",
    )

    transparent_chart(
        figure,
        500,
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    explanation_display = explanation[
        [
            "Rank",
            "Feature",
            "Original_Value",
            "Baseline_Value",
            "Risk_Contribution",
            "Influence",
        ]
    ].copy()

    explanation_display[
        "Original_Value"
    ] = explanation_display[
        "Original_Value"
    ].round(5)

    explanation_display[
        "Baseline_Value"
    ] = explanation_display[
        "Baseline_Value"
    ].round(5)

    explanation_display[
        "Risk_Contribution"
    ] = explanation_display[
        "Risk_Contribution"
    ].round(4)

    explanation_display.columns = [
        "Rank",
        "Feature",
        "Transaction Value",
        "Reference Baseline",
        "Risk Influence (pp)",
        "Influence",
    ]

    st.dataframe(
        explanation_display,
        use_container_width=True,
        hide_index=True,
    )

    section_title(
        "FraudShield Risk Interpretation"
    )

    render_html(
        f"""
        <div class="interpretation-box">
            🧠 &nbsp; {interpretation}
        </div>
        """
    )

    section_title(
        "Case Intelligence Report"
    )

    report_text = f"""
FRAUDSHIELD AI
FINANCIAL FRAUD CASE INTELLIGENCE REPORT
============================================================

CASE INFORMATION
------------------------------------------------------------

Case ID                 : {selected_case_id}
Fraud Probability       : {summary['fraud_probability']:.4f}%
Risk Classification     : {summary['risk_level']}
Model Decision          : {summary['model_decision']}
Active Detection Model  : {model_name}

MODEL RISK INTELLIGENCE
------------------------------------------------------------

Strongest Risk Feature  : {summary['strongest_risk_feature']}
Strongest Contribution  : {summary['strongest_contribution']:.4f} percentage points
Risk Increasing Signals : {summary['risk_increasing_features']}
Risk Reducing Signals   : {summary['risk_reducing_features']}

TOP RISK FEATURES
------------------------------------------------------------

{", ".join(summary['top_risk_features'])}

FRAUDSHIELD INTERPRETATION
------------------------------------------------------------

{interpretation}

TRANSACTION-LEVEL FEATURE ANALYSIS
------------------------------------------------------------

{explanation_display.to_string(index=False)}

============================================================
FraudShield AI
Machine Learning Financial Fraud Intelligence Platform

This report contains model-generated decision-support
information and should be reviewed by a qualified analyst.
============================================================
"""

    st.download_button(
        label="⬇ Download Case Intelligence Report",
        data=report_text,
        file_name=(
            f"{selected_case_id}_FraudShield_Report.txt"
        ),
        mime="text/plain",
    )


# ============================================================
# RISK INTELLIGENCE
# ============================================================

def risk_intelligence():
    hero(
        "RISK INTELLIGENCE",
        "Understand the Fraud Risk Landscape",
        (
            "Analyze fraud probability distribution, severity segments "
            "and model-driven transaction risk across the evaluation "
            "population."
        ),
    )

    if predictions.empty:
        st.warning(
            "Risk intelligence data is unavailable."
        )

        return

    risk_probabilities = (
        predictions["Fraud_Probability"] * 100
    )

    columns = st.columns(4)

    with columns[0]:
        metric_card(
            "📊",
            "Evaluated Transactions",
            f"{len(predictions):,}",
            "Held-out model evaluation population",
        )

    with columns[1]:
        metric_card(
            "⚠️",
            "Risk Above 50%",
            f"{(risk_probabilities >= 50).sum():,}",
            "Transactions crossing investigation threshold",
        )

    with columns[2]:
        metric_card(
            "🔴",
            "Risk Above 90%",
            f"{(risk_probabilities >= 90).sum():,}",
            "Critical probability segment",
        )

    with columns[3]:
        metric_card(
            "🎯",
            "Maximum Risk",
            f"{risk_probabilities.max():.2f}%",
            "Highest observed fraud probability",
        )

    section_title(
        "Fraud Probability Distribution"
    )

    figure = px.histogram(
        predictions,
        x="Fraud_Probability",
        nbins=60,
    )

    figure.update_layout(
        xaxis_title="Fraud Probability",
        yaxis_title="Transactions",
        showlegend=False,
    )

    transparent_chart(
        figure,
        460,
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    section_title(
        "Risk Segmentation"
    )

    risk_counts = (
        predictions["Risk_Level"]
        .value_counts()
        .reindex(
            [
                "Low",
                "Medium",
                "High",
                "Critical",
            ],
            fill_value=0,
        )
        .rename_axis("Risk Level")
        .reset_index(name="Transactions")
    )

    figure = px.bar(
        risk_counts,
        x="Risk Level",
        y="Transactions",
        text="Transactions",
    )

    figure.update_layout(
        xaxis_title="Risk Level",
        yaxis_title="Transactions",
        showlegend=False,
    )

    transparent_chart(
        figure,
        450,
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    section_title(
        "Highest Risk Transactions"
    )

    risk_table = predictions.sort_values(
        "Fraud_Probability",
        ascending=False,
    ).head(100).copy()

    risk_table[
        "Fraud Probability (%)"
    ] = (
        risk_table["Fraud_Probability"] * 100
    ).round(4)

    available_columns = [
        column
        for column in [
            "Fraud Probability (%)",
            "Risk_Level",
            "Predicted_Class",
            "Actual_Class",
        ]
        if column in risk_table.columns
    ]

    st.dataframe(
        risk_table[available_columns],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# EXPLAINABLE AI
# ============================================================

def explainable_ai():
    hero(
        "EXPLAINABLE FRAUD INTELLIGENCE",
        "Understand How FraudShield Makes Detection Decisions",
        (
            "Explore the global behaviour of the active fraud detection "
            "model. Permutation importance measures how strongly each "
            "transaction feature affects fraud-detection performance "
            "across the evaluation population."
        ),
    )

    if feature_importance.empty:
        st.warning(
            "Explainability report was not found. "
            "Run python src/generate_explainability.py first."
        )

        return

    feature_column = None

    for column in [
        "Feature",
        "feature",
    ]:
        if column in feature_importance.columns:
            feature_column = column
            break

    if feature_column is None:
        st.error(
            "Feature column was not found in feature_importance.csv."
        )

        return

    importance_column = None

    for column in [
        "Importance",
        "Permutation_Importance",
        "Absolute_Importance",
        "Mean_Importance",
    ]:
        if column in feature_importance.columns:
            importance_column = column
            break

    if importance_column is None:
        numeric_columns = (
            feature_importance
            .select_dtypes(include=np.number)
            .columns
            .tolist()
        )

        if not numeric_columns:
            st.error(
                "No numeric importance column was found."
            )

            return

        importance_column = numeric_columns[0]

    importance_data = feature_importance.copy()

    importance_data[
        "Absolute Importance"
    ] = importance_data[
        importance_column
    ].abs()

    importance_data = importance_data.sort_values(
        "Absolute Importance",
        ascending=False,
    ).reset_index(drop=True)

    top_feature = str(
        importance_data.iloc[0][feature_column]
    )

    positive_features = int(
        (
            importance_data[importance_column] > 0
        ).sum()
    )

    columns = st.columns(4)

    with columns[0]:
        metric_card(
            "🧠",
            "Top Detection Feature",
            top_feature,
            "Highest global model influence",
        )

    with columns[1]:
        metric_card(
            "🔬",
            "Features Analyzed",
            f"{len(importance_data)}",
            "Processed model features",
        )

    with columns[2]:
        metric_card(
            "📈",
            "Positive Influence",
            f"{positive_features}",
            "Positive permutation importance",
        )

    with columns[3]:
        metric_card(
            "🧪",
            "Explainability Method",
            "Permutation",
            "Model-agnostic global analysis",
        )

    section_title(
        "Most Influential Fraud Detection Features"
    )

    information_box(
        """
        The chart ranks transaction features by absolute permutation
        importance. A larger score means that disturbing the feature
        causes a greater change in fraud-detection performance. This is
        a global model explanation and should not be interpreted as the
        explanation for one individual transaction.
        """
    )

    top_features = (
        importance_data
        .head(15)
        .sort_values("Absolute Importance")
    )

    figure = px.bar(
        top_features,
        x="Absolute Importance",
        y=feature_column,
        orientation="h",
        text="Absolute Importance",
    )

    figure.update_traces(
        texttemplate="%{text:.5f}",
        textposition="outside",
    )

    figure.update_layout(
        xaxis_title="Absolute Permutation Importance",
        yaxis_title="Feature",
        showlegend=False,
    )

    transparent_chart(
        figure,
        580,
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    section_title(
        "Global Feature Intelligence"
    )

    st.dataframe(
        importance_data,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

def model_performance():
    hero(
        "MODEL PERFORMANCE",
        "Fraud Detection Model Validation",
        (
            "Review comparative model performance and inspect the "
            "evaluation metrics used to select FraudShield's active "
            "fraud detection model."
        ),
    )

    if model_results.empty:
        st.warning(
            "Model results are unavailable."
        )

        return

    best_result = model_results.iloc[0]

    columns = st.columns(4)

    with columns[0]:
        metric_card(
            "🎯",
            "Accuracy",
            f"{best_result['Accuracy'] * 100:.2f}%",
            "Overall classification accuracy",
        )

    with columns[1]:
        metric_card(
            "🔎",
            "Precision",
            f"{best_result['Precision'] * 100:.2f}%",
            "Fraud alert reliability",
        )

    with columns[2]:
        metric_card(
            "🚨",
            "Recall",
            f"{best_result['Recall'] * 100:.2f}%",
            "Detected fraud transactions",
        )

    with columns[3]:
        metric_card(
            "📈",
            "ROC-AUC",
            f"{best_result['ROC AUC'] * 100:.2f}%",
            "Ranking discrimination performance",
        )

    section_title(
        "Model Comparison"
    )

    st.dataframe(
        model_results,
        use_container_width=True,
        hide_index=True,
    )

    chart_columns = st.columns(2)

    with chart_columns[0]:
        subsection_title(
            "ROC Curve"
        )

        if not roc_data.empty:
            fpr_column = next(
                (
                    column
                    for column in [
                        "False Positive Rate",
                        "FPR",
                        "fpr",
                    ]
                    if column in roc_data.columns
                ),
                None,
            )

            tpr_column = next(
                (
                    column
                    for column in [
                        "True Positive Rate",
                        "TPR",
                        "tpr",
                    ]
                    if column in roc_data.columns
                ),
                None,
            )

            if fpr_column and tpr_column:
                figure = go.Figure()

                figure.add_trace(
                    go.Scatter(
                        x=roc_data[fpr_column],
                        y=roc_data[tpr_column],
                        mode="lines",
                        name="FraudShield Model",
                    )
                )

                figure.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Random Classifier",
                        line=dict(dash="dash"),
                    )
                )

                figure.update_layout(
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                )

                transparent_chart(
                    figure,
                    430,
                )

                st.plotly_chart(
                    figure,
                    use_container_width=True,
                )

            else:
                information_box(
                    "ROC curve columns could not be identified."
                )

        else:
            information_box(
                "ROC curve report is currently unavailable."
            )

    with chart_columns[1]:
        subsection_title(
            "Precision-Recall Curve"
        )

        if not pr_data.empty:
            recall_column = next(
                (
                    column
                    for column in [
                        "Recall",
                        "recall",
                    ]
                    if column in pr_data.columns
                ),
                None,
            )

            precision_column = next(
                (
                    column
                    for column in [
                        "Precision",
                        "precision",
                    ]
                    if column in pr_data.columns
                ),
                None,
            )

            if recall_column and precision_column:
                figure = go.Figure()

                figure.add_trace(
                    go.Scatter(
                        x=pr_data[recall_column],
                        y=pr_data[precision_column],
                        mode="lines",
                        name="FraudShield Model",
                    )
                )

                figure.update_layout(
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                )

                transparent_chart(
                    figure,
                    430,
                )

                st.plotly_chart(
                    figure,
                    use_container_width=True,
                )

            else:
                information_box(
                    "Precision-recall curve columns could not be identified."
                )

        else:
            information_box(
                "Precision-recall curve report is currently unavailable."
            )


# ============================================================
# ABOUT SYSTEM
# ============================================================

# ============================================================
# ABOUT SYSTEM
# ============================================================

def about_system():
    hero(
        "ABOUT FRAUDSHIELD",
        "Machine Learning Financial Fraud Intelligence",
        (
            "FraudShield AI is an end-to-end financial fraud detection "
            "and investigation platform designed to demonstrate machine "
            "learning model development, class imbalance handling, "
            "probability-based risk intelligence and explainable fraud "
            "investigation."
        ),
    )

    # ========================================================
    # SYSTEM ARCHITECTURE
    # ========================================================

    section_title("System Architecture")

    architecture_steps = [
        {
            "number": "01",
            "icon": "💳",
            "title": "Transaction Dataset",
            "description": (
                "Historical financial transaction records form the "
                "machine learning model development and evaluation population."
            ),
        },
        {
            "number": "02",
            "icon": "⚙️",
            "title": "Data Preprocessing",
            "description": (
                "Transaction features are processed using the fitted "
                "preprocessing artifact generated during model training."
            ),
        },
        {
            "number": "03",
            "icon": "⚖️",
            "title": "Class Imbalance Strategy",
            "description": (
                "SMOTE is applied only to the training population to improve "
                "representation of fraudulent transaction examples."
            ),
        },
        {
            "number": "04",
            "icon": "🧪",
            "title": "Model Comparison",
            "description": (
                "Multiple classification algorithms are trained and evaluated "
                "using fraud-focused performance metrics."
            ),
        },
        {
            "number": "05",
            "icon": "🧠",
            "title": "Fraud Probability Engine",
            "description": (
                "The selected machine learning model calculates fraud "
                "probability for evaluated financial transactions."
            ),
        },
        {
            "number": "06",
            "icon": "🚨",
            "title": "Risk Intelligence",
            "description": (
                "Fraud probabilities are converted into operational risk "
                "segments for investigation prioritization."
            ),
        },
        {
            "number": "07",
            "icon": "📊",
            "title": "Global Explainability",
            "description": (
                "Permutation importance measures the overall influence of "
                "transaction features on fraud-detection performance."
            ),
        },
        {
            "number": "08",
            "icon": "🔬",
            "title": "Case-Level Explainability",
            "description": (
                "Transaction perturbation analysis estimates how individual "
                "features influence fraud probability for a selected case."
            ),
        },
    ]

    for row_start in range(0, len(architecture_steps), 2):

        columns = st.columns(2)

        row_steps = architecture_steps[
            row_start:row_start + 2
        ]

        for column, step in zip(columns, row_steps):

            with column:

                html = (
                    '<div style="'
                    'background:#0d1628;'
                    'border:1px solid #202d45;'
                    'border-radius:16px;'
                    'padding:24px;'
                    'min-height:205px;'
                    'margin-bottom:18px;'
                    '">'
                    '<div style="'
                    'display:flex;'
                    'justify-content:space-between;'
                    'align-items:center;'
                    'margin-bottom:18px;'
                    '">'
                    '<div style="'
                    'font-size:27px;'
                    '">'
                    f'{step["icon"]}'
                    '</div>'
                    '<div style="'
                    'color:#536b95;'
                    'font-size:12px;'
                    'font-weight:800;'
                    'letter-spacing:1.5px;'
                    '">'
                    f'STEP {step["number"]}'
                    '</div>'
                    '</div>'
                    '<div style="'
                    'color:#ffffff;'
                    'font-size:20px;'
                    'font-weight:800;'
                    'margin-bottom:12px;'
                    '">'
                    f'{step["title"]}'
                    '</div>'
                    '<div style="'
                    'color:#8199be;'
                    'font-size:14px;'
                    'line-height:1.8;'
                    '">'
                    f'{step["description"]}'
                    '</div>'
                    '</div>'
                )

                st.markdown(
                    html,
                    unsafe_allow_html=True,
                )

    # ========================================================
    # FRAUDSHIELD INTELLIGENCE PIPELINE
    # ========================================================

    section_title("FraudShield Intelligence Pipeline")

    pipeline_html = (
        '<div style="'
        'background:linear-gradient('
        '110deg,rgba(28,62,145,0.35),rgba(58,32,125,0.35)'
        ');'
        'border:1px solid #304677;'
        'border-radius:18px;'
        'padding:30px 20px;'
        'margin-bottom:35px;'
        '">'
        '<div style="'
        'display:flex;'
        'align-items:center;'
        'justify-content:center;'
        'flex-wrap:wrap;'
        'gap:13px;'
        'color:#ffffff;'
        'font-weight:750;'
        'font-size:14px;'
        '">'
        '<span>💳 Transactions</span>'
        '<span style="color:#5f7eaf;">→</span>'
        '<span>⚙️ Preprocessing</span>'
        '<span style="color:#5f7eaf;">→</span>'
        '<span>⚖️ SMOTE</span>'
        '<span style="color:#5f7eaf;">→</span>'
        '<span>🧠 ML Detection</span>'
        '<span style="color:#5f7eaf;">→</span>'
        '<span>🎯 Fraud Probability</span>'
        '<span style="color:#5f7eaf;">→</span>'
        '<span>🚨 Risk Intelligence</span>'
        '<span style="color:#5f7eaf;">→</span>'
        '<span>🔬 Investigation</span>'
        '</div>'
        '</div>'
    )

    st.markdown(
        pipeline_html,
        unsafe_allow_html=True,
    )

    # ========================================================
    # TECHNOLOGY STACK
    # ========================================================

    section_title("Technology Stack")

    columns = st.columns(4)

    with columns[0]:
        metric_card(
            "🐍",
            "Programming",
            "Python",
            "Core analytics and application logic",
        )

    with columns[1]:
        metric_card(
            "🤖",
            "Machine Learning",
            "Scikit-learn",
            "Fraud classification pipeline",
        )

    with columns[2]:
        metric_card(
            "📊",
            "Analytics",
            "Pandas",
            "Transaction data processing",
        )

    with columns[3]:
        metric_card(
            "🖥️",
            "Application",
            "Streamlit",
            "Interactive fraud intelligence UI",
        )

    # ========================================================
    # PROJECT CAPABILITIES
    # ========================================================

    section_title("Project Capabilities")

    capabilities = [
        (
            "🛡️",
            "Fraud Detection",
            "Machine learning classification of financial transactions.",
        ),
        (
            "🎯",
            "Fraud Probability",
            "Probability-based risk scoring for evaluated transactions.",
        ),
        (
            "📂",
            "Investigation Queue",
            "Automated prioritization of high-risk financial activity.",
        ),
        (
            "🚨",
            "Risk Intelligence",
            "Analysis of fraud probability and severity distribution.",
        ),
        (
            "📊",
            "Global Explainability",
            "Permutation-based model feature importance analysis.",
        ),
        (
            "🔬",
            "Case Explainability",
            "Transaction-level perturbation analysis for selected cases.",
        ),
        (
            "📄",
            "Case Reporting",
            "Downloadable fraud intelligence reports for investigations.",
        ),
        (
            "🧠",
            "Model Intelligence",
            "Comparative machine learning model performance analysis.",
        ),
    ]

    for row_start in range(0, len(capabilities), 4):

        columns = st.columns(4)

        row_capabilities = capabilities[
            row_start:row_start + 4
        ]

        for column, capability in zip(
            columns,
            row_capabilities,
        ):

            icon, title, description = capability

            with column:

                capability_html = (
                    '<div style="'
                    'background:#0d1628;'
                    'border:1px solid #202d45;'
                    'border-radius:15px;'
                    'padding:22px;'
                    'min-height:190px;'
                    'margin-bottom:18px;'
                    '">'
                    '<div style="'
                    'font-size:26px;'
                    'margin-bottom:16px;'
                    '">'
                    f'{icon}'
                    '</div>'
                    '<div style="'
                    'color:#ffffff;'
                    'font-size:16px;'
                    'font-weight:800;'
                    'margin-bottom:10px;'
                    '">'
                    f'{title}'
                    '</div>'
                    '<div style="'
                    'color:#7189b1;'
                    'font-size:13px;'
                    'line-height:1.7;'
                    '">'
                    f'{description}'
                    '</div>'
                    '</div>'
                )

                st.markdown(
                    capability_html,
                    unsafe_allow_html=True,
                )

    # ========================================================
    # MODEL NOTE
    # ========================================================

    section_title("Important Model Note")

    note_html = (
        '<div style="'
        'background:linear-gradient('
        '115deg,rgba(24,53,115,0.32),rgba(65,35,125,0.30)'
        ');'
        'border:1px solid #304677;'
        'border-radius:17px;'
        'padding:27px;'
        'color:#b5c8e7;'
        'font-size:15px;'
        'line-height:1.9;'
        'margin-bottom:25px;'
        '">'
        '<div style="'
        'color:#ffffff;'
        'font-size:18px;'
        'font-weight:800;'
        'margin-bottom:12px;'
        '">'
        '⚠️ Decision-Support Intelligence'
        '</div>'
        'FraudShield AI is a machine learning portfolio project. '
        'Model outputs represent decision-support intelligence generated '
        'from the project dataset.'
        '<br><br>'
        'Fraud probability should not be treated as proof of financial '
        'misconduct. Real-world financial institutions require additional '
        'transaction context, analyst investigation, governance controls, '
        'regulatory compliance and production monitoring before taking '
        'operational action.'
        '</div>'
    )

    st.markdown(
        note_html,
        unsafe_allow_html=True,
    )

    # ========================================================
    # PROJECT FOOTER
    # ========================================================

    footer_html = (
        '<div style="'
        'text-align:center;'
        'padding:35px 20px 10px 20px;'
        'color:#526b94;'
        'font-size:13px;'
        '">'
        '<div style="'
        'color:#ffffff;'
        'font-size:18px;'
        'font-weight:800;'
        'margin-bottom:8px;'
        '">'
        '🛡️ FraudShield AI'
        '</div>'
        'Machine Learning Financial Fraud Intelligence Platform'
        '<br>'
        'End-to-End Data Science & Explainable AI Project'
        '</div>'
    )

    st.markdown(
        footer_html,
        unsafe_allow_html=True,
    )


# ============================================================
# APPLICATION ROUTER
# ============================================================

if module == "Command Center":
    command_center()

elif module == "Fraud Investigation":
    fraud_investigation()

elif module == "Risk Intelligence":
    risk_intelligence()

elif module == "Explainable AI":
    explainable_ai()

elif module == "Model Performance":
    model_performance()

elif module == "About System":
    about_system()