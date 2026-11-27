from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st

from backend.ml.pipeline import run_ml_pipeline


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Cryptonome | Wallet Intelligence",
    page_icon="🔍",
    layout="wide"
)

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "backend/data/transactions.csv"

MODEL_DIR = ROOT_DIR / "backend/ml/models/saved_synthetic"


# --------------------------------------------------
# LOAD DATA AND RUN ML PIPELINE
# --------------------------------------------------

@st.cache_data(show_spinner="Running wallet anomaly detection...")
def load_analysis():

    transactions = pd.read_csv(DATA_PATH)

    graph = nx.from_pandas_edgelist(
        transactions,
        source="wallet_from",
        target="wallet_to",
        create_using=nx.DiGraph()
    )

    results = run_ml_pipeline(
        transactions,
        graph,
        model_dir=str(MODEL_DIR)
    )

    return transactions, results


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🔍 Cryptonome")
st.subheader("Blockchain Wallet Intelligence Dashboard")

st.caption(
    "Machine learning-powered wallet anomaly detection, "
    "risk scoring and explainable AI."
)

st.divider()


# --------------------------------------------------
# LOAD PIPELINE
# --------------------------------------------------

try:
    transactions, results = load_analysis()

except Exception as error:
    st.error(f"Unable to load the ML pipeline: {error}")
    st.stop()


predictions = results["predictions"]
risk_results = results["risk_results"]
ranked_alerts = results["ranked_alerts"]
confidence_results = results["confidence_results"]
feature_matrix = results["feature_matrix"]
explanations = results["explanations"]


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Cryptonome")
st.sidebar.caption("Wallet Intelligence System")

if st.sidebar.button("🔄 Refresh Analysis"):
    load_analysis.clear()
    st.rerun()

st.sidebar.success("ML Pipeline Loaded")

st.sidebar.write(f"Transactions: {len(transactions)}")
st.sidebar.write(f"Wallets: {len(feature_matrix)}")
st.sidebar.write(f"Anomalies: {len(ranked_alerts)}")


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

overview_tab, wallet_tab, alerts_tab, explain_tab = st.tabs([
    "📊 Overview",
    "👛 Wallet Analysis",
    "🚨 Anomaly Alerts",
    "🧠 Explainability"
])


# ==================================================
# TAB 1: OVERVIEW
# ==================================================

with overview_tab:

    st.header("System Overview")

    total_wallets = len(feature_matrix)
    total_transactions = len(transactions)
    total_anomalies = len(ranked_alerts)

    average_risk = (
        risk_results["risk_score"].mean()
        if not risk_results.empty
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Wallets", total_wallets)
    col2.metric("Transactions", total_transactions)
    col3.metric("Anomalous Wallets", total_anomalies)
    col4.metric("Average Risk Score", f"{average_risk:.2f}")

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Risk Level Distribution")

        risk_distribution = (
            risk_results["risk_level"]
            .value_counts()
            .reindex(
                ["Low", "Medium", "High", "Critical"],
                fill_value=0
            )
        )

        st.bar_chart(risk_distribution)

    with right:

        st.subheader("Model Prediction Distribution")

        prediction_distribution = (
            predictions["combined_prediction"]
            .value_counts()
        )

        st.bar_chart(prediction_distribution)

    st.divider()

    st.subheader("Highest Risk Wallets")

    highest_risk = risk_results.sort_values(
        "risk_score",
        ascending=False
    ).head(10)

    st.dataframe(
        highest_risk[
            ["wallet", "risk_score", "risk_level"]
        ],
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# TAB 2: WALLET ANALYSIS
# ==================================================

with wallet_tab:

    st.header("Individual Wallet Analysis")

    wallet_list = sorted(feature_matrix["wallet"].tolist())

    selected_wallet = st.selectbox(
        "Select a wallet",
        wallet_list
    )

    wallet_risk = risk_results[
        risk_results["wallet"] == selected_wallet
    ]

    wallet_prediction = predictions[
        predictions["wallet"] == selected_wallet
    ]

    wallet_confidence = confidence_results[
        confidence_results["wallet"] == selected_wallet
    ]

    if not wallet_risk.empty:

        risk_row = wallet_risk.iloc[0]
        prediction_row = wallet_prediction.iloc[0]
        confidence_row = wallet_confidence.iloc[0]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Risk Score",
            f"{risk_row['risk_score']:.2f}/100"
        )

        col2.metric(
            "Risk Level",
            risk_row["risk_level"]
        )

        col3.metric(
            "Confidence",
            f"{confidence_row['confidence_score']:.2f}"
        )

        st.subheader("Model Predictions")

        st.write(
            "Isolation Forest:",
            prediction_row["isolation_prediction"]
        )

        st.write(
            "Autoencoder:",
            prediction_row["autoencoder_prediction"]
        )

        st.write(
            "Combined Prediction:",
            prediction_row["combined_prediction"]
        )

        st.write(
            "Model Agreement:",
            confidence_row["model_agreement"]
        )

        st.divider()

        st.subheader("Wallet Features")

        wallet_features = feature_matrix[
            feature_matrix["wallet"] == selected_wallet
        ]

        st.dataframe(
            wallet_features,
            use_container_width=True,
            hide_index=True
        )


# ==================================================
# TAB 3: ANOMALY ALERTS
# ==================================================

with alerts_tab:

    st.header("Ranked Anomaly Alerts")

    if ranked_alerts.empty:

        st.success("No anomalous wallets detected.")

    else:

        st.metric(
            "Total Active Anomaly Alerts",
            len(ranked_alerts)
        )

        priority_filter = st.multiselect(
            "Filter by priority",
            options=sorted(
                ranked_alerts["priority"].dropna().unique()
            ),
            default=sorted(
                ranked_alerts["priority"].dropna().unique()
            )
        )

        filtered_alerts = ranked_alerts[
            ranked_alerts["priority"].isin(priority_filter)
        ]

        st.dataframe(
            filtered_alerts[
                [
                    "alert_rank",
                    "wallet",
                    "risk_score",
                    "risk_level",
                    "priority_score",
                    "priority",
                    "confidence_score",
                    "model_agreement"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


# ==================================================
# TAB 4: EXPLAINABILITY
# ==================================================

with explain_tab:

    st.header("Explainable AI")

    st.write(
        "Understand which features contributed to "
        "the anomaly detection of each suspicious wallet."
    )

    if not explanations:

        st.info("No wallet explanations are available.")

    else:

        explained_wallets = [
            item["wallet"] for item in explanations
        ]

        selected_explanation = st.selectbox(
            "Select an anomalous wallet",
            explained_wallets
        )

        explanation_data = next(
            item for item in explanations
            if item["wallet"] == selected_explanation
        )

        shap_data = explanation_data["explanation"]

        reasons = explanation_data["reasons"]

        st.subheader(
            f"SHAP Feature Importance — {selected_explanation}"
        )

        shap_chart = (
            shap_data
            .set_index("feature")["importance"]
            .sort_values(ascending=True)
        )

        st.bar_chart(shap_chart)

        st.divider()

        st.subheader("Generated Anomaly Reasons")

        if reasons:

            reasons_df = pd.DataFrame(reasons)

            st.dataframe(
                reasons_df[
                    [
                        "feature",
                        "reason",
                        "direction",
                        "importance",
                        "shap_value"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info("No explanation reasons generated.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Cryptonome | ML + Explainable AI | "
    "Blockchain Wallet Anomaly Detection"
)