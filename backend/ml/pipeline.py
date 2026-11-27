import networkx as nx
import pandas as pd

from backend.ml.features.feature_builder import build_feature_matrix

from backend.ml.models.predict import predict_wallets, load_models

from backend.ml.scoring.risk_scorer import WalletRiskScorer
from backend.ml.scoring.confidence import WalletConfidenceScorer
from backend.ml.scoring.alert_ranker import WalletAlertRanker

from backend.ml.explainability.shap_explainer import WalletSHAPExplainer
from backend.ml.explainability.reason_generator import WalletReasonGenerator


def run_ml_pipeline(
    transactions: pd.DataFrame,
    graph: nx.Graph,
    model_dir="backend/ml/models/saved_synthetic"
):
    """
    Execute the complete ML pipeline for wallet anomaly detection.
    """

    # 1. Build wallet feature matrix
    feature_matrix = build_feature_matrix(
        transactions,
        graph
    )

    if feature_matrix.empty:
        raise ValueError("Feature matrix contains no wallets.")

    # 2. Generate model predictions
    predictions = predict_wallets(
        feature_matrix,
        model_dir=model_dir
    )

    # 3. Calculate risk scores
    risk_input = predictions.rename(columns={
        "isolation_anomaly_score": "anomaly_score",
        "autoencoder_reconstruction_error": "reconstruction_error"
    })

    risk_scorer = WalletRiskScorer()

    risk_results = risk_scorer.calculate_risk_score(
        risk_input
    )

    # 4. Calculate model agreement and confidence
    confidence_input = predictions.rename(columns={
        "isolation_is_anomaly": "is_anomaly_if",
        "autoencoder_is_anomaly": "is_anomaly_ae"
    })

    confidence_scorer = WalletConfidenceScorer()

    confidence_results = confidence_scorer.calculate_confidence(
        confidence_input
    )

    # 5. Combine risk and confidence results
    combined_results = risk_results.merge(
        confidence_results[
            [
                "wallet",
                "model_agreement",
                "confidence_score",
                "confidence_level"
            ]
        ],
        on="wallet",
        how="inner"
    )

    # 6. Rank anomalous wallet alerts
    alerts = combined_results[
        combined_results["combined_prediction"] == "anomaly"
    ].copy()

    alert_ranker = WalletAlertRanker()

    if alerts.empty:
        ranked_alerts = alerts.copy()
        ranked_alerts["priority_score"] = pd.Series(dtype=float)
        ranked_alerts["priority"] = pd.Series(dtype=str)
        ranked_alerts["alert_rank"] = pd.Series(dtype=int)
    else:
        ranked_alerts = alert_ranker.rank_alerts(alerts)

    # 7. Generate SHAP explanations for anomalous wallets
    isolation_forest, _ = load_models(model_dir)

    explainer = WalletSHAPExplainer(
        model=isolation_forest
    )

    explainer.fit(feature_matrix)

    reason_generator = WalletReasonGenerator()

    explanations = []

    for wallet in alerts["wallet"]:

        wallet_data = feature_matrix[
            feature_matrix["wallet"] == wallet
        ]

        if wallet_data.empty:
            continue

        explanation = explainer.explain_wallet(
            wallet_data
        )

        reasons = reason_generator.generate_reasons(
            explanation.drop(columns=["wallet"])
        )

        explanations.append({
            "wallet": wallet,
            "explanation": explanation,
            "reasons": reasons
        })

    # 8. Return complete pipeline results
    return {
        "feature_matrix": feature_matrix,
        "predictions": predictions,
        "risk_results": risk_results,
        "confidence_results": confidence_results,
        "ranked_alerts": ranked_alerts,
        "explanations": explanations
    }