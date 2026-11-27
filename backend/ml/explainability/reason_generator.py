import pandas as pd


class WalletReasonGenerator:
    """
    Converts SHAP feature explanations into human-readable
    reasons for suspicious wallet activity.
    """

    FEATURE_DESCRIPTIONS = {
        "fan_in": "unusual incoming transaction activity",
        "fan_out": "unusual outgoing transaction activity",
        "transaction_frequency": "unusually high transaction frequency",
        "average_amount": "unusual transaction amount patterns",
        "interval_count": "unusual transaction timing patterns",
        "mean_interarrival_seconds": "unusual average time between transactions",
        "std_interarrival_seconds": "irregular transaction intervals",
        "timing_irregularity": "irregular transaction timing",
        "burstiness": "unusual bursts of transaction activity",
        "unique_ip_count": "connections from multiple IP addresses",
        "shared_ip_count": "IP addresses shared across multiple wallets",
        "max_ip_wallet_reuse": "high wallet reuse of an IP address",
        "asn_diversity": "connections across multiple network providers",
        "country_diversity": "connections across multiple countries",
        "geo_inconsistency": "unusual geographic activity patterns",
        "degree": "unusual wallet network connectivity",
        "in_degree": "unusual incoming network connections",
        "out_degree": "unusual outgoing network connections",
        "betweenness_centrality": "unusual intermediary network activity",
        "closeness_centrality": "unusual network proximity",
        "component_size": "unusual position in the transaction network",
    }

    def __init__(self, top_n=3, min_importance=0.0):
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

        if min_importance < 0:
            raise ValueError("min_importance cannot be negative.")

        self.top_n = top_n
        self.min_importance = min_importance

    def generate_reasons(self, explanation: pd.DataFrame):
        """
        Generates human-readable reasons from one wallet's
        SHAP explanation.
        """

        required_columns = {"feature", "shap_value", "importance"}

        if not required_columns.issubset(explanation.columns):
            raise ValueError(
                f"Explanation must contain columns: {required_columns}"
            )

        if explanation.empty:
            return []

        explanation = explanation.copy()

        explanation["importance"] = pd.to_numeric(
            explanation["importance"], errors="coerce"
        )

        explanation["shap_value"] = pd.to_numeric(
            explanation["shap_value"], errors="coerce"
        )

        explanation = explanation.dropna(
            subset=["feature", "importance", "shap_value"]
        )

        explanation = explanation[
            explanation["importance"] >= self.min_importance
        ]

        explanation = explanation.sort_values(
            "importance", ascending=False
        )

        reasons = []

        for _, row in explanation.head(self.top_n).iterrows():
            feature = row["feature"]

            description = self.FEATURE_DESCRIPTIONS.get(
                feature,
                f"unusual {feature.replace('_', ' ')}"
            )

            direction = (
                "increased"
                if row["shap_value"] > 0
                else "decreased"
                if row["shap_value"] < 0
                else "neutral"
            )

            if direction == "neutral":
                continue

            reasons.append({
                "feature": feature,
                "reason": description,
                "direction": direction,
                "importance": float(row["importance"]),
                "shap_value": float(row["shap_value"])
            })

        return reasons