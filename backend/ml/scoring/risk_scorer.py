import numpy as np
import pandas as pd


class WalletRiskScorer:
    """
    Combines Isolation Forest and Autoencoder anomaly scores
    into a normalized wallet risk score between 0 and 100.
    """

    def __init__(
        self,
        isolation_weight=0.5,
        autoencoder_weight=0.5
    ):
        if isolation_weight < 0 or autoencoder_weight < 0:
            raise ValueError("Weights cannot be negative.")

        if isolation_weight + autoencoder_weight == 0:
            raise ValueError("At least one weight must be positive.")

        total = isolation_weight + autoencoder_weight

        self.isolation_weight = isolation_weight / total
        self.autoencoder_weight = autoencoder_weight / total

    @staticmethod
    def _normalize_scores(scores):
        """
        Min-max normalizes scores into the range 0 to 1.
        """

        scores = pd.to_numeric(scores, errors="coerce")

        if scores.isna().any():
            raise ValueError("Anomaly scores cannot contain missing values.")

        if not np.isfinite(scores).all():
            raise ValueError("Anomaly scores must be finite.")

        minimum = scores.min()
        maximum = scores.max()

        if maximum == minimum:
            return pd.Series(
                0.0,
                index=scores.index
            )

        return (scores - minimum) / (maximum - minimum)

    def calculate_risk_score(self, predictions):
        """
        Calculates a combined risk score for each wallet.

        Required columns:
        wallet
        anomaly_score
        reconstruction_error
        """

        required_columns = {
            "wallet",
            "anomaly_score",
            "reconstruction_error"
        }

        if not required_columns.issubset(predictions.columns):
            raise ValueError(
                f"Missing required columns: {required_columns}"
            )

        if predictions.empty:
            raise ValueError("Prediction data cannot be empty.")

        if predictions["wallet"].isna().any():
            raise ValueError("Wallet identifiers cannot be missing.")

        if predictions["wallet"].duplicated().any():
            raise ValueError("Duplicate wallet identifiers found.")

        result = predictions.copy()

        isolation_scores = self._normalize_scores(
            result["anomaly_score"]
        )

        autoencoder_scores = self._normalize_scores(
            result["reconstruction_error"]
        )

        result["isolation_risk"] = isolation_scores
        result["autoencoder_risk"] = autoencoder_scores

        combined_score = (
            isolation_scores * self.isolation_weight
            + autoencoder_scores * self.autoencoder_weight
        )

        result["risk_score"] = (
            combined_score * 100
        ).round(2)

        result["risk_level"] = result["risk_score"].apply(
            self._get_risk_level
        )

        return result

    @staticmethod
    def _get_risk_level(score):
        if score < 25:
            return "Low"

        if score < 50:
            return "Medium"

        if score < 75:
            return "High"

        return "Critical"