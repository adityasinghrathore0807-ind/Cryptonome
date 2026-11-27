import numpy as np
import pandas as pd


class WalletConfidenceScorer:
    """
    Estimates confidence in anomaly predictions based on
    agreement between Isolation Forest and Autoencoder.

    This is an agreement-based confidence indicator,
    NOT a statistically calibrated probability.
    """

    def __init__(self, isolation_weight=0.5, autoencoder_weight=0.5):
        if isolation_weight < 0 or autoencoder_weight < 0:
            raise ValueError("Weights cannot be negative.")

        if isolation_weight + autoencoder_weight == 0:
            raise ValueError("At least one weight must be positive.")

        total = isolation_weight + autoencoder_weight

        self.isolation_weight = isolation_weight / total
        self.autoencoder_weight = autoencoder_weight / total

    def calculate_confidence(self, predictions):
        """
        Required columns:
        wallet
        is_anomaly_if
        is_anomaly_ae
        """

        required_columns = {
            "wallet",
            "is_anomaly_if",
            "is_anomaly_ae"
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

        for column in ["is_anomaly_if", "is_anomaly_ae"]:
            if result[column].isna().any():
                raise ValueError(f"{column} cannot contain missing values.")

            if not result[column].isin([True, False, 0, 1]).all():
                raise ValueError(f"{column} must contain boolean values.")

        isolation = result["is_anomaly_if"].astype(int)
        autoencoder = result["is_anomaly_ae"].astype(int)

        agreement = (isolation == autoencoder).astype(float)

        result["model_agreement"] = agreement

        result["confidence_score"] = (
            agreement * 100
        ).round(2)

        result["confidence_level"] = result[
            "confidence_score"
        ].apply(self._get_confidence_level)

        return result

    @staticmethod
    def _get_confidence_level(score):
        if score >= 80:
            return "High"

        if score >= 50:
            return "Medium"

        return "Low"