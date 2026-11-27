import numpy as np
import pandas as pd
import shap


class WalletSHAPExplainer:
    """
    Explains Isolation Forest anomaly scores using SHAP
    PermutationExplainer.
    """

    def __init__(self, model, background_data=None):

        if not hasattr(model, "pipeline"):
            raise ValueError(
                "Invalid model: expected a trained WalletIsolationForest."
            )

        if model.pipeline is None:
            raise ValueError(
                "Isolation Forest model has not been trained."
            )

        self.model = model
        self.feature_columns = model.feature_columns

        if not self.feature_columns:
            raise ValueError("Model has no feature columns.")

        self.background_data = background_data
        self.explainer = None

    def _prepare_data(self, data):

        missing_columns = [
            col for col in self.feature_columns
            if col not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required feature columns: {missing_columns}"
            )

        features = data[self.feature_columns].copy()

        features = features.replace(
            [np.inf, -np.inf],
            np.nan
        )

        return features

    def _predict_anomaly_score(self, values):
        """
        Returns the actual Isolation Forest anomaly score.

        Higher values indicate more anomalous observations.
        """

        values = np.asarray(values, dtype=float)

        if values.ndim == 1:
            values = values.reshape(1, -1)

        features = pd.DataFrame(
            values,
            columns=self.feature_columns
        )

        return -self.model.pipeline.decision_function(features)

    def fit(self, background_data):
        """
        Initializes the permutation-based SHAP explainer.
        """

        features = self._prepare_data(background_data)

        if features.empty:
            raise ValueError(
                "Background data cannot be empty."
            )

        self.background_data = features.to_numpy(dtype=float)

        masker = shap.maskers.Independent(
            self.background_data
        )

        self.explainer = shap.Explainer(
            self._predict_anomaly_score,
            masker,
            algorithm="permutation"
        )

        return self

    def explain(self, data):
        """
        Returns SHAP contributions for each wallet.
        """

        if self.explainer is None:
            raise ValueError(
                "Explainer is not initialized. Call fit() first."
            )

        features = self._prepare_data(data)

        if features.empty:
            raise ValueError(
                "Input data cannot be empty."
            )

        shap_results = self.explainer(
            features.to_numpy(dtype=float),
            max_evals=2 * len(self.feature_columns) + 1
        )

        explanations = pd.DataFrame(
            shap_results.values,
            columns=self.feature_columns,
            index=data.index
        )

        explanations.insert(
            0,
            "wallet",
            data["wallet"].values
        )

        return explanations

    def explain_wallet(self, wallet_data):
        """
        Returns feature contributions for one wallet.
        """

        explanations = self.explain(wallet_data)

        wallet = explanations.iloc[0]["wallet"]

        feature_values = (
            explanations
            .drop(columns=["wallet"])
            .iloc[0]
        )

        result = pd.DataFrame({
            "feature": feature_values.index,
            "shap_value": feature_values.values,
            "importance": np.abs(feature_values.values)
        })

        result = result.sort_values(
            by="importance",
            ascending=False
        ).reset_index(drop=True)

        result.insert(0, "wallet", wallet)

        return result