from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class WalletIsolationForest:

    def __init__(
        self,
        contamination="auto",
        n_estimators=100,
        random_state=42
    ):
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                IsolationForest(
                    contamination=contamination,
                    n_estimators=n_estimators,
                    random_state=random_state
                )
            )
        ])

        self.feature_columns = None
        self.is_fitted = False

    def fit(self, feature_matrix: pd.DataFrame):
        if "wallet" not in feature_matrix.columns:
            raise ValueError("Feature matrix must contain a 'wallet' column.")

        if len(feature_matrix) < 2:
            raise ValueError("At least two wallets are required for training.")

        if feature_matrix["wallet"].isna().any():
            raise ValueError("Wallet column cannot contain missing values.")

        if feature_matrix["wallet"].duplicated().any():
            raise ValueError("Each wallet must appear only once.")

        numeric_features = feature_matrix.select_dtypes(
            include="number"
        )

        if numeric_features.empty:
            raise ValueError("No numeric features available for training.")

        self.feature_columns = numeric_features.columns.tolist()

        self.pipeline.fit(numeric_features)

        self.is_fitted = True

        return self

    def predict(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction.")

        if "wallet" not in feature_matrix.columns:
            raise ValueError("Feature matrix must contain a 'wallet' column.")

        missing_columns = set(self.feature_columns) - set(
            feature_matrix.columns
        )

        if missing_columns:
            raise ValueError(
                f"Missing model features: {sorted(missing_columns)}"
            )

        features = feature_matrix[self.feature_columns]

        raw_predictions = self.pipeline.predict(features)

        # Lower decision_function values indicate more unusual samples.
        # Negating the value makes higher scores indicate greater anomaly.
        anomaly_scores = -self.pipeline.decision_function(features)

        results = pd.DataFrame({
            "wallet": feature_matrix["wallet"].values,
            "prediction": [
                "anomaly" if value == -1 else "normal"
                for value in raw_predictions
            ],
            "is_anomaly": raw_predictions == -1,
            "anomaly_score": anomaly_scores
        })

        return results

    def save(self, path):
        if not self.is_fitted:
            raise RuntimeError("Cannot save an untrained model.")

        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, model_path)

    @classmethod
    def load(cls, path):
        model = joblib.load(path)

        if not isinstance(model, cls):
            raise TypeError("The saved file does not contain a WalletIsolationForest model.")

        return model