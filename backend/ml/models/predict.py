from pathlib import Path

import pandas as pd

from backend.ml.models.isolation_forest import WalletIsolationForest
from backend.ml.models.autoencoder import WalletAutoencoder


def load_models(model_dir="backend/ml/models/saved"):

    model_path = Path(model_dir)

    isolation_path = model_path / "isolation_forest.pkl"
    autoencoder_path = model_path / "autoencoder.pt"

    if not isolation_path.exists():
        raise FileNotFoundError(
            f"Isolation Forest model not found: {isolation_path}"
        )

    if not autoencoder_path.exists():
        raise FileNotFoundError(
            f"Autoencoder model not found: {autoencoder_path}"
        )

    isolation_forest = WalletIsolationForest.load(isolation_path)

    autoencoder = WalletAutoencoder.load(autoencoder_path)

    return isolation_forest, autoencoder


def predict_wallets(
    feature_matrix: pd.DataFrame,
    model_dir="backend/ml/models/saved"
):

    if feature_matrix.empty:
        raise ValueError("Feature matrix cannot be empty.")

    isolation_forest, autoencoder = load_models(model_dir)

    isolation_predictions = isolation_forest.predict(feature_matrix)

    autoencoder_predictions = autoencoder.predict(feature_matrix)

    results = isolation_predictions.merge(
        autoencoder_predictions,
        on="wallet",
        how="inner"
    )

    results["anomaly_votes"] = (
        results["is_anomaly_x"].astype(int)
        + results["is_anomaly_y"].astype(int)
    )

    results["combined_prediction"] = results["anomaly_votes"].apply(
        lambda votes: "anomaly" if votes >= 1 else "normal"
    )

    results["agreement"] = results["anomaly_votes"].apply(
        lambda votes: (
            "both_anomaly" if votes == 2
            else "both_normal" if votes == 0
            else "disagreement"
        )
    )

    results = results.rename(columns={
         "prediction_x": "isolation_prediction",
        "prediction_y": "autoencoder_prediction",
        "anomaly_score": "isolation_anomaly_score",
        "reconstruction_error": "autoencoder_reconstruction_error",
        "is_anomaly_x": "isolation_is_anomaly",
        "is_anomaly_y": "autoencoder_is_anomaly"
    })

    return results.sort_values(
        by=["anomaly_votes", "isolation_anomaly_score"],
        ascending=[False, False]
    ).reset_index(drop=True)