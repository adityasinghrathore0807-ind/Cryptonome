from pathlib import Path

import pandas as pd

from backend.ml.models.isolation_forest import WalletIsolationForest
from backend.ml.models.autoencoder import WalletAutoencoder


def train_models(
    feature_matrix: pd.DataFrame,
    output_dir="backend/ml/models/saved",
    autoencoder_epochs=50
):
    if feature_matrix.empty:
        raise ValueError("Feature matrix cannot be empty.")

    if len(feature_matrix) < 2:
        raise ValueError("At least two wallets are required for training.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Starting Isolation Forest training...")

    isolation_forest = WalletIsolationForest()

    isolation_forest.fit(feature_matrix)

    isolation_path = output_path / "isolation_forest.pkl"

    isolation_forest.save(isolation_path)

    print("Isolation Forest training completed.")

    print("Starting Autoencoder training...")

    autoencoder = WalletAutoencoder(
        epochs=autoencoder_epochs
    )

    autoencoder.fit(feature_matrix)

    autoencoder_path = output_path / "autoencoder.pt"

    autoencoder.save(autoencoder_path)

    print("Autoencoder training completed.")

    print("Both models saved successfully.")

    return {
        "isolation_forest": isolation_forest,
        "autoencoder": autoencoder,
        "isolation_forest_path": str(isolation_path),
        "autoencoder_path": str(autoencoder_path)
    }
