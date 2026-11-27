import pandas as pd
import pytest

from backend.ml.models.train import train_models


def sample_features():
    return pd.DataFrame({
        "wallet": [f"wallet_{i}" for i in range(10)],
        "fan_in": [2, 3, 2, 4, 3, 2, 3, 4, 2, 100],
        "fan_out": [3, 2, 4, 3, 2, 3, 4, 2, 3, 150],
        "transaction_frequency": [
            5, 6, 7, 5, 8, 6, 7, 5, 6, 500
        ],
        "average_amount": [
            10, 12, 11, 9, 13, 10, 12, 11, 9, 10000
        ]
    })


def test_train_both_models(tmp_path):
    features = sample_features()

    result = train_models(
        feature_matrix=features,
        output_dir=tmp_path,
        autoencoder_epochs=10
    )

    assert result["isolation_forest"].is_fitted
    assert result["autoencoder"].is_fitted

    assert (tmp_path / "isolation_forest.pkl").exists()
    assert (tmp_path / "autoencoder.pt").exists()

    assert (
        tmp_path / "autoencoder.pt.metadata"
    ).exists()


def test_training_rejects_empty_features(tmp_path):
    features = pd.DataFrame()

    with pytest.raises(ValueError):
        train_models(features, output_dir=tmp_path)


def test_training_requires_two_wallets(tmp_path):
    features = pd.DataFrame({
        "wallet": ["wallet_A"],
        "fan_in": [2],
        "fan_out": [3]
    })

    with pytest.raises(ValueError):
        train_models(features, output_dir=tmp_path)