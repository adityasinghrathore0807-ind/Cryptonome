import pandas as pd
import pytest

from backend.ml.models.train import train_models
from backend.ml.models.predict import predict_wallets


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


def test_prediction_pipeline(tmp_path):
    features = sample_features()

    train_models(
        feature_matrix=features,
        output_dir=tmp_path,
        autoencoder_epochs=10
    )

    predictions = predict_wallets(
        feature_matrix=features,
        model_dir=tmp_path
    )

    expected_columns = {
        "wallet",
        "isolation_prediction",
        "isolation_is_anomaly",
        "isolation_anomaly_score",
        "autoencoder_prediction",
        "autoencoder_is_anomaly",
        "autoencoder_reconstruction_error",
        "anomaly_votes",
        "combined_prediction",
        "agreement"
    }

    assert expected_columns.issubset(predictions.columns)

    assert len(predictions) == len(features)

    assert predictions["wallet"].is_unique

    assert predictions["anomaly_votes"].between(0, 2).all()

    assert set(predictions["combined_prediction"]).issubset({
        "normal",
        "anomaly"
    })


def test_empty_feature_matrix_fails():
    with pytest.raises(ValueError):
        predict_wallets(pd.DataFrame())


def test_missing_models_fail(tmp_path):
    features = sample_features()

    with pytest.raises(FileNotFoundError):
        predict_wallets(
            feature_matrix=features,
            model_dir=tmp_path
        )