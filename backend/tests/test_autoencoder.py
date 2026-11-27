import pandas as pd
import pytest

from backend.ml.models.autoencoder import WalletAutoencoder


def sample_features():
    return pd.DataFrame({
        "wallet": [f"wallet_{i}" for i in range(20)],
        "fan_in": [
            2, 3, 2, 4, 3, 2, 3, 4, 2, 3,
            2, 4, 3, 2, 3, 4, 2, 3, 2, 100
        ],
        "fan_out": [
            3, 2, 4, 3, 2, 3, 4, 2, 3, 2,
            4, 3, 2, 3, 4, 2, 3, 2, 3, 150
        ],
        "transaction_frequency": [
            5, 6, 7, 5, 8, 6, 7, 5, 6, 8,
            5, 7, 6, 8, 5, 6, 7, 5, 8, 500
        ],
        "average_amount": [
            10, 12, 11, 9, 13, 10, 12, 11, 9, 13,
            10, 12, 11, 9, 13, 10, 12, 11, 9, 10000
        ]
    })


def test_autoencoder_training_and_prediction():
    features = sample_features()

    model = WalletAutoencoder(epochs=10)
    model.fit(features)

    predictions = model.predict(features)

    assert len(predictions) == len(features)

    assert {
        "wallet",
        "reconstruction_error",
        "is_anomaly",
        "prediction"
    }.issubset(predictions.columns)

    assert predictions["reconstruction_error"].notna().all()

    assert (predictions["reconstruction_error"] >= 0).all()

    assert predictions["is_anomaly"].dtype == bool


def test_autoencoder_save_and_load(tmp_path):
    features = sample_features()

    model = WalletAutoencoder(epochs=10)
    model.fit(features)

    model_path = tmp_path / "autoencoder.pt"

    model.save(model_path)

    loaded_model = WalletAutoencoder.load(model_path)

    original_predictions = model.predict(features)
    loaded_predictions = loaded_model.predict(features)

    pd.testing.assert_frame_equal(
        original_predictions,
        loaded_predictions
    )


def test_prediction_before_training_fails():
    model = WalletAutoencoder()

    with pytest.raises(RuntimeError):
        model.predict(sample_features())


def test_training_requires_wallet_column():
    features = pd.DataFrame({
        "fan_in": [1, 2],
        "fan_out": [2, 3]
    })

    model = WalletAutoencoder()

    with pytest.raises(ValueError):
        model.fit(features)


def test_training_requires_at_least_two_wallets():
    features = pd.DataFrame({
        "wallet": ["wallet_A"],
        "fan_in": [2],
        "fan_out": [3]
    })

    model = WalletAutoencoder()

    with pytest.raises(ValueError):
        model.fit(features)