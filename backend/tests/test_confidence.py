import pandas as pd
import pytest

from backend.ml.scoring.confidence import WalletConfidenceScorer


@pytest.fixture
def sample_predictions():
    return pd.DataFrame({
        "wallet": ["wallet_1", "wallet_2", "wallet_3", "wallet_4"],
        "is_anomaly_if": [1, 0, 1, 0],
        "is_anomaly_ae": [1, 0, 0, 1]
    })


def test_confidence_calculation(sample_predictions):
    scorer = WalletConfidenceScorer()

    result = scorer.calculate_confidence(sample_predictions)

    assert "confidence_score" in result.columns
    assert "confidence_level" in result.columns
    assert result["confidence_score"].tolist() == [100, 100, 0, 0]


def test_model_agreement(sample_predictions):
    scorer = WalletConfidenceScorer()

    result = scorer.calculate_confidence(sample_predictions)

    assert result["model_agreement"].tolist() == [1, 1, 0, 0]


def test_confidence_categories():
    scorer = WalletConfidenceScorer()

    assert scorer._get_confidence_level(90) == "High"
    assert scorer._get_confidence_level(60) == "Medium"
    assert scorer._get_confidence_level(20) == "Low"


def test_empty_predictions():
    scorer = WalletConfidenceScorer()

    empty = pd.DataFrame(
        columns=["wallet", "is_anomaly_if", "is_anomaly_ae"]
    )

    with pytest.raises(ValueError):
        scorer.calculate_confidence(empty)


def test_missing_columns():
    scorer = WalletConfidenceScorer()

    invalid = pd.DataFrame({
        "wallet": ["wallet_1"],
        "is_anomaly_if": [1]
    })

    with pytest.raises(ValueError):
        scorer.calculate_confidence(invalid)


def test_duplicate_wallets(sample_predictions):
    scorer = WalletConfidenceScorer()

    duplicated = pd.concat(
        [sample_predictions, sample_predictions.iloc[[0]]],
        ignore_index=True
    )

    with pytest.raises(ValueError):
        scorer.calculate_confidence(duplicated)


def test_invalid_weights():
    with pytest.raises(ValueError):
        WalletConfidenceScorer(-0.5, 0.5)

    with pytest.raises(ValueError):
        WalletConfidenceScorer(0, 0)