import pandas as pd
import pytest

from backend.ml.scoring.risk_scorer import WalletRiskScorer


@pytest.fixture
def sample_predictions():
    return pd.DataFrame({
        "wallet": ["wallet_1", "wallet_2", "wallet_3", "wallet_4"],
        "anomaly_score": [0.1, 0.3, 0.6, 0.9],
        "reconstruction_error": [0.2, 0.4, 0.7, 1.0]
    })


def test_risk_score_calculation(sample_predictions):
    scorer = WalletRiskScorer()

    result = scorer.calculate_risk_score(sample_predictions)

    assert "risk_score" in result.columns
    assert "risk_level" in result.columns

    assert result["risk_score"].between(0, 100).all()


def test_risk_score_order(sample_predictions):
    scorer = WalletRiskScorer()

    result = scorer.calculate_risk_score(sample_predictions)

    assert result["risk_score"].is_monotonic_increasing


def test_risk_categories():
    scorer = WalletRiskScorer()

    assert scorer._get_risk_level(10) == "Low"
    assert scorer._get_risk_level(30) == "Medium"
    assert scorer._get_risk_level(60) == "High"
    assert scorer._get_risk_level(90) == "Critical"


def test_empty_predictions():
    scorer = WalletRiskScorer()

    empty = pd.DataFrame(
        columns=["wallet", "anomaly_score", "reconstruction_error"]
    )

    with pytest.raises(ValueError):
        scorer.calculate_risk_score(empty)


def test_missing_columns():
    scorer = WalletRiskScorer()

    invalid = pd.DataFrame({
        "wallet": ["wallet_1"],
        "anomaly_score": [0.5]
    })

    with pytest.raises(ValueError):
        scorer.calculate_risk_score(invalid)


def test_invalid_weights():
    with pytest.raises(ValueError):
        WalletRiskScorer(-0.5, 0.5)

    with pytest.raises(ValueError):
        WalletRiskScorer(0, 0)


def test_duplicate_wallets(sample_predictions):
    scorer = WalletRiskScorer()

    duplicated = pd.concat(
        [sample_predictions, sample_predictions.iloc[[0]]],
        ignore_index=True
    )

    with pytest.raises(ValueError):
        scorer.calculate_risk_score(duplicated)