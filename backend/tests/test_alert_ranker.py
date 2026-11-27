import pandas as pd
import pytest

from backend.ml.scoring.alert_ranker import WalletAlertRanker


@pytest.fixture
def sample_alerts():
    return pd.DataFrame({
        "wallet": ["wallet_1", "wallet_2", "wallet_3", "wallet_4"],
        "risk_score": [30, 90, 60, 80],
        "confidence_score": [80, 95, 70, 60],
        "risk_level": ["Medium", "Critical", "High", "Critical"]
    })


def test_alert_ranking(sample_alerts):
    ranker = WalletAlertRanker()

    result = ranker.rank_alerts(sample_alerts)

    assert "alert_rank" in result.columns
    assert "priority_score" in result.columns
    assert "priority" in result.columns

    assert result["alert_rank"].tolist() == [1, 2, 3, 4]


def test_highest_priority_first(sample_alerts):
    ranker = WalletAlertRanker()

    result = ranker.rank_alerts(sample_alerts)

    assert result.iloc[0]["wallet"] == "wallet_2"


def test_priority_categories():
    ranker = WalletAlertRanker()

    assert ranker._get_priority(90) == "Critical"
    assert ranker._get_priority(75) == "High"
    assert ranker._get_priority(50) == "Medium"
    assert ranker._get_priority(20) == "Low"


def test_empty_alerts():
    ranker = WalletAlertRanker()

    empty = pd.DataFrame(
        columns=["wallet", "risk_score", "confidence_score", "risk_level"]
    )

    with pytest.raises(ValueError):
        ranker.rank_alerts(empty)


def test_missing_columns():
    ranker = WalletAlertRanker()

    invalid = pd.DataFrame({
        "wallet": ["wallet_1"],
        "risk_score": [90]
    })

    with pytest.raises(ValueError):
        ranker.rank_alerts(invalid)


def test_invalid_scores(sample_alerts):
    ranker = WalletAlertRanker()

    sample_alerts.loc[0, "risk_score"] = 150

    with pytest.raises(ValueError):
        ranker.rank_alerts(sample_alerts)


def test_duplicate_wallets(sample_alerts):
    ranker = WalletAlertRanker()

    duplicated = pd.concat(
        [sample_alerts, sample_alerts.iloc[[0]]],
        ignore_index=True
    )

    with pytest.raises(ValueError):
        ranker.rank_alerts(duplicated)