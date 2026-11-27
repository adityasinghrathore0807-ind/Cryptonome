import pandas as pd
import pytest

from backend.ml.features.temporal import (
    calculate_temporal_features
)


def test_temporal_features():

    data = pd.DataFrame({
        "wallet_from": ["A", "A", "D"],
        "wallet_to": ["B", "C", "A"],
        "timestamp": [
            "2026-11-27 00:00:00",
            "2026-11-27 00:10:00",
            "2026-11-27 00:40:00"
        ]
    })

    result = calculate_temporal_features(data)

    wallet_a = result[result["wallet"] == "A"].iloc[0]

    assert wallet_a["interval_count"] == 2

    assert wallet_a["mean_interarrival_seconds"] == 1200

    assert wallet_a["std_interarrival_seconds"] == 600

    assert wallet_a["timing_irregularity"] == pytest.approx(0.5)

    assert wallet_a["burstiness"] == pytest.approx(-1 / 3)


def test_single_transaction():

    data = pd.DataFrame({
        "wallet_from": ["A"],
        "wallet_to": ["B"],
        "timestamp": ["2026-11-27 00:00:00"]
    })

    result = calculate_temporal_features(data)

    wallet_a = result[result["wallet"] == "A"].iloc[0]

    assert wallet_a["interval_count"] == 0
    assert wallet_a["timing_irregularity"] == 0
    assert wallet_a["burstiness"] == 0