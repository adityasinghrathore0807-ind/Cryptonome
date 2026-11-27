import pandas as pd
import pytest

from backend.ml.features.behavioral import (
    calculate_behavioral_features
)


def test_behavioral_features():

    data = pd.DataFrame({
        "wallet_from": ["A", "A", "B", "C"],
        "wallet_to": ["B", "C", "C", "A"],
        "amount": [10, 20, 30, 40]
    })

    result = calculate_behavioral_features(data)

    wallet_a = result[result["wallet"] == "A"].iloc[0]

    assert wallet_a["fan_in"] == 1
    assert wallet_a["fan_out"] == 2
    assert wallet_a["transaction_frequency"] == 3
    assert wallet_a["average_amount"] == pytest.approx(23.3333333333)