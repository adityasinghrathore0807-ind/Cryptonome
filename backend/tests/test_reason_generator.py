import pandas as pd
import pytest

from backend.ml.explainability.reason_generator import WalletReasonGenerator


def test_generate_reasons():
    explanation = pd.DataFrame({
        "feature": [
            "transaction_frequency",
            "average_amount",
            "fan_out",
            "fan_in"
        ],
        "shap_value": [0.85, 0.62, 0.31, 0.08],
        "importance": [0.85, 0.62, 0.31, 0.08]
    })

    generator = WalletReasonGenerator(top_n=3)

    reasons = generator.generate_reasons(explanation)

    assert len(reasons) == 3
    assert reasons[0]["feature"] == "transaction_frequency"
    assert reasons[0]["direction"] == "increased"


def test_empty_explanation():
    generator = WalletReasonGenerator()

    explanation = pd.DataFrame(
        columns=["feature", "shap_value", "importance"]
    )

    assert generator.generate_reasons(explanation) == []


def test_missing_columns():
    generator = WalletReasonGenerator()

    explanation = pd.DataFrame({
        "feature": ["fan_in"],
        "importance": [0.5]
    })

    with pytest.raises(ValueError):
        generator.generate_reasons(explanation)


def test_top_n_limit():
    generator = WalletReasonGenerator(top_n=2)

    explanation = pd.DataFrame({
        "feature": ["fan_in", "fan_out", "average_amount"],
        "shap_value": [0.5, 0.4, 0.3],
        "importance": [0.5, 0.4, 0.3]
    })

    reasons = generator.generate_reasons(explanation)

    assert len(reasons) == 2


def test_unknown_feature():
    generator = WalletReasonGenerator()

    explanation = pd.DataFrame({
        "feature": ["custom_feature"],
        "shap_value": [0.5],
        "importance": [0.5]
    })

    reasons = generator.generate_reasons(explanation)

    assert reasons[0]["reason"] == "unusual custom feature"