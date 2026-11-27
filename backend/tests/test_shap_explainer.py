import pandas as pd
import pytest

from backend.ml.models.isolation_forest import WalletIsolationForest
from backend.ml.explainability.shap_explainer import WalletSHAPExplainer


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "wallet": [f"wallet_{i}" for i in range(1, 21)],
        "fan_in": [2, 3, 4, 2, 5, 3, 4, 2, 3, 5,
                   4, 3, 2, 5, 4, 3, 2, 4, 5, 3],
        "fan_out": [3, 2, 5, 4, 2, 3, 4, 5, 2, 3,
                    4, 2, 5, 3, 4, 2, 3, 5, 4, 2],
        "transaction_frequency": [5, 8, 6, 4, 9, 7, 5, 6, 8, 4,
                                  7, 5, 6, 9, 4, 8, 5, 7, 6, 4],
        "average_amount": [100, 200, 150, 300, 120, 180, 250, 130, 220, 170,
                           190, 140, 260, 110, 230, 160, 210, 125, 280, 175]
    })


def test_shap_explainer_initialization(sample_data):
    model = WalletIsolationForest()
    model.fit(sample_data)

    explainer = WalletSHAPExplainer(model)

    assert explainer.model is model
    assert explainer.feature_columns == [
        "fan_in",
        "fan_out",
        "transaction_frequency",
        "average_amount"
    ]


def test_shap_explanations(sample_data):
    model = WalletIsolationForest()
    model.fit(sample_data)

    explainer = WalletSHAPExplainer(model)
    explainer.fit(sample_data)

    result = explainer.explain(sample_data)

    assert len(result) == len(sample_data)
    assert "wallet" in result.columns
    assert len(result.columns) == 5


def test_explain_single_wallet(sample_data):
    model = WalletIsolationForest()
    model.fit(sample_data)

    explainer = WalletSHAPExplainer(model)
    explainer.fit(sample_data)

    result = explainer.explain_wallet(sample_data.iloc[[0]])

    assert len(result) == 4
    assert "feature" in result.columns
    assert "shap_value" in result.columns
    assert "importance" in result.columns


def test_explain_before_fit_fails(sample_data):
    model = WalletIsolationForest()
    model.fit(sample_data)

    explainer = WalletSHAPExplainer(model)

    with pytest.raises(ValueError):
        explainer.explain(sample_data)


def test_missing_feature_column_fails(sample_data):
    model = WalletIsolationForest()
    model.fit(sample_data)

    explainer = WalletSHAPExplainer(model)

    invalid_data = sample_data.drop(columns=["fan_in"])

    with pytest.raises(ValueError):
        explainer._prepare_data(invalid_data)