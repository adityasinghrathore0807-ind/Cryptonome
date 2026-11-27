import pandas as pd

from backend.ml.features.network import (
    calculate_network_features
)


def test_network_features():

    data = pd.DataFrame({
        "wallet_from": ["A", "C", "A"],
        "wallet_to": ["B", "A", "D"],

        "source_ip": [
            "10.0.0.1",
            "10.0.0.1",
            "10.0.0.4"
        ],

        "destination_ip": [
            "10.0.0.2",
            "10.0.0.3",
            "10.0.0.5"
        ],

        "source_asn": ["AS1", "AS1", "AS4"],
        "destination_asn": ["AS2", "AS3", "AS5"],

        "source_country": ["US", "US", "GB"],
        "destination_country": ["DE", "FR", "GB"]
    })

    result = calculate_network_features(data)

    wallet_a = result[
        result["wallet"] == "A"
    ].iloc[0]

    assert wallet_a["unique_ip_count"] == 3
    assert wallet_a["shared_ip_count"] == 1
    assert wallet_a["max_ip_wallet_reuse"] == 2
    assert wallet_a["asn_diversity"] == 3
    assert wallet_a["country_diversity"] == 3
    assert wallet_a["geo_inconsistency"] == 2


def test_missing_network_metadata():

    data = pd.DataFrame({
        "wallet_from": ["A"],
        "wallet_to": ["B"]
    })

    result = calculate_network_features(data)

    wallet_a = result[
        result["wallet"] == "A"
    ].iloc[0]

    assert wallet_a["unique_ip_count"] == 0
    assert wallet_a["shared_ip_count"] == 0
    assert wallet_a["asn_diversity"] == 0
    assert wallet_a["geo_inconsistency"] == 0