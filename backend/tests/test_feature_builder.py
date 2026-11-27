import pandas as pd
import networkx as nx

from backend.ml.features.feature_builder import (
    build_feature_matrix
)


def test_feature_builder():

    transactions = pd.DataFrame({

        "wallet_from": ["A", "A", "B", "C"],

        "wallet_to": ["B", "C", "C", "A"],

        "amount": [10, 20, 30, 40],

        "timestamp": [
            "2026-11-27 10:00:00",
            "2026-11-27 10:10:00",
            "2026-11-27 10:20:00",
            "2026-11-27 10:40:00"
        ],

        "source_ip": [
            "10.0.0.1",
            "10.0.0.2",
            "10.0.0.3",
            "10.0.0.4"
        ],

        "destination_ip": [
            "10.0.0.5",
            "10.0.0.6",
            "10.0.0.7",
            "10.0.0.8"
        ],

        "source_asn": ["AS1", "AS1", "AS2", "AS3"],

        "destination_asn": ["AS2", "AS3", "AS4", "AS1"],

        "source_country": ["US", "US", "GB", "IN"],

        "destination_country": ["GB", "IN", "IN", "US"]
    })

    graph = nx.DiGraph()

    graph.add_edges_from([
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
        ("C", "A")
    ])

    result = build_feature_matrix(
        transactions,
        graph
    )

    # Check wallet records.
    assert len(result) == 3

    assert result["wallet"].is_unique

    # Check behavioral features.
    assert "fan_in" in result.columns
    assert "fan_out" in result.columns

    # Check temporal features.
    assert "timing_irregularity" in result.columns
    assert "burstiness" in result.columns

    # Check network features.
    assert "asn_diversity" in result.columns
    assert "geo_inconsistency" in result.columns

    # Check graph features.
    assert "degree" in result.columns
    assert "betweenness_centrality" in result.columns

    # Check numerical data.
    numeric_data = result.select_dtypes(
        include="number"
    )

    assert numeric_data.notna().all().all()

    assert not numeric_data.isin(
        [float("inf"), float("-inf")]
    ).any().any()