import networkx as nx

from backend.ml.features.graph_features import (
    calculate_graph_features
)


def test_graph_features():

    graph = nx.DiGraph()

    graph.add_edges_from([
        ("A", "B"),
        ("A", "C"),
        ("B", "C")
    ])

    graph.add_node("D")

    result = calculate_graph_features(graph)

    wallet_a = result[
        result["wallet"] == "A"
    ].iloc[0]

    wallet_c = result[
        result["wallet"] == "C"
    ].iloc[0]

    wallet_d = result[
        result["wallet"] == "D"
    ].iloc[0]

    assert wallet_a["degree"] == 2
    assert wallet_a["out_degree"] == 2
    assert wallet_a["in_degree"] == 0

    assert wallet_c["in_degree"] == 2
    assert wallet_c["out_degree"] == 0

    assert wallet_a["cluster_size"] == 3
    assert wallet_d["cluster_size"] == 1

    assert wallet_a["betweenness_centrality"] >= 0
    assert wallet_a["closeness_centrality"] >= 0


def test_empty_graph():

    graph = nx.DiGraph()

    result = calculate_graph_features(graph)

    assert result.empty