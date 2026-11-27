
from graph.builder import build_graph, wallet_projection
from graph.clustering import detect_communities
from graph.exporter import ego_json, graph_to_json, json_to_graph
from graph.queries import (
    shortest_path,
    trace_forward,
    wallet_trail,
    wallets_sharing_ip,
)
from sample_data import make_sample_df


def test_graph_structure(tmp_path):
    df = make_sample_df(tmp_path)
    G = build_graph(df)

    types = [d["type"] for _, d in G.nodes(data=True)]

    assert types.count("wallet") == 6
    assert types.count("tx") == 5
    assert G.has_edge("w:W1", "t:a1", "SENT")
    assert G.has_edge("t:a1", "w:W2", "RECEIVED_BY")
    assert G.has_edge("t:a1", "ip:10.1.0.5", "ORIGIN_IP")
    assert G.nodes["ip:10.4.0.7"]["is_hosting"] is True
    assert G.nodes["ip:10.4.0.7"]["country"] == "DE"
    assert G["w:W1"]["ip:10.1.0.5"]["USED_IP"]["count"] == 2


def test_queries(tmp_path):
    df = make_sample_df(tmp_path)
    G = build_graph(df)
    WG = wallet_projection(df)

    assert shortest_path(G, "w:W1", "w:W6") is not None

    hops = trace_forward(WG, "W1", max_depth=3)
    assert {h["to"] for h in hops} >= {"W2", "W5", "W6"}

    assert len(wallet_trail(G, "W1")) == 3
    assert wallets_sharing_ip(G, "W1") == {}

    assert detect_communities(G).keys() == {
        "W1", "W2", "W3", "W4", "W5", "W6"
    }


def test_json_roundtrip(tmp_path):
    df = make_sample_df(tmp_path)
    G = build_graph(df)

    data = graph_to_json(G, risk_map={"W1": 0.9})

    assert next(
        n for n in data["nodes"] if n["id"] == "w:W1"
    )["risk"] == 0.9

    G2 = json_to_graph(data)

    assert G2.number_of_nodes() == G.number_of_nodes()
    assert G2.number_of_edges() == G.number_of_edges()

    ego = ego_json(data, "W6", hops=2)

    assert ego and any(n["id"] == "w:W5" for n in ego["nodes"])
    assert ego_json(data, "nope") is None
