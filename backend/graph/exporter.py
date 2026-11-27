
"""Graph <-> JSON (nodes/edges) for the D3.js front-end and the API."""
from typing import Dict, Iterable, Optional

import networkx as nx

import config
from graph.queries import k_hop_subgraph, wallet_node
from utils.helpers import write_json

_SCALAR = (str, int, float, bool, type(None))
_NODE_SKIP = {"type", "label"}
_EDGE_SKIP = {"relation"}


def graph_to_json(G, risk_map: Optional[Dict[str, float]] = None) -> dict:
    nodes = []
    for n, d in G.nodes(data=True):
        item = {"id": n, "type": d.get("type"), "label": d.get("label", n)}
        item.update({
            k: v for k, v in d.items()
            if k not in _NODE_SKIP and isinstance(v, _SCALAR)
        })
        if risk_map is not None and d.get("type") == "wallet":
            item["risk"] = risk_map.get(d["label"])
        nodes.append(item)

    edges = []
    for u, v, key, d in G.edges(keys=True, data=True):
        item = {"source": u, "target": v, "type": key}
        item.update({
            k: val for k, val in d.items()
            if k not in _EDGE_SKIP and isinstance(val, _SCALAR)
        })
        edges.append(item)

    return {"nodes": nodes, "edges": edges, "truncated": False}


def select_alert_subgraph(
    G, wallets: Iterable[str], hops: int = config.EGO_HOPS
) -> nx.MultiDiGraph:
    core = nx.Graph(
        G.subgraph([
            n for n, d in G.nodes(data=True)
            if d.get("type") != "asn"
        ])
    )
    keep = set()

    for w in wallets:
        node = wallet_node(w)
        if node in core:
            keep |= set(
                nx.single_source_shortest_path_length(
                    core, node, cutoff=hops
                )
            )

    return G.subgraph(keep).copy()


def export_graph(G, path, risk_map=None, alert_wallets=()) -> dict:
    """Write graph.json. Large graphs are reduced to alerted wallet neighbourhoods."""
    truncated = G.number_of_nodes() > config.MAX_EXPORT_NODES
    target = (
        select_alert_subgraph(G, alert_wallets)
        if truncated and alert_wallets else G
    )
    data = graph_to_json(target, risk_map)
    data["truncated"] = truncated
    write_json(path, data)
    return data


def json_to_graph(data: dict) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    for n in data["nodes"]:
        G.add_node(
            n["id"],
            **{k: v for k, v in n.items() if k != "id"}
        )

    for e in data["edges"]:
        attrs = {
            k: v for k, v in e.items()
            if k not in ("source", "target", "type")
        }
        G.add_edge(
            e["source"],
            e["target"],
            key=e["type"],
            relation=e["type"],
            **attrs
        )

    return G


def ego_json(
    data: dict, wallet: str, hops: int = config.EGO_HOPS
) -> Optional[dict]:
    G = json_to_graph(data)
    node = wallet_node(wallet)

    if node not in G:
        return None

    return graph_to_json(k_hop_subgraph(G, node, hops))
