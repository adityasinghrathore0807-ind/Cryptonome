
"""Investigator-style graph queries."""
from collections import deque
from typing import Dict, List, Optional

import networkx as nx


def wallet_node(wallet: str) -> str:
    return f"w:{wallet}"


def neighbors(G, node: str, types=None) -> List[str]:
    found = set(G.successors(node)) | set(G.predecessors(node))
    if types:
        found = {n for n in found if G.nodes[n].get("type") in types}
    return sorted(found)


def k_hop_subgraph(G, node: str, k: int = 2, exclude_types=("asn",)) -> nx.MultiDiGraph:
    """Everything within k hops of node (ASN hubs are excluded)."""
    if node not in G:
        raise KeyError(node)
    keep = [n for n, d in G.nodes(data=True) if d.get("type") not in exclude_types]
    return nx.ego_graph(G.subgraph(keep), node, radius=k, undirected=True)


def shortest_path(G, source: str, target: str) -> Optional[List[str]]:
    try:
        return nx.shortest_path(G.to_undirected(as_view=True), source, target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def trace_forward(WG: nx.DiGraph, wallet: str, max_depth: int = 4) -> List[dict]:
    """Follow the money downstream using BFS."""
    if wallet not in WG:
        return []

    hops, seen, queue = [], {wallet}, deque([(wallet, 0)])

    while queue:
        cur, depth = queue.popleft()
        if depth >= max_depth:
            continue

        for nxt in WG.successors(cur):
            e = WG[cur][nxt]
            hops.append({
                "depth": depth + 1,
                "from": cur,
                "to": nxt,
                "amount_btc": e.get("weight", 0.0),
                "count": e.get("count", 1)
            })

            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, depth + 1))

    return hops


def wallets_sharing_ip(G, wallet: str) -> Dict[str, List[str]]:
    """Return IP addresses and other wallets that used the same IP."""
    node, result = wallet_node(wallet), {}

    if node not in G:
        return result

    for ip in G.successors(node):
        if G.nodes[ip].get("type") != "ip" or not G.has_edge(node, ip, "USED_IP"):
            continue

        others = [
            G.nodes[w]["label"]
            for w in G.predecessors(ip)
            if w != node
            and G.nodes[w].get("type") == "wallet"
            and G.has_edge(w, ip, "USED_IP")
        ]

        if others:
            result[G.nodes[ip]["label"]] = sorted(others)

    return result


def wallet_trail(G, wallet: str, limit: int = 50) -> List[dict]:
    """Return wallet -> transaction -> IP trail."""
    node, trail = wallet_node(wallet), []

    if node not in G:
        return trail

    for tx in G.successors(node):
        if G.nodes[tx].get("type") != "tx" or not G.has_edge(node, tx, "SENT"):
            continue

        receivers = [
            G.nodes[r]["label"]
            for r in G.successors(tx)
            if G.nodes[r].get("type") == "wallet"
        ]

        ips = [
            G.nodes[i]["label"]
            for i in G.successors(tx)
            if G.nodes[i].get("type") == "ip"
        ]

        trail.append({
            "txid": G.nodes[tx]["txid"],
            "timestamp": G.nodes[tx].get("timestamp"),
            "amount_btc": G.nodes[tx].get("amount_btc"),
            "receivers": receivers,
            "ips": ips
        })

    trail.sort(key=lambda t: t["timestamp"] or "")
    return trail[:limit]
