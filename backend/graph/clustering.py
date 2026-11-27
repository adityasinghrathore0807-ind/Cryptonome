
"""Community / cluster detection over wallets, transactions and shared IPs (Louvain)."""
import networkx as nx

import config


def detect_communities(G, seed: int = config.RANDOM_STATE) -> dict:
    """Returns {wallet_address: cluster_id}. Cluster 0 is the largest. ASN nodes are ignored."""
    H = nx.Graph()
    for u, v in G.edges():
        if G.nodes[u].get("type") == "asn" or G.nodes[v].get("type") == "asn":
            continue
        H.add_edge(u, v)

    if H.number_of_nodes() == 0:
        return {}

    communities = nx.community.louvain_communities(H, seed=seed)
    mapping = {}

    for cid, members in enumerate(sorted(communities, key=len, reverse=True)):
        for n in members:
            if G.nodes[n].get("type") == "wallet":
                mapping[G.nodes[n]["label"]] = cid

    return mapping
