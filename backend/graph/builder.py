
"""Unified investigation graph: wallets <-> transactions <-> IPs <-> ASNs.

Node ids : w:<wallet>  t:<txid>  ip:<ip>  asn:AS<number>
Edges    : SENT (wallet->tx), RECEIVED_BY (tx->wallet), ORIGIN_IP / DEST_IP (tx->ip),
           USED_IP (wallet->ip, with a count), IN_ASN (ip->asn)
"""
import networkx as nx
import pandas as pd


def _present(v) -> bool:
    return v is not None and not pd.isna(v)


def _add_ip(G, ip, country, city, asn, org, hosting):
    node = f"ip:{ip}"
    G.add_node(
        node,
        type="ip",
        label=ip,
        country=country or "??",
        city=city or "",
        asn=int(asn) if _present(asn) else 0,
        org=org or "",
        is_hosting=bool(hosting)
    )

    if _present(asn) and int(asn) > 0:
        asn_node = f"asn:AS{int(asn)}"
        G.add_node(asn_node, type="asn", label=f"AS{int(asn)}", org=org or "")
        if not G.has_edge(node, asn_node, "IN_ASN"):
            G.add_edge(node, asn_node, key="IN_ASN", relation="IN_ASN")

    return node


def _bump(G, u, v, key):
    if G.has_edge(u, v, key):
        G[u][v][key]["count"] += 1
    else:
        G.add_edge(u, v, key=key, relation=key, count=1)


def build_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    for r in df.itertuples(index=False):
        w_s = f"w:{r.sender}"
        w_r = f"w:{r.receiver}"
        tx = f"t:{r.txid}"

        G.add_node(w_s, type="wallet", label=r.sender)
        G.add_node(w_r, type="wallet", label=r.receiver)

        G.add_node(
            tx,
            type="tx",
            label=str(r.txid)[:12],
            txid=r.txid,
            amount_btc=float(r.amount_btc),
            timestamp=r.timestamp.isoformat()
        )

        G.add_edge(w_s, tx, key="SENT", relation="SENT",
                   amount_btc=float(r.amount_btc))
        G.add_edge(tx, w_r, key="RECEIVED_BY", relation="RECEIVED_BY",
                   amount_btc=float(r.amount_btc))

        if _present(r.src_ip):
            ip = _add_ip(
                G, r.src_ip,
                getattr(r, "src_country", None),
                getattr(r, "src_city", None),
                getattr(r, "src_asn", None),
                getattr(r, "src_org", None),
                getattr(r, "src_is_hosting", False)
            )

            if not G.has_edge(tx, ip, "ORIGIN_IP"):
                G.add_edge(
                    tx, ip, key="ORIGIN_IP",
                    relation="ORIGIN_IP",
                    port=int(r.src_port) if _present(r.src_port) else None
                )

            _bump(G, w_s, ip, "USED_IP")

        if _present(r.dst_ip):
            ip = _add_ip(
                G, r.dst_ip,
                getattr(r, "dst_country", None),
                getattr(r, "dst_city", None),
                getattr(r, "dst_asn", None),
                getattr(r, "dst_org", None),
                getattr(r, "dst_is_hosting", False)
            )

            if not G.has_edge(tx, ip, "DEST_IP"):
                G.add_edge(
                    tx, ip, key="DEST_IP",
                    relation="DEST_IP",
                    port=int(r.dst_port) if _present(r.dst_port) else None
                )

    return G


def wallet_projection(df: pd.DataFrame) -> nx.DiGraph:
    """Wallet -> wallet fund-flow graph (weight = BTC moved, count = number of transfers)."""

    edges = (
        df.groupby(["sender", "receiver"])
        .agg(
            weight=("amount_btc", "sum"),
            count=("amount_btc", "size")
        )
        .reset_index()
    )

    return nx.from_pandas_edgelist(
        edges,
        "sender",
        "receiver",
        edge_attr=["weight", "count"],
        create_using=nx.DiGraph
    )
