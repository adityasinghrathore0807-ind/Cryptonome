import networkx as nx
import pandas as pd


def calculate_graph_features(graph: nx.DiGraph) -> pd.DataFrame:
    """
    Calculate graph-based features for Bitcoin wallets.

    Input:
        NetworkX directed graph.
        Nodes represent wallets.
        Directed edges represent transactions.

    Output:
        Wallet-level graph feature DataFrame.
    """

    if not isinstance(graph, nx.Graph):
        raise TypeError("Input must be a NetworkX graph.")

    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=[
            "wallet",
            "degree",
            "in_degree",
            "out_degree",
            "betweenness_centrality",
            "closeness_centrality",
            "cluster_size"
        ])

    # Convert graph into an undirected graph for
    # centrality and connected-component calculations.
    undirected_graph = graph.to_undirected()

    # Calculate centrality measures.
    betweenness = nx.betweenness_centrality(
        undirected_graph,
        normalized=True
    )

    closeness = nx.closeness_centrality(
        undirected_graph
    )

    # Calculate connected component sizes.
    component_sizes = {}

    for component in nx.connected_components(undirected_graph):
        size = len(component)

        for wallet in component:
            component_sizes[wallet] = size

    results = []

    for wallet in graph.nodes():

        results.append({
            "wallet": wallet,

            "degree": graph.degree(wallet),

            "in_degree": (
                graph.in_degree(wallet)
                if graph.is_directed()
                else 0
            ),

            "out_degree": (
                graph.out_degree(wallet)
                if graph.is_directed()
                else 0
            ),

            "betweenness_centrality": betweenness[wallet],

            "closeness_centrality": closeness[wallet],

            "cluster_size": component_sizes[wallet]
        })

    return pd.DataFrame(results)