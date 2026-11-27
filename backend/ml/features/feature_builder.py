import pandas as pd
import networkx as nx

from backend.ml.features.behavioral import (
    calculate_behavioral_features
)

from backend.ml.features.temporal import (
    calculate_temporal_features
)

from backend.ml.features.network import (
    calculate_network_features
)

from backend.ml.features.graph_features import (
    calculate_graph_features
)


def build_feature_matrix(
    transactions: pd.DataFrame,
    graph: nx.Graph
) -> pd.DataFrame:
    """
    Combine behavioral, temporal, network and graph
    features into one unified wallet-level feature matrix.
    """

    if transactions.empty:
        raise ValueError("Transaction dataset cannot be empty.")

    # Generate individual feature sets.
    behavioral = calculate_behavioral_features(transactions)

    temporal = calculate_temporal_features(transactions)

    network = calculate_network_features(transactions)

    graph_features = calculate_graph_features(graph)

    # Merge all features using wallet as the common key.
    feature_matrix = behavioral.merge(
        temporal,
        on="wallet",
        how="outer"
    )

    feature_matrix = feature_matrix.merge(
        network,
        on="wallet",
        how="outer"
    )

    feature_matrix = feature_matrix.merge(
        graph_features,
        on="wallet",
        how="outer"
    )

    # Replace infinite values and missing numerical values.
    numeric_columns = feature_matrix.select_dtypes(
        include="number"
    ).columns

    feature_matrix[numeric_columns] = (
        feature_matrix[numeric_columns]
        .replace([float("inf"), float("-inf")], float("nan"))
        .fillna(0)
    )

    # Remove invalid wallet records.
    feature_matrix = feature_matrix.dropna(
        subset=["wallet"]
    )

    # Ensure one row per wallet.
    if feature_matrix["wallet"].duplicated().any():
        raise ValueError("Duplicate wallet records detected.")

    # Return a clean, sorted feature matrix.
    return (
        feature_matrix
        .sort_values("wallet")
        .reset_index(drop=True)
    )