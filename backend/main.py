
"""Cryptonome Backend 1 - Main pipeline."""

from pathlib import Path
import sys

from ingestion.loader import load_files
from preprocessing.normalizer import normalize
from preprocessing.cleaner import clean_transactions
from enrichment.geoip_enricher import GeoIPEnricher

from graph.builder import build_graph, wallet_projection
from graph.clustering import detect_communities
from graph.exporter import export_graph


def run_pipeline(paths):
    # 1. Load, normalize, clean and enrich transactions
    df = load_files(paths)
    df = normalize(df)
    df = clean_transactions(df)
    df = GeoIPEnricher().enrich(df)

    # 2. Build graph outputs
    G = build_graph(df)
    WG = wallet_projection(df)
    clusters = detect_communities(G)

    # 3. Save processed outputs
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "transactions_clean.csv", index=False)
    export_graph(G, output_dir / "graph.json")

    print("\n========== CRYPTONOME BACKEND 1 ==========")
    print(f"Clean transactions: {len(df)}")
    print(f"Unified graph nodes: {G.number_of_nodes()}")
    print(f"Unified graph edges: {G.number_of_edges()}")
    print(f"Wallet flow nodes: {WG.number_of_nodes()}")
    print(f"Wallet flow edges: {WG.number_of_edges()}")
    print(f"Wallets clustered: {len(clusters)}")
    print(f"Outputs saved in: {output_dir.resolve()}")
    print("==========================================")

    return df, G, WG, clusters


if __name__ == "__main__":
    input_paths = sys.argv[1:] or ["data/samples/sample.csv"]
    run_pipeline(input_paths)
