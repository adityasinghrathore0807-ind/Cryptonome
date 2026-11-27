import pandas as pd
import numpy as np


def calculate_temporal_features(
    transactions: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate temporal behavioral features for Bitcoin wallets.

    Required columns:
    wallet_from, wallet_to, timestamp
    """

    required = {"wallet_from", "wallet_to", "timestamp"}

    if not required.issubset(transactions.columns):
        missing = required - set(transactions.columns)
        raise ValueError(f"Missing columns: {missing}")

    df = transactions.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True
    )

    df = df.dropna(
        subset=["wallet_from", "wallet_to", "timestamp"]
    )

    # Create one activity record for each participating wallet.
    sent = df[["wallet_from", "timestamp"]].rename(
        columns={"wallet_from": "wallet"}
    )

    received = df[["wallet_to", "timestamp"]].rename(
        columns={"wallet_to": "wallet"}
    )

    activities = pd.concat(
        [sent, received],
        ignore_index=True
    )

    results = []

    for wallet, group in activities.groupby("wallet"):

        timestamps = group["timestamp"].sort_values()

        # Calculate time between consecutive transactions.
        intervals = (
            timestamps.diff()
            .dt.total_seconds()
            .dropna()
        )

        interval_count = len(intervals)

        if interval_count == 0:
            mean_interval = 0.0
            std_interval = 0.0
            timing_irregularity = 0.0
            burstiness = 0.0

        else:
            mean_interval = float(intervals.mean())

            std_interval = float(intervals.std(ddof=0))

            # Coefficient of variation.
            timing_irregularity = (
                std_interval / mean_interval
                if mean_interval > 0
                else 0.0
            )

            # Burstiness coefficient.
            denominator = std_interval + mean_interval

            burstiness = (
                (std_interval - mean_interval) / denominator
                if denominator > 0
                else 0.0
            )

        results.append({
            "wallet": wallet,
            "interval_count": interval_count,
            "mean_interarrival_seconds": mean_interval,
            "std_interarrival_seconds": std_interval,
            "timing_irregularity": timing_irregularity,
            "burstiness": burstiness
        })

    return pd.DataFrame(results)