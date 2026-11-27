import pandas as pd


def calculate_network_features(
    transactions: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate network features for Bitcoin wallets.

    Required columns:
        wallet_from, wallet_to

    Optional columns:
        source_ip, destination_ip
        source_asn, destination_asn
        source_country, destination_country
    """

    required = {"wallet_from", "wallet_to"}

    if not required.issubset(transactions.columns):
        missing = required - set(transactions.columns)
        raise ValueError(f"Missing columns: {missing}")

    df = transactions.copy()

    optional_columns = [
        "source_ip",
        "destination_ip",
        "source_asn",
        "destination_asn",
        "source_country",
        "destination_country"
    ]

    for column in optional_columns:
        if column not in df.columns:
            df[column] = None

    # Associate source network information with sender wallets.
    source = pd.DataFrame({
        "wallet": df["wallet_from"],
        "ip": df["source_ip"],
        "asn": df["source_asn"],
        "country": df["source_country"]
    })

    # Associate destination network information with receiver wallets.
    destination = pd.DataFrame({
        "wallet": df["wallet_to"],
        "ip": df["destination_ip"],
        "asn": df["destination_asn"],
        "country": df["destination_country"]
    })

    associations = pd.concat(
        [source, destination],
        ignore_index=True
    )

    associations = associations.dropna(subset=["wallet"])

    # Count how many distinct wallets use each IP address.
    ip_usage = (
        associations.dropna(subset=["ip"])
        .groupby("ip")["wallet"]
        .nunique()
    )

    results = []

    for wallet, group in associations.groupby("wallet"):

        # Unique IP addresses associated with this wallet.
        wallet_ips = group["ip"].dropna().unique()

        # Number of IPs shared with other wallets.
        shared_ip_count = sum(
            ip_usage.get(ip, 0) > 1
            for ip in wallet_ips
        )

        # Maximum number of distinct wallets sharing one IP.
        max_ip_wallet_reuse = max(
            [ip_usage.get(ip, 0) for ip in wallet_ips],
            default=0
        )

        # Number of distinct ASNs.
        asn_diversity = group["asn"].nunique()

        # Number of distinct countries.
        country_diversity = group["country"].nunique()

        results.append({
            "wallet": wallet,
            "unique_ip_count": len(wallet_ips),
            "shared_ip_count": int(shared_ip_count),
            "max_ip_wallet_reuse": int(max_ip_wallet_reuse),
            "asn_diversity": int(asn_diversity),
            "country_diversity": int(country_diversity),
            "geo_inconsistency": max(
                int(country_diversity) - 1, 0
            )
        })

    return pd.DataFrame(results)