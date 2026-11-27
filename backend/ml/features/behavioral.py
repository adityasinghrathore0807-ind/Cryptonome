import pandas as pd


def calculate_behavioral_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate behavioral features for Bitcoin wallets.

    Required columns:
    wallet_from, wallet_to, amount
    """

    required_columns = {"wallet_from", "wallet_to", "amount"}

    if not required_columns.issubset(transactions.columns):
        missing = required_columns - set(transactions.columns)
        raise ValueError(f"Missing columns: {missing}")

    df = transactions.copy()

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    df = df.dropna(
        subset=["wallet_from", "wallet_to", "amount"]
    )

    wallets = set(df["wallet_from"]) | set(df["wallet_to"])

    results = []

    for wallet in wallets:

        sent = df[df["wallet_from"] == wallet]
        received = df[df["wallet_to"] == wallet]

        fan_out = sent["wallet_to"].nunique()

        fan_in = received["wallet_from"].nunique()

        transaction_frequency = len(sent) + len(received)

        wallet_transactions = pd.concat(
            [sent, received]
        ).drop_duplicates()

        average_amount = wallet_transactions["amount"].mean()

        results.append({
            "wallet": wallet,
            "fan_in": fan_in,
            "fan_out": fan_out,
            "transaction_frequency": transaction_frequency,
            "average_amount": average_amount
        })

    return pd.DataFrame(results)