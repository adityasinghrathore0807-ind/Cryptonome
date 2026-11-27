import random
from datetime import datetime, timedelta

import pandas as pd


random.seed(42)

transactions = []

wallets = [f"wallet_{i}" for i in range(100)]

start_time = datetime(2026, 1, 1, 0, 0, 0)

# Generate normal wallet transactions
for i in range(450):
    sender = random.choice(wallets[:90])
    receiver = random.choice(wallets[:90])

    while receiver == sender:
        receiver = random.choice(wallets[:90])

    transactions.append({
        "txid": f"tx_{i}",
        "wallet_from": sender,
        "wallet_to": receiver,
        "amount": round(random.uniform(1, 500), 2),
        "timestamp": start_time + timedelta(
            minutes=random.randint(1, 100000)
        )
    })


# Generate deliberately unusual transactions
for i in range(450, 500):
    sender = random.choice(wallets[90:])
    receiver = random.choice(wallets[:90])

    transactions.append({
        "txid": f"tx_{i}",
        "wallet_from": sender,
        "wallet_to": receiver,
        "amount": round(random.uniform(50000, 200000), 2),
        "timestamp": start_time + timedelta(
            minutes=random.randint(1, 100000)
        )
    })


df = pd.DataFrame(transactions)

df = df.sort_values("timestamp").reset_index(drop=True)

df["timestamp"] = df["timestamp"].astype(str)

output_path = "backend/data/transactions.csv"

df.to_csv(output_path, index=False)

print("Dataset generated successfully!")
print("Total transactions:", len(df))
print("Total unique wallets:", len(
    set(df["wallet_from"]) | set(df["wallet_to"])
))
print("Saved to:", output_path)
