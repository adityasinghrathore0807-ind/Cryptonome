
"""Unified schema for Cryptonome transaction datasets."""

import re
from typing import List

import pandas as pd

CANONICAL_COLUMNS = [
    "txid", "timestamp", "sender", "receiver", "amount_btc", "amount_satoshi",
    "src_ip", "dst_ip", "src_port", "dst_port", "conn_timestamp",
]

REQUIRED_TX_COLUMNS = ["txid", "timestamp", "sender", "receiver"]

NETWORK_COLUMNS = ["src_ip", "dst_ip", "src_port", "dst_port", "conn_timestamp"]

ALIASES = {
    "txid": ["txid", "tx_id", "tx_hash", "transaction_id", "transaction_hash", "hash", "tx", "id"],
    "timestamp": ["timestamp", "time", "tx_time", "tx_timestamp", "block_time", "datetime", "date"],
    "sender": ["sender", "from", "from_wallet", "from_address", "source_wallet", "source_address",
               "input_address", "input_wallet", "wallet_from"],
    "receiver": ["receiver", "to", "to_wallet", "to_address", "destination_wallet", "destination_address",
                "output_address", "output_wallet", "wallet_to", "recipient"],
    "amount_btc": ["amount_btc", "amount", "value", "value_btc", "btc", "btc_amount", "output_amount"],
    "amount_satoshi": ["amount_satoshi", "satoshi", "satoshis", "value_satoshi", "sats"],
    "src_ip": ["src_ip", "source_ip", "ip", "ip_address", "origin_ip", "client_ip", "src_ip_address"],
    "dst_ip": ["dst_ip", "destination_ip", "dest_ip", "peer_ip", "remote_ip"],
    "src_port": ["src_port", "source_port", "sport"],
    "dst_port": ["dst_port", "destination_port", "dest_port", "dport", "port"],
    "conn_timestamp": ["conn_timestamp", "connection_time", "connection_timestamp", "conn_time",
                       "seen_at", "first_seen"],
}

_REVERSE = {
    alias: canon
    for canon, aliases in ALIASES.items()
    for alias in aliases
}


class SchemaError(ValueError):
    """Raised when input data cannot be mapped onto the unified schema."""


def _clean_name(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")


def _resolve(col: str):
    """Map raw column names to canonical names."""
    clean = _clean_name(col)
    parts = clean.split("_")

    for i in range(len(parts)):
        candidate = "_".join(parts[i:])
        if candidate in _REVERSE:
            return _REVERSE[candidate]

    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename recognised columns to canonical names; retain unknown columns."""
    rename, used = {}, set()

    for col in df.columns:
        canon = _resolve(col)
        if canon and canon not in used:
            rename[col] = canon
            used.add(canon)

    out = df.rename(columns=rename)

    keep = [c for c in out.columns if c in used or _resolve(c) is None]

    return out.loc[:, ~out.columns.duplicated()][
        [c for c in out.columns if c in keep]
    ]


def is_transaction_table(df: pd.DataFrame) -> bool:
    return {"sender", "receiver"}.issubset(df.columns)


def validate_schema(df: pd.DataFrame) -> None:
    missing: List[str] = [
        c for c in REQUIRED_TX_COLUMNS if c not in df.columns
    ]

    if "amount_btc" not in df.columns and "amount_satoshi" not in df.columns:
        missing.append("amount_btc (or amount_satoshi)")

    if missing:
        raise SchemaError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Found: {', '.join(map(str, df.columns))}"
        )
