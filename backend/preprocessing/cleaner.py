
"""Row-level cleaning: invalid rows, duplicates, bad IPs / ports."""
import hashlib
import ipaddress

import pandas as pd

from utils.logger import get_logger

log = get_logger("cleaner")


def _clean_str(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s or None


def _valid_ip(v):
    s = _clean_str(v)
    if s is None:
        return None
    try:
        return ipaddress.ip_address(s).compressed
    except ValueError:
        return None


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    report = {"input_rows": len(out)}

    for col in ("txid", "sender", "receiver"):
        out[col] = out[col].map(_clean_str).astype("object")
    out["txid"] = out["txid"].map(
        lambda v: v.lower() if isinstance(v, str) else v
    )

    n = len(out)
    out = out.dropna(
        subset=["sender", "receiver", "timestamp", "amount_btc"]
    )
    report["dropped_missing_fields"] = n - len(out)

    n = len(out)
    out = out[out["amount_btc"] > 0]
    report["dropped_non_positive_amount"] = n - len(out)

    n = len(out)
    out = out[out["sender"] != out["receiver"]]
    report["dropped_self_transfers"] = n - len(out)

    # Deterministic txid for records that have none
    missing = out["txid"].isna()
    if missing.any():
        sub = out.loc[missing]
        out.loc[missing, "txid"] = [
            hashlib.sha1(
                f"{s}|{r}|{t}|{a}".encode()
            ).hexdigest()[:32]
            for s, r, t, a in zip(
                sub["sender"],
                sub["receiver"],
                sub["timestamp"],
                sub["amount_btc"]
            )
        ]

    n = len(out)
    out = out.drop_duplicates(
        subset=["txid", "sender", "receiver"],
        keep="first"
    )
    report["dropped_duplicates"] = n - len(out)

    for col in ("src_ip", "dst_ip"):
        cleaned = out[col].map(_valid_ip).astype("object")
        report[f"invalid_{col}"] = int(
            (out[col].notna() & cleaned.isna()).sum()
        )
        out[col] = cleaned

    for col in ("src_port", "dst_port"):
        port = pd.to_numeric(out[col], errors="coerce").round()
        port = port.where((port >= 1) & (port <= 65535))
        out[col] = port.astype("Int64")

    out = out.reset_index(drop=True)
    report["output_rows"] = len(out)
    out.attrs["clean_report"] = report

    log.info("Cleaning report: %s", report)
    return out
