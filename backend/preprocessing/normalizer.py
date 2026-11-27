
"""Type normalisation: UTC timestamps, BTC amounts, guaranteed canonical columns."""
import numpy as np
import pandas as pd

from ingestion.schema import NETWORK_COLUMNS

SATOSHI = 1e8


def parse_datetime(s: pd.Series) -> pd.Series:
    """Accepts ISO strings, epoch seconds and epoch milliseconds. Returns UTC datetimes (NaT if invalid)."""
    s = s.astype("object")
    num = pd.to_numeric(s, errors="coerce")
    is_num = num.notna()
    is_ms = is_num & (num.abs() > 1e11)
    is_sec = is_num & ~is_ms
    is_text = ~is_num & s.notna()
    parts = []

    if is_sec.any():
        parts.append(pd.to_datetime(num[is_sec], unit="s", utc=True))

    if is_ms.any():
        parts.append(pd.to_datetime(num[is_ms], unit="ms", utc=True))

    if is_text.any():
        parts.append(pd.to_datetime(
            s[is_text].astype(str),
            utc=True,
            errors="coerce",
            format="mixed"
        ))

    if not parts:
        return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")

    return pd.concat(parts).reindex(s.index)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in NETWORK_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out["timestamp"] = parse_datetime(out["timestamp"])
    out["conn_timestamp"] = parse_datetime(out["conn_timestamp"])

    btc = pd.to_numeric(out["amount_btc"], errors="coerce") if "amount_btc" in out.columns \
        else pd.Series(np.nan, index=out.index, dtype="float64")

    if "amount_satoshi" in out.columns:
        sats = pd.to_numeric(out["amount_satoshi"], errors="coerce") / SATOSHI
        btc = btc.fillna(sats)
        out = out.drop(columns=["amount_satoshi"])

    out["amount_btc"] = btc.astype("float64")

    return out.sort_values("timestamp", kind="stable").reset_index(drop=True)
