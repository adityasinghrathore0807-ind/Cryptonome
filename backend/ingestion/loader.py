
"""Detects the file format, routes to the right parser and returns one validated unified table."""

from pathlib import Path
from typing import Iterable, List, Union

import pandas as pd

from ingestion.csv_parser import parse_csv
from ingestion.json_parser import parse_json
from ingestion.schema import (
    NETWORK_COLUMNS,
    SchemaError,
    is_transaction_table,
    normalize_columns,
    validate_schema,
)
from ingestion.xml_parser import parse_xml
from utils.logger import get_logger

log = get_logger("ingestion")

PARSERS = {
    ".csv": parse_csv,
    ".tsv": parse_csv,
    ".json": parse_json,
    ".jsonl": parse_json,
    ".ndjson": parse_json,
    ".xml": parse_xml,
}


def load_file(path) -> pd.DataFrame:
    path = Path(path)
    parser = PARSERS.get(path.suffix.lower())
    if parser is None:
        raise SchemaError(f"Unsupported file type: {path.suffix} ({path.name})")
    df = normalize_columns(parser(path))
    log.info("Loaded %s: %d rows, columns=%s", path.name, len(df), list(df.columns))
    return df


def merge_network_logs(tx: pd.DataFrame, net: pd.DataFrame) -> pd.DataFrame:
    """Attach IP / port / connection-time observations to transactions via txid."""
    if "txid" not in net.columns or "txid" not in tx.columns:
        log.warning("Network log has no txid column; cannot be linked to transactions - skipped.")
        return tx
    cols = ["txid"] + [c for c in NETWORK_COLUMNS if c in net.columns]
    net = net[cols].drop_duplicates("txid")
    merged = tx.merge(net, on="txid", how="left", suffixes=("", "__net"))
    for col in cols[1:]:
        if f"{col}__net" in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[f"{col}__net"])
            merged = merged.drop(columns=[f"{col}__net"])
    return merged


def load_files(paths: Iterable[Union[str, Path]]) -> pd.DataFrame:
    """Load several files. Transaction tables are concatenated, pure network logs are joined on txid."""
    tx_frames: List[pd.DataFrame] = []
    net_frames: List[pd.DataFrame] = []
    for p in paths:
        df = load_file(p)
        if df.empty:
            log.warning("%s is empty - skipped", p)
        elif is_transaction_table(df):
            tx_frames.append(df)
        elif "txid" in df.columns and any(c in df.columns for c in NETWORK_COLUMNS):
            net_frames.append(df)
        else:
            log.warning("%s does not look like a transaction or network log - skipped", p)
    if not tx_frames:
        raise SchemaError("No transaction records found in the supplied files.")
    tx = pd.concat(tx_frames, ignore_index=True, sort=False)
    if net_frames:
        tx = merge_network_logs(tx, pd.concat(net_frames, ignore_index=True, sort=False))
    validate_schema(tx)
    return tx


def load_directory(directory) -> pd.DataFrame:
    directory = Path(directory)
    files = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in PARSERS
    )
    if not files:
        raise SchemaError(f"No supported input files (CSV/JSON/XML) in {directory}")
    return load_files(files)
