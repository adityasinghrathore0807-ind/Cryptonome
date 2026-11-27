
import json
from pathlib import Path

import pandas as pd

_RECORD_KEYS = ("transactions", "records", "data", "items", "txs", "results", "network_logs", "logs")


def _find_records(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in _RECORD_KEYS:
            if isinstance(data.get(key), list):
                return data[key]
        list_values = [v for v in data.values() if isinstance(v, list)]
        if len(list_values) == 1:
            return list_values[0]
        return [data]
    raise ValueError("Unsupported JSON structure")


def parse_json(path) -> pd.DataFrame:
    """Supports a JSON array, an object wrapping an array, or JSON-lines. Nested keys are flattened with '_'."""
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    records = [r for r in _find_records(data) if isinstance(r, dict)]
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records, sep="_")
