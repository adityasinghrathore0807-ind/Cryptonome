import pandas as pd


def parse_csv(path) -> pd.DataFrame:
    """Read CSV/TSV with automatic delimiter detection."""
    return pd.read_csv(
        path,
        dtype=str,
        sep=None,
        engine="python",
        encoding="utf-8-sig"
    )
