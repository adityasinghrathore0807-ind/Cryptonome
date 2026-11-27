
"""Tiny hand-made dataset with known answers."""
import pandas as pd

from enrichment.geoip_enricher import GeoIPEnricher
from preprocessing.cleaner import clean_transactions
from preprocessing.normalizer import normalize

ROWS = [
    ("a1", "2026-03-01T10:00:00Z", "W1", "W2", "1.0", "10.1.0.5", "10.2.0.9", 40000, 8333),
    ("a2", "2026-03-01T10:05:00Z", "W1", "W3", "0.5", "10.1.0.5", "10.2.0.9", 40001, 8333),
    ("a3", "2026-03-01T10:10:00Z", "W1", "W4", "0.5", "10.4.0.7", "10.2.0.9", 40002, 9050),
    ("a4", "2026-03-02T09:00:00Z", "W2", "W5", "0.9", "10.2.0.1", "10.3.0.2", 40003, 8333),
    ("a5", "2026-03-02T09:30:00Z", "W5", "W6", "0.89", "10.3.0.3", "10.3.0.2", 40004, 8333),
]

COLUMNS = [
    "txid", "timestamp", "sender", "receiver", "amount_btc",
    "src_ip", "dst_ip", "src_port", "dst_port"
]

RANGES_CSV = """cidr,country,city,asn,org
10.1.0.0/16,IN,Delhi,64512,SynthNet Residential IN
10.2.0.0/16,US,Seattle,64513,SynthNet Residential US
10.3.0.0/16,GB,London,64514,SynthNet Residential GB
10.4.0.0/16,DE,Frankfurt,64601,SynthHost Hosting DE
"""


def raw_frame() -> pd.DataFrame:
    return pd.DataFrame(ROWS, columns=COLUMNS)


def make_sample_df(tmp_dir) -> pd.DataFrame:
    """Cleaned + GeoIP-enriched frame."""
    (tmp_dir / "ip_ranges.csv").write_text(RANGES_CSV)
    df = clean_transactions(normalize(raw_frame().astype(str)))
    return GeoIPEnricher(tmp_dir).enrich(df)
