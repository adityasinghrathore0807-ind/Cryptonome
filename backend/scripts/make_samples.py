
"""Create sample transactions as CSV, JSON and XML for testing."""

import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "samples"
OUT.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "txid", "timestamp", "sender", "receiver", "amount_btc",
    "src_ip", "dst_ip", "src_port", "dst_port"
]

ROWS = [
    ("a1", "2026-03-01T10:00:00Z", "W1", "W2", "1.0", "10.1.0.5", "10.2.0.9", 40000, 8333),
    ("a2", "2026-03-01T10:05:00Z", "W1", "W3", "0.5", "10.1.0.5", "10.2.0.9", 40001, 8333),
    ("a3", "2026-03-01T10:10:00Z", "W1", "W4", "0.5", "10.4.0.7", "10.2.0.9", 40002, 9050),
    ("a4", "2026-03-02T09:00:00Z", "W2", "W5", "0.9", "10.2.0.1", "10.3.0.2", 40003, 8333),
    ("a5", "2026-03-02T09:30:00Z", "W5", "W6", "0.89", "10.3.0.3", "10.3.0.2", 40004, 8333),
]

df = pd.DataFrame(ROWS, columns=COLUMNS)

# CSV
df.to_csv(OUT / "sample.csv", index=False)

# JSON
(OUT / "sample.json").write_text(
    json.dumps({"transactions": df.to_dict("records")}, indent=2),
    encoding="utf-8"
)

# XML with different field names and nested network data
items = "".join(
    f"  <tx><txid>{r.txid}</txid><time>{r.timestamp}</time>"
    f"<from>{r.sender}</from><to>{r.receiver}</to><amount>{r.amount_btc}</amount>"
    f"<network><src_ip>{r.src_ip}</src_ip><dst_ip>{r.dst_ip}</dst_ip>"
    f"<src_port>{r.src_port}</src_port><dst_port>{r.dst_port}</dst_port>"
    f"</network></tx>\n"
    for r in df.itertuples()
)

(OUT / "sample.xml").write_text(
    f"<root>\n<transactions>\n{items}</transactions>\n</root>\n",
    encoding="utf-8"
)

print("Created:", ", ".join(sorted(p.name for p in OUT.iterdir())))
