
import json

import pytest

from ingestion.loader import load_files
from ingestion.schema import SchemaError, normalize_columns
from preprocessing.cleaner import clean_transactions
from preprocessing.normalizer import normalize
from sample_data import raw_frame


def _write_all(tmp_path):
    df = raw_frame()
    csv_p = tmp_path / "t.csv"
    json_p = tmp_path / "t.json"
    xml_p = tmp_path / "t.xml"

    df.to_csv(csv_p, index=False)
    json_p.write_text(json.dumps({"transactions": df.to_dict("records")}))

    items = "".join(
        f"<tx><txid>{r.txid}</txid><time>{r.timestamp}</time><from>{r.sender}</from><to>{r.receiver}</to>"
        f"<amount>{r.amount_btc}</amount><network><src_ip>{r.src_ip}</src_ip><dst_ip>{r.dst_ip}</dst_ip>"
        f"<src_port>{r.src_port}</src_port><dst_port>{r.dst_port}</dst_port></network></tx>"
        for r in df.itertuples()
    )

    xml_p.write_text(f"<root><transactions>{items}</transactions></root>")
    return csv_p, json_p, xml_p


def test_all_formats_give_same_result(tmp_path):
    results = []

    for p in _write_all(tmp_path):
        df = clean_transactions(normalize(load_files([p])))
        results.append(
            df[["txid", "sender", "receiver", "amount_btc", "src_ip", "dst_port"]]
            .astype(str).values.tolist()
        )

    assert len(results[0]) == 5
    assert results[0] == results[1] == results[2]


def test_column_aliases():
    import pandas as pd

    df = normalize_columns(
        pd.DataFrame(columns=[
            "Tx Hash", "Block_Time", "From Address",
            "To Address", "Value", "IP"
        ])
    )

    assert {
        "txid", "timestamp", "sender", "receiver", "amount_btc", "src_ip"
    } <= set(df.columns)


def test_missing_columns_raise(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("foo,bar\n1,2\n")

    with pytest.raises(SchemaError):
        load_files([p])


def test_satoshi_and_epoch_and_cleaning(tmp_path):
    p = tmp_path / "s.csv"

    p.write_text(
        "txid,timestamp,sender,receiver,satoshi,ip\n"
        "T1,1772359200,A,B,100000000,10.1.0.1\n"
        "T1,1772359200,A,B,100000000,10.1.0.1\n"
        "T2,1772359200000,A,A,5,10.1.0.1\n"
        "T3,not-a-date,A,B,5,10.1.0.1\n"
        "T4,1772359300,A,C,200000000,999.1.1.1\n"
    )

    df = clean_transactions(normalize(load_files([p])))

    assert list(df["txid"]) == ["t1", "t4"]
    assert df["amount_btc"].tolist() == [1.0, 2.0]
    assert df["src_ip"].isna().tolist() == [False, True]
    assert df["timestamp"].dt.year.eq(2026).all()


def test_network_log_merged_on_txid(tmp_path):
    (tmp_path / "tx.csv").write_text(
        "txid,timestamp,sender,receiver,amount\n"
        "T1,2026-03-01T00:00:00Z,A,B,1\n"
    )

    (tmp_path / "net.json").write_text(
        json.dumps([
            {"tx_hash": "T1", "source_ip": "10.1.0.1", "port": 8333}
        ])
    )

    df = load_files([tmp_path / "tx.csv", tmp_path / "net.json"])

    assert df.loc[0, "src_ip"] == "10.1.0.1"
    assert str(df.loc[0, "dst_port"]) == "8333"
