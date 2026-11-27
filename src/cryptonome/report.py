"""Cryptonome investigation report generation."""

import csv
import json
import math
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path


STATIC_DIR = Path(__file__).parent / "dashboard" / "static"


def to_jsonable(value):
    """Convert common Python and scientific values into JSON-safe values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]

    if isinstance(value, set):
        return [to_jsonable(v) for v in sorted(value, key=str)]

    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())

    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except (ValueError, TypeError):
            pass

    return str(value)


def script_safe_json(value):
    """Serialize JSON safely for embedding inside an HTML script element."""
    result = json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return (
        result.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _read_static(name):
    """Read a file from the dashboard static directory."""
    path = STATIC_DIR / name
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def vendor_d3():
    """Return the locally installed D3.js source, if available."""
    return _read_static("vendor/d3.min.js")


def _graph_to_payload(graph):
    """Convert a NetworkX-style graph into the Backend 1 JSON format."""
    nodes = []

    for node_id, attrs in graph.nodes(data=True):
        item = dict(attrs)
        item["id"] = str(node_id)
        item["type"] = attrs.get("type")
        item["label"] = attrs.get("label", str(node_id))
        nodes.append(to_jsonable(item))

    edges = []

    if graph.is_multigraph():
        graph_edges = graph.edges(keys=True, data=True)
        for source, target, key, attrs in graph_edges:
            item = dict(attrs)
            item["source"] = str(source)
            item["target"] = str(target)
            item["type"] = attrs.get("type", key)
            edges.append(to_jsonable(item))
    else:
        for source, target, attrs in graph.edges(data=True):
            item = dict(attrs)
            item["source"] = str(source)
            item["target"] = str(target)
            item["type"] = attrs.get(
                "type", attrs.get("relation", "")
            )
            edges.append(to_jsonable(item))

    return {
        "nodes": nodes,
        "edges": edges,
        "truncated": False,
    }


def build_payload(res):
    """Extract graph data from a pipeline result or existing graph payload."""
    if isinstance(res, Mapping):
        if "nodes" in res and "edges" in res:
            return to_jsonable({
                "nodes": res["nodes"],
                "edges": res["edges"],
                "truncated": res.get("truncated", False),
            })

        for key in ("graph", "G"):
            candidate = res.get(key)
            if candidate is not None and hasattr(candidate, "nodes"):
                return _graph_to_payload(candidate)

    if isinstance(res, (tuple, list)):
        for item in res:
            if (
                hasattr(item, "nodes")
                and callable(item.nodes)
                and hasattr(item, "edges")
            ):
                return _graph_to_payload(item)

    if (
        hasattr(res, "nodes")
        and callable(res.nodes)
        and hasattr(res, "edges")
    ):
        return _graph_to_payload(res)

    raise TypeError(
        "Could not find graph data. Supply a graph, a "
        "{'nodes': ..., 'edges': ...} payload, or a pipeline result."
    )


def write_json(res, path):
    """Write the investigation graph as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_payload(res)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return path


def _find_transactions(res):
    """Find the transaction DataFrame or records in a pipeline result."""
    if isinstance(res, Mapping):
        for key in ("transactions", "df", "dataframe"):
            if key in res:
                return res[key]

    if isinstance((res), (tuple, list)):
        for item in res:
            if callable(getattr(item, "to_csv", None)):
                return item

    if callable(getattr(res, "to_csv", None)):
        return res

    return None


def write_csv(res, path):
    """Write pipeline transactions as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    transactions = _find_transactions(res)

    if transactions is None:
        raise TypeError(
            "Could not find transaction data in the pipeline result."
        )

    if callable(getattr(transactions, "to_csv", None)):
        transactions.to_csv(path, index=False)
    else:
        records = list(transactions)
        if not records:
            path.write_text("", encoding="utf-8")
            return path

        fieldnames = sorted({
            str(key)
            for record in records
            if isinstance(record, Mapping)
            for key in record
        })

        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(to_jsonable(record))

    return path


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cryptonome Investigation Report</title>
<style>
body {
    margin: 0;
    padding: 24px;
    background: #0b1120;
    color: #e2e8f0;
    font-family: Arial, sans-serif;
}
h1 { margin-top: 0; }
#graph {
    width: 100%;
    min-height: __HEIGHT__px;
    background: #0f172a;
    border: 1px solid #263449;
    border-radius: 12px;
    overflow: hidden;
}
#details {
    margin-top: 18px;
    padding: 16px;
    background: #111827;
    border: 1px solid #263449;
    border-radius: 10px;
    overflow-wrap: anywhere;
}
.detail {
    padding: 8px 0;
    border-bottom: 1px solid #263449;
}
.detail-label {
    display: block;
    color: #94a3b8;
    font-size: 12px;
    margin-bottom: 4px;
}
.detail-value { font-size: 14px; }
#status { color: #94a3b8; margin: 12px 0; }
</style>
</head>
<body>
<h1>Cryptonome Investigation Report</h1>
<p>Interactive transaction and wallet graph.</p>
<div id="status">Loading graph...</div>
<div id="graph"></div>
<section id="details">Select a graph node to inspect it.</section>

<script type="application/json" id="graph-data">__GRAPH_DATA__</script>
<script>__GRAPH_JS__</script>
<script>__ADAPTER_JS__</script>
<script>
(function () {
    "use strict";

    const status = document.getElementById("status");
    const details = document.getElementById("details");

    function addDetail(label, value) {
        const row = document.createElement("div");
        row.className = "detail";

        const heading = document.createElement("span");
        heading.className = "detail-label";
        heading.textContent = label;

        const content = document.createElement("div");
        content.className = "detail-value";
        content.textContent =
            value === undefined || value === null || value === ""
                ? "Not available"
                : String(value);

        row.appendChild(heading);
        row.appendChild(content);
        details.appendChild(row);
    }

    function selectNode(node) {
        details.innerHTML = "";
        addDetail("Node type", node.kind);
        addDetail("Node ID", node.id);
        addDetail("Label", node.label);
        addDetail("Country", node.country);
        addDetail("City", node.city);
        addDetail("ASN", node.asn);
        addDetail("Organization", node.org);
        addDetail("Amount (BTC)", node.amount);
        addDetail("Timestamp", node.timestamp);
        status.textContent = "Selected " + node.kind + ": " + node.label;
    }

    try {
        const raw = JSON.parse(
            document.getElementById("graph-data").textContent
        );

        const payload = window.CryptonomeBackendAdapter.convertGraph(raw);

        window.CryptonomeGraph.render(
            document.getElementById("graph"),
            payload,
            {
                height: __HEIGHT__,
                onSelect: selectNode
            }
        );

        status.textContent =
            "Loaded " + payload.nodes.length + " nodes and " +
            payload.edges.length + " edges.";
    } catch (error) {
        status.textContent = "Could not load graph: " + error.message;
        console.error(error);
    }
})();
</script>
</body>
</html>
"""


def graph_page(payload, height=520):
    """Build a standalone interactive HTML graph page."""
    height = max(200, min(int(height), 2000))

    graph_js = _read_static("graph.js")
    adapter_js = _read_static("backend_adapter.js")

    if not graph_js or not adapter_js:
        raise FileNotFoundError(
            "graph.js or backend_adapter.js is missing from dashboard/static."
        )

    safe_graph_js = graph_js.replace("</script", "<\\/script")
    safe_adapter_js = adapter_js.replace("</script", "<\\/script")

    return (
        HTML_TEMPLATE
        .replace("__HEIGHT__", str(height))
        .replace("__GRAPH_DATA__", script_safe_json(payload))
        .replace("__GRAPH_JS__", safe_graph_js)
        .replace("__ADAPTER_JS__", safe_adapter_js)
    )


def write_html(res, path):
    """Write a standalone interactive HTML investigation report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_payload(res)
    page = graph_page(payload)

    path.write_text(page, encoding="utf-8")
    return path
