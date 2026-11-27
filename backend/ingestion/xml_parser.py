
from collections import Counter

import pandas as pd
from lxml import etree


def _local(elem) -> str:
    return etree.QName(elem).localname


def _record_elements(root):
    """Descend until we find a repeated child tag; those elements are the records."""
    node = root
    while True:
        kids = [k for k in node if isinstance(k.tag, str)]
        if not kids:
            return [node]
        tag, count = Counter(_local(k) for k in kids).most_common(1)[0]
        if count > 1:
            return [k for k in kids if _local(k) == tag]
        if len(kids) == 1:
            node = kids[0]
            continue
        return kids


def _flatten(elem, prefix: str = "") -> dict:
    row = {f"{prefix}{_local_attr(k)}": v for k, v in elem.attrib.items()}
    for child in elem:
        if not isinstance(child.tag, str):
            continue
        name = _local(child)
        if len(child) or child.attrib:
            row.update(_flatten(child, f"{prefix}{name}_"))
            if child.text and child.text.strip():
                row[f"{prefix}{name}"] = child.text.strip()
        else:
            row[f"{prefix}{name}"] = (child.text or "").strip() or None
    return row


def _local_attr(name: str) -> str:
    return name.split("}")[-1]


def parse_xml(path) -> pd.DataFrame:
    """Safe XML parsing (no entity resolution, no network access)."""
    parser = etree.XMLParser(resolve_entities=False, no_network=True, remove_comments=True, huge_tree=False)
    root = etree.parse(str(path), parser).getroot()
    rows = [_flatten(e) for e in _record_elements(root)]
    return pd.DataFrame(rows)
