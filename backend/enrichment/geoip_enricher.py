
"""Offline IP enrichment: IP -> country, city, ASN, organisation (+ hosting/VPN/Tor flag).

Preferred backend : MaxMind GeoLite2 .mmdb files in data/geoip/ (needs geoip2).
Fallback backend  : data/geoip/ip_ranges.csv
No network access is ever used.
"""
import ipaddress
from pathlib import Path

import pandas as pd

import config
from utils.logger import get_logger

log = get_logger("geoip")

FIELDS = ["country", "city", "asn", "org", "is_hosting"]
UNKNOWN = {
    "country": "??",
    "city": "",
    "asn": 0,
    "org": "",
    "is_hosting": False
}


def is_hosting_org(org: str) -> bool:
    org = (org or "").lower()
    return any(k in org for k in config.HOSTING_KEYWORDS)


class GeoIPEnricher:
    def __init__(self, geoip_dir=config.GEOIP_DIR):
        geoip_dir = Path(geoip_dir)
        self._cache = {}
        self._city = self._asn = None
        self._networks = []
        self.backend = "none"
        self._open_mmdb(geoip_dir)
        if self.backend == "none":
            self._load_ranges(geoip_dir / config.GEOIP_FALLBACK_CSV)
        if self.backend == "none":
            log.warning(
                "No GeoIP data found in %s - all IPs will be 'unknown'.",
                geoip_dir
            )

    def _open_mmdb(self, geoip_dir: Path):
        city_p = geoip_dir / config.GEOIP_CITY_DB
        asn_p = geoip_dir / config.GEOIP_ASN_DB
        if not (city_p.exists() or asn_p.exists()):
            return
        try:
            import geoip2.database
            if city_p.exists():
                self._city = geoip2.database.Reader(str(city_p))
            if asn_p.exists():
                self._asn = geoip2.database.Reader(str(asn_p))
            self.backend = "mmdb"
        except ImportError:
            log.warning("geoip2 not installed - cannot read .mmdb files.")

    def _load_ranges(self, csv_path: Path):
        if not csv_path.exists():
            return
        table = pd.read_csv(csv_path, dtype={"asn": "Int64"})
        for row in table.itertuples(index=False):
            net = ipaddress.ip_network(row.cidr, strict=False)
            self._networks.append((net, {
                "country": row.country,
                "city": row.city if isinstance(row.city, str) else "",
                "asn": int(row.asn) if pd.notna(row.asn) else 0,
                "org": row.org if isinstance(row.org, str) else ""
            }))
        self._networks.sort(
            key=lambda x: x[0].prefixlen,
            reverse=True
        )
        self.backend = "csv"

    def lookup(self, ip: str) -> dict:
        if ip in self._cache:
            return self._cache[ip]
        rec = dict(UNKNOWN)
        try:
            addr = ipaddress.ip_address(ip)
            if self.backend == "mmdb":
                rec.update(self._lookup_mmdb(ip))
            elif self.backend == "csv":
                for net, info in self._networks:
                    if addr.version == net.version and addr in net:
                        rec.update(info)
                        break
        except ValueError:
            pass
        rec["is_hosting"] = is_hosting_org(rec["org"])
        self._cache[ip] = rec
        return rec

    def _lookup_mmdb(self, ip: str) -> dict:
        info = {}
        if self._city is not None:
            try:
                r = self._city.city(ip)
                info["country"] = r.country.iso_code or "??"
                info["city"] = r.city.name or ""
            except Exception:
                pass
        if self._asn is not None:
            try:
                r = self._asn.asn(ip)
                info["asn"] = int(r.autonomous_system_number or 0)
                info["org"] = r.autonomous_system_organization or ""
            except Exception:
                pass
        return info

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for side in ("src", "dst"):
            col = f"{side}_ip"
            unique_ips = out[col].dropna().unique()
            table = pd.DataFrame(
                [self.lookup(ip) for ip in unique_ips],
                index=list(unique_ips),
                columns=FIELDS
            )
            for field in FIELDS:
                mapped = (
                    out[col].map(table[field])
                    if len(table)
                    else pd.Series(pd.NA, index=out.index)
                )
                out[f"{side}_{field}"] = mapped

            out[f"{side}_country"] = (
                out[f"{side}_country"].fillna("??").astype(str)
            )
            out[f"{side}_city"] = (
                out[f"{side}_city"].fillna("").astype(str)
            )
            out[f"{side}_org"] = (
                out[f"{side}_org"].fillna("").astype(str)
            )
            out[f"{side}_asn"] = (
                pd.to_numeric(out[f"{side}_asn"], errors="coerce")
                .fillna(0).astype("int64")
            )
            out[f"{side}_is_hosting"] = (
                out[f"{side}_is_hosting"].fillna(False).astype(bool)
            )

        log.info(
            "GeoIP enrichment done (backend=%s, %d unique IPs)",
            self.backend,
            len(self._cache)
        )
        return out
