
"""Central configuration for the Cryptonome backend (offline, Linux-friendly)."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GEOIP_DIR = DATA_DIR / "geoip"
OUTPUT_DIR = DATA_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models" / "saved"

# ---- GeoIP (offline) ----
GEOIP_CITY_DB = "GeoLite2-City.mmdb"
GEOIP_ASN_DB = "GeoLite2-ASN.mmdb"
GEOIP_FALLBACK_CSV = "ip_ranges.csv"

HOSTING_KEYWORDS = (
    "hosting", "vpn", "proxy", "cloud", "datacenter", "data center", "vps",
    "exit relay", "tor exit", "digitalocean", "ovh", "hetzner", "linode",
    "amazon", "google cloud", "microsoft azure",
)

STANDARD_BTC_PORTS = (8333,)

# ---- Graph ----
MAX_EXPORT_NODES = 5000
EGO_HOPS = 2
BETWEENNESS_SAMPLE = 100

# ---- ML ----
RANDOM_STATE = 42
IF_N_ESTIMATORS = 200
IF_CONTAMINATION = "auto"
AE_EPOCHS = 150
AE_BATCH_SIZE = 64
AE_LEARNING_RATE = 1e-3

MODEL_WEIGHTS = {
    "isolation_forest": 0.5,
    "autoencoder": 0.5,
}

AE_TRAIN_MAX_IF_SCORE = 0.90

# ---- Scoring / alerts ----
RISK_WEIGHTS = {
    "model": 0.50,
    "behavioral": 0.15,
    "temporal": 0.10,
    "network": 0.15,
    "graph": 0.10,
}

RISK_GAMMA = 2.0
GROUP_ELEVATED_THRESHOLD = 0.80

SEVERITY_THRESHOLDS = (
    ("critical", 0.70),
    ("high", 0.50),
    ("medium", 0.35),
)

ALERT_MIN_RISK = 0.35
TOP_N_ALERTS = 50
TOP_K_REASONS = 5

# ---- API ----
CORS_ORIGINS = [
    "http://localhost:8501",
    "http://localhost:3000",
    "http://127.0.0.1:8501",
]

MAX_UPLOAD_MB = 50

ALLOWED_UPLOAD_EXTENSIONS = (
    ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xml"
)
