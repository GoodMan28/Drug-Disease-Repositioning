from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"      # caches + human-readable mapping tables
PROCESSED = ROOT / "data" / "processed"  # model-ready .npz files
RESULTS = ROOT / "results"

for p in (INTERIM, PROCESSED, RESULTS):
    p.mkdir(parents=True, exist_ok=True)

DATASETS = {"F": "Fdataset", "C": "Cdataset"}


def dataset_name(key: str) -> str:
    """Accept 'F', 'Fdataset', 'f' ... and return the canonical name."""
    k = key.strip()[0].upper()
    if k not in DATASETS:
        raise ValueError(f"unknown dataset {key!r}; use F or C")
    return DATASETS[k]
