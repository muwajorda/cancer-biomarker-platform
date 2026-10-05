# Lightweight data ingestion utilities for the biomarker platform

import os
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
BRONZE_DIR.mkdir(parents=True, exist_ok=True)
SILVER_DIR.mkdir(parents=True, exist_ok=True)


def download_csv(url: str, dest_dir: Path = BRONZE_DIR) -> Path:
    """Download a CSV/TSV from a URL and save as a local file in the bronze directory.

    Returns the Path to the saved file.
    """
    parsed = urlparse(url)
    filename = Path(parsed.path).name or "ingest.csv"
    dest_path = dest_dir / filename

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return dest_path


def ingest_csv_to_parquet(csv_path: str | Path, dest_dir: Path = BRONZE_DIR) -> Path:
    """Load CSV into a dataframe, apply light normalization, and write to parquet in the bronze directory."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Basic normalization: lowercase column names and strip whitespace
    df.columns = [c.strip() for c in df.columns]

    # Ensure required columns exist for the MVP; otherwise return raw
    expected = {"gene_symbol", "cancer_type", "tumor_tpm", "normal_tpm"}
    if not expected.issubset(set(map(str.lower, df.columns))):
        # Try to guess common variants
        lower_map = {c.lower(): c for c in df.columns}
        rename_map = {}
        for e in expected:
            if e in lower_map:
                rename_map[lower_map[e]] = e
        if rename_map:
            df = df.rename(columns=rename_map)

    parquet_name = csv_path.with_suffix(".parquet").name
    parquet_path = dest_dir / parquet_name
    df.to_parquet(parquet_path, index=False)
    return parquet_path


def curate_parquet_to_silver(bronze_parquet: str | Path, dest_dir: Path = SILVER_DIR) -> Path:
    """Simple curation step: read bronze parquet, coerce types, compute tumor_normal_ratio, and write to silver."""
    p = Path(bronze_parquet)
    df = pd.read_parquet(p)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Safe column mapping
    col_map = {
        col: col
        for col in df.columns
    }

    # Ensure numeric columns
    for col in ["tumor_tpm", "normal_tpm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Compute ratio
    if "tumor_tpm" in df.columns and "normal_tpm" in df.columns:
        df["tumor_normal_ratio"] = df["tumor_tpm"] / (df["normal_tpm"] + 1e-6)
    else:
        df["tumor_normal_ratio"] = 0.0

    out_path = dest_dir / p.name
    df.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    # Simple CLI for manual ingestion
    import argparse

    parser = argparse.ArgumentParser(description="Ingest CSV by URL into bronze+silver layers")
    parser.add_argument("url", help="URL to CSV/TSV file to ingest")
    args = parser.parse_args()

    print("Downloading...", args.url)
    csv_path = download_csv(args.url)
    print("Saved to", csv_path)
    p = ingest_csv_to_parquet(csv_path)
    print("Bronze parquet:", p)
    s = curate_parquet_to_silver(p)
    print("Silver parquet:", s)
