"""
generate_data_manifest.py  --  FloodSentinel AI (Phase 2 Deliverable)
====================================================================
Generates data_manifest.csv with md5 checksums, file sizes, sample counts,
licenses, and provenance for all data sources in the project.
"""

import os
import hashlib
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ML_DIR = PROJECT_ROOT / "flood_project" / "ml_training"
DATA_DIR = ML_DIR / "data"

def compute_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def count_records(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path, nrows=5)
            # count lines quickly
            with open(path, "rb") as f:
                lines = sum(1 for _ in f) - 1
            return f"{lines} rows"
        except Exception:
            return "N/A"
    elif path.suffix.lower() == ".npy":
        try:
            import numpy as np
            arr = np.load(path, mmap_mode="r")
            return f"shape: {arr.shape}"
        except Exception:
            return "N/A"
    return "N/A"

def generate():
    records = []

    # 1. INDOFLOODS raw
    indofloods_dir = DATA_DIR / "raw" / "indofloods"
    if indofloods_dir.exists():
        for p in indofloods_dir.glob("*.*"):
            records.append({
                "dataset": "INDOFLOODS",
                "phase": "Phase 2 (Hydrology)",
                "type": "Raw Hydrology Time-Series & Catchment",
                "relative_path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "file_name": p.name,
                "size_bytes": p.stat().st_size,
                "md5": compute_md5(p),
                "records_or_shape": count_records(p),
                "license": "CC BY 4.0",
                "doi_or_url": "https://doi.org/10.5281/zenodo.14584655",
            })

    # 2. IFI-Impacts raw
    ifi_dir = DATA_DIR / "raw" / "ifi_impacts"
    if ifi_dir.exists():
        for p in ifi_dir.glob("*.*"):
            records.append({
                "dataset": "IFI-Impacts DFSI",
                "phase": "Phase 2 (Hydrology)",
                "type": "Disaster Frequency Severity Index",
                "relative_path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "file_name": p.name,
                "size_bytes": p.stat().st_size,
                "md5": compute_md5(p),
                "records_or_shape": count_records(p),
                "license": "CC BY 4.0",
                "doi_or_url": "https://doi.org/10.5281/zenodo.16994648",
            })

    # 3. Processed v3
    v3_dir = DATA_DIR / "processed_v3"
    if v3_dir.exists():
        for p in v3_dir.glob("*.*"):
            records.append({
                "dataset": "INDOFLOODS+IFI (Processed v3)",
                "phase": "Phase 1-3 Clean Pipeline",
                "type": "Normalized 11-Feature Arrays / Metadata",
                "relative_path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "file_name": p.name,
                "size_bytes": p.stat().st_size,
                "md5": compute_md5(p),
                "records_or_shape": count_records(p),
                "license": "CC BY 4.0 Derivative",
                "doi_or_url": "Zenodo 14584655 + Zenodo 16994648",
            })

    # 4. Open-Meteo ERA5-Land (Historical Archive)
    records.append({
        "dataset": "ERA5-Land (Open-Meteo Archive API)",
        "phase": "Phase 2 (Hydrology Expansion)",
        "type": "Hourly/Daily Weather Reanalysis (Precip, Soil Moisture, ET)",
        "relative_path": "flood_project/ml_training/scripts/download_era5_openmeteo.py",
        "file_name": "API Service / download_era5_openmeteo.py",
        "size_bytes": (ML_DIR / "scripts" / "download_era5_openmeteo.py").stat().st_size,
        "md5": compute_md5(ML_DIR / "scripts" / "download_era5_openmeteo.py"),
        "records_or_shape": "All 155 INDOFLOODS gauge lat/lons",
        "license": "CC BY 4.0 / Copernicus C3S",
        "doi_or_url": "https://archive-api.open-meteo.com/v1/archive",
    })

    # 5. GloFAS River Discharge
    records.append({
        "dataset": "GloFAS Reanalysis (Open-Meteo Flood API)",
        "phase": "Phase 2 (Hydrology Expansion)",
        "type": "Historical River Discharge Reanalysis",
        "relative_path": "flood_project/ml_training/scripts/download_glofas_openmeteo.py",
        "file_name": "API Service / download_glofas_openmeteo.py",
        "size_bytes": (ML_DIR / "scripts" / "download_glofas_openmeteo.py").stat().st_size,
        "md5": compute_md5(ML_DIR / "scripts" / "download_glofas_openmeteo.py"),
        "records_or_shape": "River discharge (m3/s) for gauge points",
        "license": "CC BY 4.0 / Copernicus EWDS",
        "doi_or_url": "https://flood-api.open-meteo.com/v1/flood",
    })

    # 6. Sen1Floods11
    records.append({
        "dataset": "Sen1Floods11",
        "phase": "Phase 2 (Satellite Imagery)",
        "type": "Sentinel-1 SAR VV/VH + Sentinel-2 Flood Masks",
        "relative_path": "flood_project/ml_training/scripts/download_sen1floods11.py",
        "file_name": "GCS Bucket gs://sen1floods11 / download_sen1floods11.py",
        "size_bytes": (ML_DIR / "scripts" / "download_sen1floods11.py").stat().st_size,
        "md5": compute_md5(ML_DIR / "scripts" / "download_sen1floods11.py"),
        "records_or_shape": "446 hand-labeled chips, ~4,831 weakly-labeled chips",
        "license": "CC BY 4.0",
        "doi_or_url": "https://github.com/cloudtostreet/Sen1Floods11 (Bonafilia et al. 2020)",
    })

    # 7. SpaceNet 8
    records.append({
        "dataset": "SpaceNet 8 (Flood & Roads)",
        "phase": "Phase 2 + 6 (Routing Validation)",
        "type": "Pre/Post-flood SAR/Optical with Road Speed & Geometry",
        "relative_path": "AWS Open Data s3://spacenet-dataset/spacenet/SN8_floods/",
        "file_name": "SN8_floods AWS S3 Bucket",
        "size_bytes": 0,
        "md5": "AWS-Open-Data-Managed",
        "records_or_shape": "Flooded Road Polygons & Shelter graphs",
        "license": "CC BY-SA 4.0",
        "doi_or_url": "https://spacenet.ai/sn8-challenge/",
    })

    df = pd.DataFrame(records)
    out_csv = PROJECT_ROOT / "data_manifest.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] data_manifest.csv created with {len(df)} records at {out_csv}")

if __name__ == "__main__":
    generate()
