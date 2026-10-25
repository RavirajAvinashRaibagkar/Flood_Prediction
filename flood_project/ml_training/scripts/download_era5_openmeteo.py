"""
download_era5_openmeteo.py  --  FloodSentinel AI Phase 2
=========================================================
Downloads ERA5-Land reanalysis (hourly rainfall, soil moisture, runoff)
via the Open-Meteo Archive API for each gauge location in the INDOFLOODS
metadata. Free, no API key required.

License: Open-Meteo data is CC BY 4.0
Docs: https://open-meteo.com/en/docs/historical-weather-api

Usage:
  python download_era5_openmeteo.py [--start 1970 --end 2020] [--outdir data/raw/era5]

Output:
  data/raw/era5/<GaugeID>.csv  — one CSV per gauge with daily values
  data/raw/era5/manifest.csv   — list of gauges, date ranges, variable counts
  data/raw/era5/CHECKSUMS.md5  — md5 per file for integrity verification
"""

import os
import sys
import time
import json
import hashlib
import argparse
import requests
import pandas as pd
from pathlib import Path
from datetime import date

BASE     = Path(__file__).resolve().parent.parent
META_CSV = BASE / "data" / "raw" / "indofloods" / "metadata_indofloods.csv"

ERA5_VARS = [
    "precipitation_sum",          # total daily precip (mm)
    "et0_fao_evapotranspiration", # reference ET (mm)
    "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
]

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def compute_md5(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_era5_gauge(lat: float, lon: float, start: str, end: str,
                     retries: int = 3, backoff: float = 2.0) -> pd.DataFrame:
    """Fetch ERA5-Land daily data for one lat/lon from Open-Meteo Archive API."""
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": start,
        "end_date":   end,
        "daily":      ",".join(ERA5_VARS),
        "timezone":   "Asia/Kolkata",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            daily = data.get("daily", {})
            df = pd.DataFrame(daily)
            df.rename(columns={"time": "date"}, inplace=True)
            df["lat"] = lat
            df["lon"] = lon
            return df
        except Exception as e:
            if attempt < retries - 1:
                wait = backoff ** (attempt + 1)
                print(f"    Retry {attempt+1}/{retries} after {wait:.0f}s: {e}")
                time.sleep(wait)
            else:
                print(f"    FAILED after {retries} retries: {e}")
                return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Download ERA5-Land via Open-Meteo Archive API")
    parser.add_argument("--start",  default="1970-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default=str(date.today()), help="End date YYYY-MM-DD")
    parser.add_argument("--outdir", default=str(BASE / "data" / "raw" / "era5"))
    parser.add_argument("--limit",  type=int, default=None, help="Max gauges (for testing)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(META_CSV)
    meta = meta.dropna(subset=["Latitude", "Longitude"])
    if args.limit:
        meta = meta.head(args.limit)

    print(f"[ERA5] Downloading for {len(meta)} gauges: {args.start} to {args.end}")
    print(f"[ERA5] Variables: {ERA5_VARS}")
    print(f"[ERA5] Output: {outdir}")

    manifest_rows = []
    checksums = []

    for idx, row in meta.iterrows():
        gauge_id = str(row["GaugeID"])
        lat      = float(row["Latitude"])
        lon      = float(row["Longitude"])
        outfile  = outdir / f"{gauge_id}.csv"

        if outfile.exists():
            print(f"  [{idx+1}/{len(meta)}] {gauge_id} — already downloaded, skipping")
            n_rows = len(pd.read_csv(outfile))
        else:
            print(f"  [{idx+1}/{len(meta)}] {gauge_id} ({lat:.4f}, {lon:.4f}) ...")
            df = fetch_era5_gauge(lat, lon, args.start, args.end)
            if df.empty:
                print(f"    [SKIP] No data returned")
                continue
            df["GaugeID"] = gauge_id
            df.to_csv(outfile, index=False)
            n_rows = len(df)
            print(f"    Saved {n_rows} rows -> {outfile.name}")
            time.sleep(0.2)  # rate limit: ~5 req/s

        md5 = compute_md5(str(outfile))
        checksums.append(f"{md5}  {outfile.name}")
        manifest_rows.append({
            "GaugeID":    gauge_id,
            "lat":        lat,
            "lon":        lon,
            "start_date": args.start,
            "end_date":   args.end,
            "n_rows":     n_rows,
            "file":       outfile.name,
            "md5":        md5,
        })

    # Save manifest
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = outdir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\n[ERA5] Manifest saved: {manifest_path} ({len(manifest)} gauges)")

    # Save checksums
    checksum_path = outdir / "CHECKSUMS.md5"
    with open(checksum_path, "w") as f:
        f.write("\n".join(checksums) + "\n")
    print(f"[ERA5] Checksums saved: {checksum_path}")
    print("[ERA5] Done.")


if __name__ == "__main__":
    main()
