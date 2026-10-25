"""
download_glofas_openmeteo.py  --  FloodSentinel AI Phase 2
===========================================================
Downloads GloFAS reanalysis river discharge (m^3/s) via the
Open-Meteo Flood API for each gauge location in the INDOFLOODS metadata.
Free, no API key required.

License: GloFAS data via Open-Meteo is CC BY 4.0
Docs: https://open-meteo.com/en/docs/flood-api

Usage:
  python download_glofas_openmeteo.py [--start 1984 --end 2023] [--outdir data/raw/glofas]

Output:
  data/raw/glofas/<GaugeID>.csv  -- daily discharge per gauge
  data/raw/glofas/manifest.csv
  data/raw/glofas/CHECKSUMS.md5
"""

import os
import sys
import time
import hashlib
import argparse
import requests
import pandas as pd
from pathlib import Path
from datetime import date

BASE     = Path(__file__).resolve().parent.parent
META_CSV = BASE / "data" / "raw" / "indofloods" / "metadata_indofloods.csv"

GLOFAS_URL = "https://flood-api.open-meteo.com/v1/flood"


def compute_md5(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_glofas_gauge(lat: float, lon: float, start: str, end: str,
                       retries: int = 3, backoff: float = 2.0) -> pd.DataFrame:
    """Fetch GloFAS daily discharge for one lat/lon."""
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start,
        "end_date":   end,
        "daily":      "river_discharge",
        "ensemble":   "false",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(GLOFAS_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            daily = data.get("daily", {})
            df = pd.DataFrame(daily)
            if df.empty:
                return df
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
    parser = argparse.ArgumentParser(description="Download GloFAS discharge via Open-Meteo Flood API")
    parser.add_argument("--start",  default="1984-01-01")
    parser.add_argument("--end",    default=str(date.today()))
    parser.add_argument("--outdir", default=str(BASE / "data" / "raw" / "glofas"))
    parser.add_argument("--limit",  type=int, default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(META_CSV)
    meta = meta.dropna(subset=["Latitude", "Longitude"])
    if args.limit:
        meta = meta.head(args.limit)

    print(f"[GloFAS] Downloading for {len(meta)} gauges: {args.start} to {args.end}")
    print(f"[GloFAS] Output: {outdir}")

    manifest_rows = []
    checksums = []

    for idx, row in meta.iterrows():
        gauge_id = str(row["GaugeID"])
        lat      = float(row["Latitude"])
        lon      = float(row["Longitude"])
        outfile  = outdir / f"{gauge_id}.csv"

        if outfile.exists():
            print(f"  [{idx+1}/{len(meta)}] {gauge_id} — cached, skipping")
            n_rows = len(pd.read_csv(outfile))
        else:
            print(f"  [{idx+1}/{len(meta)}] {gauge_id} ({lat:.4f}, {lon:.4f}) ...")
            df = fetch_glofas_gauge(lat, lon, args.start, args.end)
            if df.empty:
                print(f"    [SKIP] No data returned")
                continue
            df["GaugeID"] = gauge_id
            df.to_csv(outfile, index=False)
            n_rows = len(df)
            print(f"    Saved {n_rows} rows -> {outfile.name}")
            time.sleep(0.2)

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

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = outdir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\n[GloFAS] Manifest: {manifest_path} ({len(manifest)} gauges)")

    with open(outdir / "CHECKSUMS.md5", "w") as f:
        f.write("\n".join(checksums) + "\n")
    print(f"[GloFAS] Checksums saved.")
    print("[GloFAS] Done.")


if __name__ == "__main__":
    main()
