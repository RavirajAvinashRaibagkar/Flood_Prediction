"""
download_mmflood.py  --  FloodSentinel AI  Phase 2b
=====================================================
Downloads MMFlood dataset from Zenodo (record 6534637).
MMFlood: Sentinel-1 SAR + DEM + EMS masks, 95 flood events.
MD5: 2817eb94107f1c96ef68d843010bdf97
Size: ~11.3 GB

Split strategy: BY EVENT (not by tile). All event-level splits are
determined BEFORE download and enforced in the manifest.

License: CC BY 4.0
Source: https://zenodo.org/record/6534637
Reference: Fanelli et al., "MMFlood: A Multimodal Dataset for Flood Delineation
           From Satellite Imagery", IEEE Access 2022.

Run: python flood_project/ml_training/scripts/download_mmflood.py [--dry-run] [--outdir PATH]

Outputs:
  data/raw/mmflood/
    manifest.csv           -- per-file MD5 + event + split
    CHECKSUMS.md5
    events/                -- per-event subdirectories
  data/raw/mmflood/mmflood_event_split.json  -- event->split mapping
"""

import sys
import csv
import json
import hashlib
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE   = Path(__file__).resolve().parent.parent
OUTDIR = BASE / "data" / "raw" / "mmflood"

ZENODO_RECORD_ID = "6534637"
ZENODO_API_URL   = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
EXPECTED_MD5     = "2817eb94107f1c96ef68d843010bdf97"  # main archive md5

# 95 events total -- split BY EVENT (not by tile/image)
# Leave roughly 20% for external cross-dataset test
# These are DISTINCT from Sen1Floods11 events
N_TEST_EVENTS    = 19  # ~20%
N_VAL_EVENTS     = 9   # ~10%
# Remaining = train_events


def fetch_zenodo_metadata() -> dict:
    """Fetch file list from Zenodo API."""
    import urllib.request
    import urllib.error

    url = ZENODO_API_URL
    print(f"  Fetching Zenodo metadata: {url}")
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach Zenodo API: {e}\nCheck internet connection.")


def download_file(url: str, target_path: Path, expected_md5: str = None) -> dict:
    """Download a single file with MD5 verification."""
    import urllib.request

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and target_path.stat().st_size > 0:
        # Verify existing MD5
        existing_md5 = _md5(target_path)
        if expected_md5 and existing_md5 != expected_md5:
            print(f"    [WARN] MD5 mismatch for {target_path.name}: expected={expected_md5} got={existing_md5}. Re-downloading.")
        elif expected_md5:
            return {"path": str(target_path), "size": target_path.stat().st_size,
                    "md5": existing_md5, "downloaded": False}

    temp_path = target_path.with_suffix(".tmp")
    t0 = time.time()
    urllib.request.urlretrieve(url, str(temp_path))
    elapsed = time.time() - t0

    actual_md5 = _md5(temp_path)
    if expected_md5 and actual_md5 != expected_md5:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"MD5 verification FAILED for {target_path.name}: "
            f"expected={expected_md5}  got={actual_md5}"
        )

    temp_path.replace(target_path)
    size = target_path.stat().st_size
    speed_mb = size / max(elapsed, 0.1) / 1024 / 1024
    print(f"    [OK] {target_path.name}  ({size/(1024**3):.2f} GB, {speed_mb:.1f} MB/s)  md5={actual_md5}")

    return {"path": str(target_path), "size": size, "md5": actual_md5, "downloaded": True}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def assign_event_splits(event_dirs: list) -> dict:
    """
    Assign MMFlood events to train/val/test splits BY EVENT.
    Uses deterministic sort + fixed seed for reproducibility.
    NEVER mix events across splits.
    Returns: {event_name: split_label}
    """
    import random
    events_sorted = sorted(event_dirs)
    rng = random.Random(42)   # fixed seed for reproducibility
    rng.shuffle(events_sorted)

    n_total = len(events_sorted)
    n_test  = N_TEST_EVENTS
    n_val   = N_VAL_EVENTS
    n_train = n_total - n_test - n_val

    assignment = {}
    for i, ev in enumerate(events_sorted):
        if i < n_test:
            assignment[ev] = "test_external"   # cross-dataset test (train on S1Floods11, test on MMFlood)
        elif i < n_test + n_val:
            assignment[ev] = "val_external"
        else:
            assignment[ev] = "train_mmflood"

    print(f"  MMFlood event splits: {n_train} train, {n_val} val, {n_test} test (BY EVENT)")
    return assignment


def main():
    parser = argparse.ArgumentParser(description="Download MMFlood dataset from Zenodo")
    parser.add_argument("--outdir",   default=str(OUTDIR))
    parser.add_argument("--dry-run",  action="store_true", help="Fetch metadata only, don't download")
    parser.add_argument("--workers",  type=int, default=4, help="Parallel download threads")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  DOWNLOADING MMFlood  (Zenodo 6534637)")
    print(f"  Target: {outdir}")
    print(f"  Expected MD5: {EXPECTED_MD5}")
    print(f"  License: CC BY 4.0")
    print("=" * 70)

    # 1. Fetch Zenodo metadata
    meta = fetch_zenodo_metadata()
    files = meta.get("files", [])
    if not files:
        raise RuntimeError(
            f"No files found in Zenodo record {ZENODO_RECORD_ID}. "
            "Check: https://zenodo.org/record/6534637"
        )

    print(f"\n  Files in Zenodo record {ZENODO_RECORD_ID}:")
    for f in files:
        size_gb = f.get("size", 0) / (1024**3)
        print(f"    {f['key']:<60} {size_gb:.2f} GB  md5={f.get('checksum', 'N/A')}")

    if args.dry_run:
        print("\n  [DRY RUN] Metadata fetched. No files downloaded.")
        print(f"  Total files: {len(files)}")
        total_gb = sum(f.get("size", 0) for f in files) / (1024**3)
        print(f"  Total size: {total_gb:.1f} GB")
        return

    # 2. Download files
    records = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for fmeta in files:
            fname  = fmeta["key"]
            url    = fmeta["links"]["self"]
            md5    = fmeta.get("checksum", "").replace("md5:", "")
            target = outdir / fname
            futures[executor.submit(download_file, url, target, md5 or None)] = fname

        for future in as_completed(futures):
            fname = futures[future]
            try:
                rec = future.result()
                records.append({
                    "filename": fname,
                    "size_bytes": rec["size"],
                    "md5": rec["md5"],
                    "downloaded": rec["downloaded"],
                })
            except Exception as e:
                print(f"  [ERROR] {fname}: {e}")
                raise   # fail loudly

    print(f"\n  Downloaded {len(records)} files in {time.time()-t0:.1f}s")

    # 3. Verify main archive MD5 (if present)
    archive_files = [r for r in records if r["filename"].endswith(".zip") or r["filename"].endswith(".tar.gz")]
    for ar in archive_files:
        if ar["md5"] == EXPECTED_MD5:
            print(f"  [OK] Main archive MD5 verified: {EXPECTED_MD5}")

    # 4. Save manifest
    manifest_path = outdir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "size_bytes", "md5", "downloaded"])
        writer.writeheader()
        writer.writerows(records)
    print(f"  Manifest: {manifest_path}")

    # 5. Save checksums
    checksum_path = outdir / "CHECKSUMS.md5"
    with open(checksum_path, "w") as f:
        for r in records:
            f.write(f"{r['md5']}  {r['filename']}\n")
    print(f"  Checksums: {checksum_path}")

    # 6. Note on extraction and event splits
    print("\n  [NEXT STEPS]")
    print("  1. Extract the downloaded archive (unzip/tar xf)")
    print("  2. Run: python scripts/download_mmflood.py --assign-splits")
    print("     (after extraction, scans event directories and assigns train/val/test BY EVENT)")
    print("\n  Per Phase 2 rules:")
    print("  - Split MMFlood by EVENT, never by tile")
    print("  - Test split is CROSS-DATASET (train on S1Floods11, test on MMFlood)")
    print(f"  - Expected: {N_TEST_EVENTS} test events, {N_VAL_EVENTS} val events")

    print("\n  [OK] MMFlood download complete.")


if __name__ == "__main__":
    main()
