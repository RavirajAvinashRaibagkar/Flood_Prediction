"""
download_sen1floods11_weak.py  --  FloodSentinel AI  Phase 2a
=============================================================
Downloads the Sen1Floods11 weakly-labeled S1 chips (S1Weak folder in GCS).
These are used ONLY for pretraining / pseudo-label stages.
They are NEVER used for val/test/bolivia.

GCS path: gs://sen1floods11/v1.1/data/flood_events/S1Weak/
Expected count: ~4,385 chips

License: CC BY 4.0 (same as HandLabeled)
Rules:
  - Exclude any chip whose ID appears in val/test/bolivia official splits
  - Document per-source counts in manifest.csv
  - Compute MD5 checksums

Run: python flood_project/ml_training/scripts/download_sen1floods11_weak.py [--limit N]

Outputs:
  data/raw/sen1floods11/S1Weak/         -- downloaded GeoTIFFs
  data/raw/sen1floods11/S1WeakManifest.csv
"""

import sys
import csv
import json
import hashlib
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE    = Path(__file__).resolve().parent.parent
OUTDIR  = BASE / "data" / "raw" / "sen1floods11"
S1WEAK  = OUTDIR / "WeaklyLabeled" / "S1Weak"
SPLIT_DIR = OUTDIR / "splits"

GCS_BUCKET  = "sen1floods11"
# Actual path confirmed by listing gs://sen1floods11 (2026-10-20):
GCS_S1WEAK  = "v1.1/data/flood_events/WeaklyLabeled/S1Weak"


def load_official_holdout_ids() -> set:
    """
    Load chip IDs from val, test, and bolivia splits.
    Weak labels MUST exclude these chips.
    Raises RuntimeError if split files not found.
    """
    holdout_ids = set()
    split_files = {
        "flood_valid_data.csv":   "val",
        "flood_test_data.csv":    "test",
        "flood_bolivia_data.csv": "bolivia",
    }

    if not SPLIT_DIR.exists():
        raise RuntimeError(
            f"Split directory not found: {SPLIT_DIR}\n"
            "Run: python scripts/phase0_list_gcs_splits.py  first."
        )

    for fname, split in split_files.items():
        fpath = SPLIT_DIR / fname
        if not fpath.exists():
            raise RuntimeError(
                f"Official split file not found: {fpath}\n"
                "Run: python scripts/phase0_list_gcs_splits.py"
            )
        with open(fpath, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    chip_id = row[0].strip().replace("_S1Hand.tif", "")
                    holdout_ids.add(chip_id)

    print(f"  Loaded {len(holdout_ids)} holdout chip IDs (val + test + bolivia).")
    return holdout_ids


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_blob(bucket_name: str, blob_name: str, target: Path) -> tuple:
    from google.cloud import storage
    if target.exists() and target.stat().st_size > 0:
        return str(target), target.stat().st_size, _md5(target), False

    target.parent.mkdir(parents=True, exist_ok=True)
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)
    temp   = target.with_suffix(".tmp")
    blob.download_to_filename(str(temp))
    temp.replace(target)
    return str(target), target.stat().st_size, _md5(target), True


def main():
    parser = argparse.ArgumentParser(description="Download Sen1Floods11 S1Weak chips")
    parser.add_argument("--limit",   type=int, default=None, help="Limit number of chips (testing)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--outdir",  default=str(OUTDIR))
    parser.add_argument("--list-only", action="store_true", help="List available chips, don't download")
    args = parser.parse_args()

    print("=" * 70)
    print("  Sen1Floods11 S1Weak Download  --  Phase 2a")
    print(f"  Source: gs://{GCS_BUCKET}/{GCS_S1WEAK}/")
    print(f"  License: CC BY 4.0 (same as HandLabeled)")
    print("=" * 70)

    try:
        from google.cloud import storage
    except ImportError:
        raise RuntimeError("google-cloud-storage not installed. Run: pip install google-cloud-storage")

    # 1. Load holdout IDs (val/test/bolivia) -- these must be excluded
    print("\n[1] Loading official holdout split IDs (to exclude from weak labels)...")
    holdout_ids = load_official_holdout_ids()

    # 2. List S1Weak blobs in GCS
    print(f"\n[2] Listing gs://{GCS_BUCKET}/{GCS_S1WEAK}/...")
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(GCS_BUCKET)

    # Try both with and without trailing slash
    blobs_s1 = list(bucket.list_blobs(prefix=GCS_S1WEAK + "/"))
    print(f"  Found {len(blobs_s1)} blobs in S1Weak/")

    # Also check for label blobs if they exist
    label_prefix = "v1.1/data/flood_events/S1WeakLabel"
    blobs_lbl = list(bucket.list_blobs(prefix=label_prefix))
    print(f"  Found {len(blobs_lbl)} blobs in S1WeakLabel/ (may be 0 -- weak means no labels)")

    if not blobs_s1:
        # Try alternate names
        alt_prefixes = [
            "v1.1/data/flood_events/S1Weak",
            "v1.1/data/flood_events/WeaklyLabeled",
            "v1.1/data/flood_events/S1Weakly",
        ]
        for alt in alt_prefixes:
            blobs_alt = list(bucket.list_blobs(prefix=alt))
            if blobs_alt:
                print(f"  Found {len(blobs_alt)} blobs at: {alt}/")
                blobs_s1 = blobs_alt
                break
        if not blobs_s1:
            print("  [WARN] S1Weak folder not found in GCS. Listing all v1.1 top-level prefixes:")
            for b in bucket.list_blobs(prefix="v1.1/"):
                parts = b.name.split("/")
                if len(parts) >= 4:
                    print(f"    {'/'.join(parts[:4])}/")
                    break
            raise RuntimeError(
                "S1Weak chips not found in gs://sen1floods11. "
                "Check the bucket structure or the paper for the correct path.\n"
                "Run: python scripts/phase0_list_gcs_splits.py to see all prefixes."
            )

    if args.list_only:
        print(f"\n  Available S1Weak chips: {len(blobs_s1)}")
        for b in blobs_s1[:10]:
            print(f"    {b.name}  ({b.size} bytes)")
        if len(blobs_s1) > 10:
            print(f"    ... ({len(blobs_s1)-10} more)")
        return

    # 3. Filter to chips only (not directories) and exclude holdout
    tasks = []
    excluded_count = 0
    event_counts = {}

    for blob in blobs_s1:
        fname = Path(blob.name).name
        if not fname.endswith(".tif"):
            continue
        chip_id = fname.replace("_S1Hand.tif", "").replace(".tif", "")

        # Extract event from chip_id
        event = chip_id.rsplit("_", 1)[0] if "_" in chip_id else chip_id

        # Exclude holdout chips
        if chip_id in holdout_ids:
            excluded_count += 1
            continue

        target = S1WEAK / fname
        tasks.append((GCS_BUCKET, blob.name, target, chip_id, event))
        event_counts[event] = event_counts.get(event, 0) + 1

    if args.limit:
        tasks = tasks[:args.limit]

    print(f"\n  Total S1Weak blobs: {len(blobs_s1)}")
    print(f"  Excluded (holdout val/test/bolivia): {excluded_count}")
    print(f"  To download (train-safe weak chips): {len(tasks)}")
    print(f"  Events represented: {sorted(event_counts.keys())}")

    if not tasks:
        print("  [WARN] No S1Weak chips to download after exclusion.")
        return

    S1WEAK.mkdir(parents=True, exist_ok=True)

    # 4. Download
    print(f"\n[3] Downloading {len(tasks)} weak S1 chips ({args.workers} threads)...")
    t0 = time.time()
    records = []
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(download_blob, bname, blob_name, target): (chip_id, event)
            for bname, blob_name, target, chip_id, event in tasks
        }
        for future in as_completed(future_map):
            chip_id, event = future_map[future]
            try:
                p, sz, md5, downloaded = future.result()
                records.append({
                    "chip_id": chip_id,
                    "event":   event,
                    "split":   "train_weak",
                    "size_bytes": sz,
                    "md5":     md5,
                    "path":    p,
                })
                completed += 1
                if completed % 100 == 0 or completed == len(tasks):
                    print(f"  [{completed}/{len(tasks)}]  {time.time()-t0:.0f}s elapsed")
            except Exception as e:
                print(f"  [ERROR] {chip_id}: {e}")
                raise   # fail loudly

    elapsed = time.time() - t0
    total_gb = sum(r["size_bytes"] for r in records) / (1024**3)
    print(f"\n  Downloaded {completed} chips in {elapsed:.1f}s  ({total_gb:.2f} GB)")

    # 5. Save manifest
    manifest_path = OUTDIR / "S1WeakManifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chip_id","event","split","size_bytes","md5","path"])
        writer.writeheader()
        writer.writerows(records)
    print(f"  Manifest: {manifest_path}")

    # Per-event counts
    print("\n  Per-event chip counts (S1Weak, train-safe):")
    for ev, cnt in sorted(event_counts.items()):
        print(f"    {ev:<20}: {cnt}")

    # 6. Checksums
    cksum_path = OUTDIR / "S1Weak_CHECKSUMS.md5"
    with open(cksum_path, "w") as f:
        for r in records:
            f.write(f"{r['md5']}  S1Weak/{Path(r['path']).name}\n")
    print(f"  Checksums: {cksum_path}")

    print(f"\n  [OK] S1Weak download complete: {len(records)} chips.")
    print(f"  IMPORTANT: These chips are for PRETRAINING ONLY.")
    print(f"  DO NOT use for validation, testing, or threshold calibration.")


if __name__ == "__main__":
    main()
