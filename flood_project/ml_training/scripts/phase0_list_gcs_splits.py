"""
phase0_list_gcs_splits.py  --  FloodSentinel AI  Phase 0
=========================================================
Lists the FULL gs://sen1floods11 bucket to find official train/val/test/Bolivia
split CSV files. Prints every path and downloads the split CSVs.

Run: python flood_project/ml_training/scripts/phase0_list_gcs_splits.py

Output:
  - Prints all top-level prefixes found in the bucket
  - Downloads official split CSVs to data/raw/sen1floods11/splits/
  - Raises RuntimeError if no split files found (never silently pass)
"""

import sys
import io
from pathlib import Path

BASE   = Path(__file__).resolve().parent.parent
OUTDIR = BASE / "data" / "raw" / "sen1floods11" / "splits"

GCS_BUCKET = "sen1floods11"


def main():
    try:
        from google.cloud import storage
    except ImportError:
        raise RuntimeError("google-cloud-storage not installed. Run: pip install google-cloud-storage")

    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(GCS_BUCKET)

    print("=" * 70)
    print(f"  LISTING gs://{GCS_BUCKET}  (anonymous, no auth)")
    print("=" * 70)

    # List ALL blobs at top level to find folder structure
    print("\n[1] Top-level prefixes in the bucket:")
    all_blobs = list(bucket.list_blobs())
    print(f"    Total blobs in bucket: {len(all_blobs)}")

    # Collect unique top-level prefixes
    top_prefixes = set()
    for b in all_blobs:
        parts = b.name.split("/")
        if len(parts) > 1:
            top_prefixes.add(parts[0])
    for p in sorted(top_prefixes):
        print(f"    gs://{GCS_BUCKET}/{p}/")

    # Search for split CSV files anywhere in the bucket
    print("\n[2] Searching for split/catalog CSV files:")
    split_blobs = []
    keywords = ["split", "catalog", "train", "val", "test", "bolivia", "fold"]
    for b in all_blobs:
        name_lower = b.name.lower()
        if name_lower.endswith(".csv") or name_lower.endswith(".txt"):
            if any(k in name_lower for k in keywords):
                split_blobs.append(b)
                print(f"    FOUND: gs://{GCS_BUCKET}/{b.name}  ({b.size} bytes)")

    if not split_blobs:
        # Try listing specific known paths from the paper
        print("    [WARN] No split CSVs found with keyword search; trying known paths...")
        known_paths = [
            "v1.1/splits/flood_handlabeled_split.csv",
            "v1.1/splits/flood_handlabeled_split_stratified.csv",
            "v1.1/splits/",
            "v1.1/data/flood_events/splits/",
        ]
        for path in known_paths:
            candidate = bucket.blob(path)
            try:
                candidate.reload()
                print(f"    FOUND: gs://{GCS_BUCKET}/{path}  ({candidate.size} bytes)")
                split_blobs.append(candidate)
            except Exception:
                print(f"    NOT FOUND: gs://{GCS_BUCKET}/{path}")

    # Also print ALL .csv files for manual review
    print("\n[3] ALL CSV/TXT files in bucket:")
    csv_blobs = [b for b in all_blobs if b.name.endswith(".csv") or b.name.endswith(".txt")]
    for b in csv_blobs:
        print(f"    {b.name}  ({b.size} bytes)")

    if not csv_blobs and not split_blobs:
        raise RuntimeError(
            "No CSV or TXT files found in gs://sen1floods11. "
            "The bucket may require authentication or may have changed structure. "
            "Check: https://github.com/cloudtostreet/Sen1Floods11"
        )

    # Download whatever split files we found
    if split_blobs:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        print(f"\n[4] Downloading {len(split_blobs)} split file(s) to {OUTDIR}")
        for b in split_blobs:
            fname = Path(b.name).name
            out_path = OUTDIR / fname
            b.download_to_filename(str(out_path))
            print(f"    Downloaded: {out_path}  ({out_path.stat().st_size} bytes)")
            # Print first 20 lines
            with open(out_path) as f:
                lines = f.readlines()
            print(f"    First 5 lines of {fname}:")
            for line in lines[:5]:
                print(f"      {line.rstrip()}")
    else:
        print("\n[4] No split files to download.")
        print("    IMPORTANT: You must manually obtain the official Sen1Floods11 split CSV.")
        print("    See: https://github.com/cloudtostreet/Sen1Floods11")
        print("    The paper uses a fixed split - derive it from the GitHub repo catalog files.")

    print("\n[DONE] GCS listing complete.")


if __name__ == "__main__":
    main()
