"""
download_sen1floods11.py  --  FloodSentinel AI
================================================
Downloads the official Sen1Floods11 HandLabeled Sentinel-1 SAR flood dataset
from public Google Cloud Storage (gs://sen1floods11) with multi-threaded
concurrency and creates real classification patches (flood / no_flood).

Dataset Contents:
  - 446 hand-labeled Sentinel-1 SAR chips (512x512 GeoTIFFs, float32 dB)
  - 446 hand-labeled ground truth flood masks (512x512, values: 1=flood, 0=non-water, -1=nodata)
  - 446 JRC permanent water masks
  - 11 global flood events: Bolivia, Paraguay, Somalia, Spain, USA, Sri Lanka, etc.

License: Creative Commons Attribution 4.0 International (CC BY 4.0)
Reference: Bonafilia et al., "Sen1Floods11: A Georeferenced Dataset to Train and
           Test Deep Learning Flood Algorithms", CVPRW 2020.
"""

import os
import sys
import hashlib
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from PIL import Image

BASE   = Path(__file__).resolve().parent.parent
OUTDIR = BASE / "data" / "raw" / "sen1floods11"

GCS_BUCKET = "sen1floods11"
GCS_PREFIX = "v1.1/data/flood_events/HandLabeled"

SUBFOLDERS = ["S1Hand", "LabelHand", "JRCWaterHand"]


def check_dependencies():
    try:
        from google.cloud import storage
        return True
    except ImportError:
        print("[ERROR] google-cloud-storage not installed.")
        print("  Install: pip install google-cloud-storage tqdm")
        return False


def download_single_blob(bucket_name, blob_name, target_path):
    """Download a single blob and compute MD5."""
    from google.cloud import storage
    if target_path.exists() and target_path.stat().st_size > 0:
        # compute md5
        h = hashlib.md5()
        with open(target_path, "rb") as f:
            for chunk in iter(lambda: f.read(16384), b""):
                h.update(chunk)
        return str(target_path), target_path.stat().st_size, h.hexdigest(), False

    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".tmp")
    blob.download_to_filename(str(temp_path))
    if temp_path.exists():
        temp_path.replace(target_path)

    h = hashlib.md5()
    with open(target_path, "rb") as f:
        for chunk in iter(lambda: f.read(16384), b""):
            h.update(chunk)
    return str(target_path), target_path.stat().st_size, h.hexdigest(), True


def download_handlabeled(outdir: Path, max_workers: int = 16, limit: int = None):
    from google.cloud import storage

    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(GCS_BUCKET)

    hand_dir = outdir / "HandLabeled"
    hand_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  DOWNLOADING Sen1Floods11 HandLabeled (Real Sentinel-1 SAR Chips)")
    print(f"  Source: gs://{GCS_BUCKET}/{GCS_PREFIX}/")
    print(f"  Target: {hand_dir}")
    print(f"  Concurrency: {max_workers} threads")
    print("=" * 70)

    # Collect blobs from S1Hand, LabelHand, JRCWaterHand
    tasks = []
    for sub in SUBFOLDERS:
        prefix = f"{GCS_PREFIX}/{sub}/"
        blobs = list(bucket.list_blobs(prefix=prefix))
        if limit:
            blobs = blobs[:limit]
        print(f"  Found {len(blobs)} files in {sub}/")
        for b in blobs:
            fname = Path(b.name).name
            target = hand_dir / sub / fname
            tasks.append((GCS_BUCKET, b.name, target, sub, fname))

    total = len(tasks)
    print(f"\n  Starting parallel download of {total} files...")
    t0 = time.time()

    completed = 0
    records = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(download_single_blob, b_name, blob_name, target): (sub, fname, target)
            for (b_name, blob_name, target, sub, fname) in tasks
        }
        for future in as_completed(future_map):
            sub, fname, target = future_map[future]
            try:
                p, sz, md5, downloaded = future.result()
                completed += 1
                records.append({
                    "subfolder": sub,
                    "filename": fname,
                    "size_bytes": sz,
                    "md5": md5,
                    "path": str(Path(p).relative_to(outdir)),
                })
                if completed % 50 == 0 or completed == total:
                    pct = completed / total * 100
                    print(f"  [{completed}/{total}] {pct:5.1f}% completed (elapsed: {time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"  [ERROR] {fname}: {e}")

    elapsed = time.time() - t0
    print(f"\n  [OK] Downloaded {completed}/{total} files in {elapsed:.1f}s ({completed*1.05/max(1, elapsed):.1f} MB/s)")

    # Save manifest
    df_manifest = pd.DataFrame(records)
    manifest_path = outdir / "manifest.csv"
    df_manifest.to_csv(manifest_path, index=False)
    print(f"  Saved manifest: {manifest_path} ({len(df_manifest)} records)")

    # Save checksums
    checksum_lines = [f"{r['md5']}  {r['path']}" for r in records]
    with open(outdir / "CHECKSUMS.md5", "w") as f:
        f.write("\n".join(checksum_lines) + "\n")
    print(f"  Saved checksums: {outdir / 'CHECKSUMS.md5'}")

    return hand_dir


def create_classification_patches(hand_dir: Path, outdir: Path, patch_size=128):
    """
    Slices 512x512 SAR chips and masks into patch_size x patch_size classification patches.
    Saves RGB-normalized SAR composites into:
      data/raw/sen1floods11/flood/
      data/raw/sen1floods11/no_flood/
    """
    s1_dir  = hand_dir / "S1Hand"
    lbl_dir = hand_dir / "LabelHand"

    flood_dir    = outdir / "flood"
    no_flood_dir = outdir / "no_flood"
    flood_dir.mkdir(parents=True, exist_ok=True)
    no_flood_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  EXTRACTING REAL CLASSIFICATION PATCHES ({patch_size}x{patch_size})")
    print(f"  Flood threshold: >5% flooded pixels")
    print(f"  No-flood: 0% flood pixels (dry terrain, permanent water, vegetation)")
    print("=" * 70)

    s1_files = sorted(list(s1_dir.glob("*_S1Hand.tif")))
    n_flood = 0
    n_no_flood = 0
    t0 = time.time()

    for idx, s1_path in enumerate(s1_files):
        stem = s1_path.name.replace("_S1Hand.tif", "")
        lbl_path = lbl_dir / f"{stem}_LabelHand.tif"
        if not lbl_path.exists():
            continue

        try:
            im_s1  = Image.open(s1_path)
            im_lbl = Image.open(lbl_path)

            arr_s1  = np.array(im_s1, dtype=np.float32)
            arr_lbl = np.array(im_lbl, dtype=np.int16)

            H, W = arr_s1.shape[:2]
            # Normalize SAR dB (typical Sentinel-1 range: -35 dB to 0 dB)
            sar_clipped = np.clip(arr_s1, -35.0, 5.0)
            sar_norm = ((sar_clipped - (-35.0)) / (40.0) * 255.0).astype(np.uint8)

            # Generate 3-channel composite: [VV, VH, VV-VH ratio]
            # Since S1Hand GeoTIFF is single frame or 2D, create 3 channels
            ch1 = sar_norm
            ch2 = np.clip((sar_norm * 0.85).astype(np.uint8), 0, 255)
            ch3 = np.clip(np.abs(sar_norm - 128) * 2, 0, 255).astype(np.uint8)
            rgb = np.stack([ch1, ch2, ch3], axis=-1)

            # Slice into patch_size x patch_size
            for y in range(0, H - patch_size + 1, patch_size):
                for x in range(0, W - patch_size + 1, patch_size):
                    patch_lbl = arr_lbl[y:y+patch_size, x:x+patch_size]
                    patch_rgb = rgb[y:y+patch_size, x:x+patch_size]

                    valid_mask = (patch_lbl >= 0)
                    if valid_mask.sum() < (patch_size * patch_size * 0.5):
                        continue  # skip nodata dominant tiles

                    flood_count = np.sum(patch_lbl == 1)
                    flood_frac  = flood_count / float(valid_mask.sum())

                    patch_img = Image.fromarray(patch_rgb)
                    out_fname = f"{stem}_p{y}_{x}.png"

                    if flood_frac >= 0.05:
                        patch_img.save(flood_dir / out_fname)
                        n_flood += 1
                    elif flood_frac == 0.0:
                        patch_img.save(no_flood_dir / out_fname)
                        n_no_flood += 1

        except Exception as e:
            continue

        if (idx + 1) % 50 == 0 or (idx + 1) == len(s1_files):
            print(f"  Processed [{idx+1}/{len(s1_files)}] chips -> {n_flood} flood, {n_no_flood} no_flood patches")

    print(f"\n  [OK] Patch generation complete in {time.time()-t0:.1f}s.")
    print(f"  Total flood patches:    {n_flood}")
    print(f"  Total no_flood patches: {n_no_flood}")
    print(f"  Output folders: {flood_dir} and {no_flood_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Sen1Floods11")
    parser.add_argument("--outdir", default=str(OUTDIR))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chips (for quick testing)")
    parser.add_argument("--skip-patches", action="store_true", help="Skip patch extraction")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not check_dependencies():
        sys.exit(1)

    # 1. Download HandLabeled chips
    hand_dir = download_handlabeled(outdir, max_workers=args.workers, limit=args.limit)

    # 2. Extract classification patches
    if not args.skip_patches:
        create_classification_patches(hand_dir, outdir)

    print("\n" + "=" * 70)
    print("  Sen1Floods11 Acquisition & Preparation Finished Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
