"""
download_pretrained_checkpoints.py  --  FloodSentinel AI
=========================================================
Downloads pre-trained model checkpoints from gs://sen1floods11/v1.1/checkpoints/

Available checkpoints (confirmed 2026-10-20):
  - 5e4_flood_0_89_0.5418742895126343.cp        (~126 MB)  Flood-specific
  - 5e4_handlabeled_0_8_0.43163052201271057.cp  (~126 MB)  Hand-labeled fine-tuned
  - 5e4_permanent_0_180_0.8432363867759705.cp   (~126 MB)  Permanent water
  - 5e4_s1weak_2_3_0.45159465074539185.cp       (~126 MB)  S1 weakly-labeled

These are TensorFlow/Keras checkpoint files from the original Bonafilia et al. paper.
NOTE: These are for the original paper's CNN -- NOT the new U-Net architecture.
They may be useful as:
  - Feature extractors for initializing encoder weights
  - Baseline comparison (original paper model performance)
  - Understanding original paper's architecture

WARNING: The original paper used S2 optical imagery (not just S1 SAR).
  The `5e4_handlabeled` checkpoint may be S1-only but verify before use.

Run: python scripts/download_pretrained_checkpoints.py

Output: data/raw/sen1floods11/checkpoints/
"""

import sys
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE     = Path(__file__).resolve().parent.parent
CKPT_DIR = BASE / "data" / "raw" / "sen1floods11" / "checkpoints"
GCS_BUCKET = "sen1floods11"

# Confirmed in GCS listing 2026-10-20
CHECKPOINTS = [
    "v1.1/checkpoints/5e4_flood_0_89_0.5418742895126343.cp",
    "v1.1/checkpoints/5e4_handlabeled_0_8_0.43163052201271057.cp",
    "v1.1/checkpoints/5e4_permanent_0_180_0.8432363867759705.cp",
    "v1.1/checkpoints/5e4_s1weak_2_3_0.45159465074539185.cp",
]


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_checkpoint(blob_name: str) -> dict:
    from google.cloud import storage
    fname  = Path(blob_name).name
    target = CKPT_DIR / fname

    if target.exists() and target.stat().st_size > 1000:
        md5 = _md5(target)
        print(f"  [EXISTS] {fname}  ({target.stat().st_size/(1024**2):.1f} MB)  md5={md5}")
        return {"name": fname, "size": target.stat().st_size, "md5": md5, "downloaded": False}

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(GCS_BUCKET)
    blob   = bucket.blob(blob_name)

    import time
    t0 = time.time()
    temp = target.with_suffix(".tmp")
    blob.download_to_filename(str(temp))
    temp.replace(target)
    elapsed = time.time() - t0
    md5 = _md5(target)
    size = target.stat().st_size
    print(f"  [OK] {fname}  ({size/(1024**2):.1f} MB, {size/elapsed/1024**2:.1f} MB/s)  md5={md5}")
    return {"name": fname, "size": size, "md5": md5, "downloaded": True}


def main():
    try:
        from google.cloud import storage
    except ImportError:
        raise RuntimeError("google-cloud-storage not installed. Run: pip install google-cloud-storage")

    print("=" * 70)
    print("  Downloading Sen1Floods11 Pre-trained Checkpoints")
    print(f"  Source: gs://{GCS_BUCKET}/v1.1/checkpoints/")
    print(f"  Target: {CKPT_DIR}")
    print("  NOTE: These are TF/Keras checkpoints from the original paper.")
    print("  Use for reference comparison only; NOT as the new U-Net baseline.")
    print("=" * 70)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for blob_name in CHECKPOINTS:
        try:
            r = download_checkpoint(blob_name)
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] {blob_name}: {e}")
            raise   # fail loudly

    print(f"\n  Downloaded {len(results)} checkpoints to {CKPT_DIR}")

    # Save manifest
    import json
    manifest = {
        "source":        f"gs://{GCS_BUCKET}/v1.1/checkpoints/",
        "license":       "CC BY 4.0 (Sen1Floods11 dataset license)",
        "paper":         "Bonafilia et al. CVPRW 2020",
        "model_type":    "TensorFlow/Keras (original paper architecture)",
        "note":          "These are NOT the new U-Net architecture. Use for comparison only.",
        "checkpoints":   results,
    }
    manifest_path = CKPT_DIR / "checkpoint_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {manifest_path}")

    print("\n  [OK] Checkpoint download complete.")
    print("  Next: Inspect checkpoint architecture to determine S1/S2 input channels.")


if __name__ == "__main__":
    main()
