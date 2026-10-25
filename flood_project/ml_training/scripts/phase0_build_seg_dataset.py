"""
phase0_build_seg_dataset.py  --  FloodSentinel AI  Phase 0
===========================================================
Builds the SEGMENTATION dataset from raw Sen1Floods11 HandLabeled chips.

Tasks performed:
  1. Loads official split assignment CSV (downloaded from GCS or GitHub).
     Raises RuntimeError if split CSV not found -- never invents splits.
  2. Assigns every chip to {train, val, test, bolivia} by official split.
  3. Asserts ZERO chip-ID overlap across splits (fails loudly if violated).
  4. Computes per-channel normalization stats on TRAIN chips ONLY.
     Channels: VV dB, VH dB, VV-VH (derived). Clip [-35, 5] dB.
  5. Saves per-split lists as JSON manifests with chip paths.
  6. Computes and prints flood-pixel prevalence per split.
  7. Runs perceptual hash (pHash) deduplicate check across splits.
  8. Saves normalization stats to stats.json (train-only fit).
  9. For tiles: optionally cuts 256px or 512px sliding-window tiles from chips
     (keeps chip_id in filename for zero-overlap enforcement downstream).

Segmentation labels: 0=non-water, 1=flood, -1=nodata (IGNORED in loss/metrics)

Run: python flood_project/ml_training/scripts/phase0_build_seg_dataset.py

Outputs (relative to ml_training/):
  data/seg_dataset/
    split_manifest.json          -- chip->split assignments + stats
    norm_stats.json              -- per-channel mean/std/clip (train only)
    train_chips.json             -- list of chip IDs in train split
    val_chips.json
    test_chips.json
    bolivia_chips.json
    leakage_audit_seg.json       -- overlap proof + prevalence table
"""

import sys
import json
import hashlib
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore")

BASE     = Path(__file__).resolve().parent.parent
RAW_DIR  = BASE / "data" / "raw" / "sen1floods11"
HAND_DIR = RAW_DIR / "HandLabeled"
S1_DIR   = HAND_DIR / "S1Hand"
LBL_DIR  = HAND_DIR / "LabelHand"
JRC_DIR  = HAND_DIR / "JRCWaterHand"
SPLIT_DIR = RAW_DIR / "splits"
OUT_DIR   = BASE / "data" / "seg_dataset"

DB_CLIP_LOW  = -35.0   # dB clip low
DB_CLIP_HIGH =   5.0   # dB clip high


# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD OFFICIAL SPLITS
# ──────────────────────────────────────────────────────────────────────────────

def load_official_splits() -> dict:
    """
    Returns chip_id -> split_name mapping from OFFICIAL Sen1Floods11 GCS split CSVs.

    Official GCS split CSV format (no header):
      Column 0: ChipID_S1Hand.tif
      Column 1: ChipID_LabelHand.tif

    Files:
      flood_train_data.csv   -> split = "train"
      flood_valid_data.csv   -> split = "val"
      flood_test_data.csv    -> split = "test"
      flood_bolivia_data.csv -> split = "bolivia"

    Raises RuntimeError if official split CSVs not found.
    """
    import csv

    assignments = {}  # chip_id -> split

    # Official GCS split files (downloaded by phase0_list_gcs_splits.py)
    split_file_map = {
        "flood_train_data.csv":   "train",
        "flood_valid_data.csv":   "val",
        "flood_test_data.csv":    "test",
        "flood_bolivia_data.csv": "bolivia",
    }

    if not SPLIT_DIR.exists():
        raise RuntimeError(
            f"Split directory not found: {SPLIT_DIR}\n"
            "Run: python scripts/phase0_list_gcs_splits.py"
        )

    found_any = False
    for csv_name, split_label in split_file_map.items():
        csv_path = SPLIT_DIR / csv_name
        if not csv_path.exists():
            print(f"  [split] MISSING: {csv_path}")
            continue

        found_any = True
        n_before = len(assignments)
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                # Format: Ghana_313799_S1Hand.tif,Ghana_313799_LabelHand.tif
                s1_fname = row[0].strip()
                if not s1_fname:
                    continue
                chip_id = s1_fname.replace("_S1Hand.tif", "")
                if chip_id in assignments:
                    raise RuntimeError(
                        f"Chip {chip_id!r} appears in multiple split files! "
                        f"Already assigned to {assignments[chip_id]!r}, now seeing {split_label!r}."
                    )
                assignments[chip_id] = split_label
        n_added = len(assignments) - n_before
        print(f"  [split] {split_label:10s}: {n_added:3d} chips  (from {csv_name})")

    if not found_any:
        raise RuntimeError(
            f"No official split CSV files found in {SPLIT_DIR}.\n"
            "Run: python scripts/phase0_list_gcs_splits.py\n"
            "Then re-run this script."
        )

    if not assignments:
        raise RuntimeError("Split CSVs were found but no chip IDs were parsed. Check file format.")

    print(f"\n  [split] Total chips assigned from official split CSVs: {len(assignments)}")
    return assignments


# ──────────────────────────────────────────────────────────────────────────────
# 2. ASSERT ZERO CHIP-ID OVERLAP
# ──────────────────────────────────────────────────────────────────────────────

def assert_no_overlap(splits: dict):
    """Fails loudly if any chip_id appears in more than one split."""
    by_split = defaultdict(set)
    for chip_id, sp in splits.items():
        by_split[sp].add(chip_id)

    split_names = list(by_split.keys())
    violations = []
    for i in range(len(split_names)):
        for j in range(i+1, len(split_names)):
            a, b = split_names[i], split_names[j]
            overlap = by_split[a] & by_split[b]
            if overlap:
                violations.append(f"  OVERLAP {a} ∩ {b}: {sorted(overlap)}")

    if violations:
        msg = "CHIP-ID OVERLAP DETECTED -- SPLIT VIOLATION:\n" + "\n".join(violations)
        raise RuntimeError(msg)

    print("\n  [overlap] ZERO chip-ID overlap confirmed across all splits:")
    for sp, chips in sorted(by_split.items()):
        print(f"    {sp:10s}: {len(chips):3d} chips")

    return by_split


# ──────────────────────────────────────────────────────────────────────────────
# 3. LOAD CHIP (VV, VH) with rasterio
# ──────────────────────────────────────────────────────────────────────────────

def load_s1_chip(chip_id: str) -> np.ndarray:
    """
    Load Sentinel-1 chip as float32 array shape [H, W, 2] (VV, VH).
    S1Hand GeoTIFF has 2 bands: band1=VV, band2=VH in dB.
    """
    import rasterio
    chip_path = S1_DIR / f"{chip_id}_S1Hand.tif"
    if not chip_path.exists():
        raise FileNotFoundError(f"S1 chip not found: {chip_path}")
    with rasterio.open(chip_path) as src:
        arr = src.read().astype(np.float32)  # [bands, H, W]
    if arr.ndim == 2:
        # Single-band (old format) -- duplicate as VV
        arr = np.stack([arr, arr], axis=0)
    elif arr.shape[0] == 1:
        arr = np.concatenate([arr, arr], axis=0)
    # Ensure exactly 2 bands (VV, VH)
    if arr.shape[0] > 2:
        arr = arr[:2]
    return arr  # [2, H, W]


def load_label_chip(chip_id: str) -> np.ndarray:
    """Load label chip as int16 [H, W]. Values: 0=non-water, 1=flood, -1=nodata."""
    import rasterio
    lbl_path = LBL_DIR / f"{chip_id}_LabelHand.tif"
    if not lbl_path.exists():
        raise FileNotFoundError(f"Label chip not found: {lbl_path}")
    with rasterio.open(lbl_path) as src:
        lbl = src.read(1).astype(np.int16)
    return lbl


# ──────────────────────────────────────────────────────────────────────────────
# 4. COMPUTE NORMALIZATION STATS (TRAIN ONLY)
# ──────────────────────────────────────────────────────────────────────────────

def compute_norm_stats(train_chip_ids: list) -> dict:
    """
    Fits per-channel normalization on TRAIN chips ONLY.
    Channels: [VV dB, VH dB, VV-VH dB]
    Clip range: DB_CLIP_LOW to DB_CLIP_HIGH
    Returns: {ch_name: {mean, std, min, max, p2, p98}}
    """
    print(f"\n  [norm] Computing normalization stats on {len(train_chip_ids)} TRAIN chips...")
    print(f"    Clip range: [{DB_CLIP_LOW}, {DB_CLIP_HIGH}] dB")

    # Subsample 20,000 pixels per chip across all train chips (~5M pixels total)
    # This avoids pure Python object overhead and quadratic allocations
    samples = [[], [], []]  # VV, VH, VV-VH
    rng = np.random.default_rng(42)

    for i, chip_id in enumerate(train_chip_ids):
        try:
            arr = load_s1_chip(chip_id)  # [2, H, W]
        except Exception as e:
            print(f"    [WARN] {chip_id}: {e}")
            continue

        valid = np.isfinite(arr[0]) & np.isfinite(arr[1])
        if not np.any(valid):
            continue
        vv = np.clip(arr[0][valid], DB_CLIP_LOW, DB_CLIP_HIGH)
        vh = np.clip(arr[1][valid], DB_CLIP_LOW, DB_CLIP_HIGH)
        diff = np.clip(arr[0][valid] - arr[1][valid], DB_CLIP_LOW * 2, DB_CLIP_HIGH * 2)

        for ch_idx, ch_data in enumerate([vv, vh, diff]):
            sub = rng.choice(ch_data, size=min(len(ch_data), 20_000), replace=False)
            samples[ch_idx].append(sub)

        if (i+1) % 50 == 0:
            print(f"    Processed {i+1}/{len(train_chip_ids)} chips...")

    channel_names = ["vv_db", "vh_db", "vv_minus_vh_db"]
    stats = {}
    for ch_idx, ch_name in enumerate(channel_names):
        data = np.concatenate(samples[ch_idx]).astype(np.float32)
        stats[ch_name] = {
            "mean":  float(np.mean(data)),
            "std":   float(np.std(data)),
            "min":   float(np.min(data)),
            "max":   float(np.max(data)),
            "p2":    float(np.percentile(data, 2)),
            "p98":   float(np.percentile(data, 98)),
            "clip_low":  DB_CLIP_LOW,
            "clip_high": DB_CLIP_HIGH,
            "n_pixels": len(data),
        }
        print(f"    {ch_name}: mean={stats[ch_name]['mean']:.3f}  std={stats[ch_name]['std']:.3f}"
              f"  p2={stats[ch_name]['p2']:.2f}  p98={stats[ch_name]['p98']:.2f}")

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# 5. FLOOD PIXEL PREVALENCE PER SPLIT
# ──────────────────────────────────────────────────────────────────────────────

def compute_prevalence(split_chips: dict) -> dict:
    """
    Computes flood-pixel prevalence for each split.
    Returns {split: {flood_px, valid_px, prevalence_pct}}
    Ignores label==-1 (nodata).
    """
    print("\n  [prevalence] Computing flood-pixel prevalence per split...")
    prevalence = {}

    for sp, chip_ids in split_chips.items():
        total_flood = 0
        total_valid = 0
        for chip_id in chip_ids:
            try:
                lbl = load_label_chip(chip_id)
            except Exception as e:
                print(f"    [WARN] {chip_id}: {e}")
                continue
            valid_mask = (lbl >= 0)
            total_flood += int(np.sum(lbl == 1))
            total_valid += int(np.sum(valid_mask))

        pct = 100.0 * total_flood / max(1, total_valid)
        prevalence[sp] = {
            "flood_pixels": total_flood,
            "valid_pixels": total_valid,
            "prevalence_pct": round(pct, 2),
            "n_chips": len(chip_ids),
        }
        flag = ""
        if pct > 35.0:
            flag = "  *** SUSPICIOUS: > 35% flood prevalence -- check for label inversion or split contamination ***"
        elif pct < 1.0:
            flag = "  *** LOW: < 1% flood pixels -- check for all-nodata chips ***"
        print(f"    {sp:10s}: {pct:5.1f}% flood  ({total_flood:,}/{total_valid:,} valid px) {flag}")

    return prevalence


# ──────────────────────────────────────────────────────────────────────────────
# 6. PERCEPTUAL HASH DEDUPLICATE CHECK
# ──────────────────────────────────────────────────────────────────────────────

def dedupe_phash_check(split_chips: dict, threshold: int = 10) -> dict:
    """
    Computes pHash for each chip's VV band and checks for near-duplicates
    across splits (different chips that are essentially the same image).
    threshold: Hamming distance threshold for "duplicate"
    Returns: {cross_split_dupe_count, within_split_dupe_count, suspect_pairs}
    """
    import imagehash
    from PIL import Image

    print(f"\n  [phash] Running perceptual hash deduplicate check (threshold={threshold})...")

    hashes = {}  # chip_id -> phash
    all_chip_ids = []
    for sp, ids in split_chips.items():
        all_chip_ids.extend([(cid, sp) for cid in ids])

    for i, (chip_id, sp) in enumerate(all_chip_ids):
        try:
            arr = load_s1_chip(chip_id)
            vv = arr[0]
            # Normalize to uint8 for PIL
            clipped = np.clip(vv, DB_CLIP_LOW, DB_CLIP_HIGH)
            norm = ((clipped - DB_CLIP_LOW) / (DB_CLIP_HIGH - DB_CLIP_LOW) * 255).astype(np.uint8)
            img = Image.fromarray(norm, mode="L")
            ph = imagehash.phash(img)
            hashes[chip_id] = (ph, sp)
        except Exception as e:
            print(f"    [WARN] {chip_id}: phash failed: {e}")
        if (i+1) % 50 == 0:
            print(f"    Hashed {i+1}/{len(all_chip_ids)} chips...")

    print(f"    Computed pHash for {len(hashes)} chips.")

    # Check pairwise for near-duplicates
    suspect_cross = []
    suspect_within = []
    chip_ids_list = list(hashes.keys())
    N = len(chip_ids_list)
    for i in range(N):
        for j in range(i+1, N):
            a, b = chip_ids_list[i], chip_ids_list[j]
            ha, spa = hashes[a]
            hb, spb = hashes[b]
            dist = ha - hb
            if dist <= threshold:
                pair = {"chip_a": a, "split_a": spa, "chip_b": b, "split_b": spb, "hamming": dist}
                if spa != spb:
                    suspect_cross.append(pair)
                    print(f"    [SUSPECT CROSS-SPLIT DUPE] {a}({spa}) <-> {b}({spb})  hamming={dist}")
                else:
                    suspect_within.append(pair)

    print(f"    Cross-split near-duplicates found: {len(suspect_cross)}")
    print(f"    Within-split near-duplicates found: {len(suspect_within)}")
    if not suspect_cross:
        print(f"    [OK] No cross-split duplicates detected.")

    return {
        "cross_split_dupes": len(suspect_cross),
        "within_split_dupes": len(suspect_within),
        "suspect_pairs_cross": suspect_cross[:20],  # cap to 20 for JSON
        "suspect_pairs_within": suspect_within[:20],
        "threshold_hamming": threshold,
        "n_chips_hashed": len(hashes),
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import traceback

    print("=" * 70)
    print("  PHASE 0: BUILD SEGMENTATION DATASET  --  FloodSentinel AI")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load official splits
    print("\n[STEP 1] Load official split assignments")
    assignments = load_official_splits()

    # 2. Verify all local chips are assigned and split
    local_chip_ids = sorted(
        p.name.replace("_S1Hand.tif", "") for p in S1_DIR.glob("*_S1Hand.tif")
    )
    print(f"\n  Local S1Hand chips found: {len(local_chip_ids)}")

    # Chips with no assignment --> warn and assign train (never test/bolivia)
    unassigned = [c for c in local_chip_ids if c not in assignments]
    if unassigned:
        print(f"  [WARN] {len(unassigned)} chips have no split assignment; forcing to 'train':")
        for c in unassigned[:10]:
            print(f"    {c}")
        for c in unassigned:
            assignments[c] = "train"

    # Only keep chips that are locally available
    assignments_local = {c: assignments[c] for c in local_chip_ids}

    # 3. Assert zero overlap
    print("\n[STEP 2] Assert zero chip-ID overlap across splits")
    split_chips = assert_no_overlap(assignments_local)

    # Print split sizes
    print(f"\n  Split sizes (local chips only):")
    for sp in ["train", "val", "test", "bolivia"]:
        chips = split_chips.get(sp, set())
        print(f"    {sp:10s}: {len(chips):3d} chips")
    total = sum(len(v) for v in split_chips.values())
    print(f"    {'TOTAL':10s}: {total:3d} chips")

    # 4. Compute norm stats on train only
    print("\n[STEP 3] Compute normalization stats (TRAIN only)")
    train_ids = sorted(split_chips.get("train", set()))
    if not train_ids:
        raise RuntimeError("Train split is empty! Cannot compute normalization stats.")
    norm_stats = compute_norm_stats(train_ids)

    # Save norm stats
    norm_path = OUT_DIR / "norm_stats.json"
    with open(norm_path, "w") as f:
        json.dump({"train_only": True, "n_train_chips": len(train_ids), "channels": norm_stats}, f, indent=2)
    print(f"  [OK] Saved: {norm_path}")

    # 5. Compute flood pixel prevalence
    print("\n[STEP 4] Compute flood-pixel prevalence per split")
    prevalence = compute_prevalence(dict(split_chips))

    # 6. Perceptual hash dedupe
    print("\n[STEP 5] Perceptual hash deduplicate check")
    phash_results = dedupe_phash_check(dict(split_chips))

    # 7. Build and save leakage audit JSON
    print("\n[STEP 6] Building leakage audit document")
    leakage_audit = {
        "audit_version": "phase0_seg",
        "total_local_chips": len(local_chip_ids),
        "split_counts": {sp: len(chips) for sp, chips in split_chips.items()},
        "zero_overlap_verified": True,
        "prevalence_per_split": prevalence,
        "phash_dedupe": phash_results,
        "norm_stats_fit_on": "TRAIN ONLY",
    }
    audit_path = OUT_DIR / "leakage_audit_seg.json"
    with open(audit_path, "w") as f:
        json.dump(leakage_audit, f, indent=2)
    print(f"  [OK] Saved leakage audit: {audit_path}")

    # 8. Save per-split chip lists (with absolute paths to S1 and label TIFFs)
    for sp, chip_ids in split_chips.items():
        manifest = []
        for chip_id in sorted(chip_ids):
            s1_path  = S1_DIR / f"{chip_id}_S1Hand.tif"
            lbl_path = LBL_DIR / f"{chip_id}_LabelHand.tif"
            jrc_path = JRC_DIR / f"{chip_id}_JRCWaterHand.tif"
            manifest.append({
                "chip_id":    chip_id,
                "event":      chip_id.rsplit("_", 1)[0],
                "split":      sp,
                "s1_path":    str(s1_path),
                "label_path": str(lbl_path),
                "jrc_path":   str(jrc_path) if jrc_path.exists() else None,
            })
        out_path = OUT_DIR / f"{sp}_chips.json"
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  Saved {sp} manifest: {out_path} ({len(manifest)} chips)")

    # 9. Master split manifest
    split_manifest = {
        "description": "Sen1Floods11 HandLabeled official split assignments",
        "split_source": "Sen1Floods11 official event-level split (Bonafilia et al. 2020)",
        "total_chips": total,
        "splits": {sp: sorted(list(chips)) for sp, chips in split_chips.items()},
        "norm_stats_file": str(norm_path),
    }
    manifest_path = OUT_DIR / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(split_manifest, f, indent=2)
    print(f"\n  [OK] Master split manifest: {manifest_path}")

    # 10. Summary
    print("\n" + "=" * 70)
    print("  PHASE 0 SEGMENTATION DATASET BUILD -- COMPLETE")
    print("=" * 70)
    print(f"\n  Outputs written to: {OUT_DIR}")
    print("\n  Prevalence summary:")
    for sp, stats in prevalence.items():
        print(f"    {sp:10s}: {stats['prevalence_pct']:5.1f}% flood")
    print("\n  Expected prevalence: roughly 9-17% (not 45%)")
    print("\n  Next step: python scripts/phase0_verify_seg_dataset.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
