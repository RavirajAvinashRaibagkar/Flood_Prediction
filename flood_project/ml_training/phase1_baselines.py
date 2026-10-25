"""
phase1_baselines.py  --  FloodSentinel AI  Phase 1
===================================================
Evaluates classical baselines on Sen1Floods11 HandLabeled SEGMENTATION task:
  1. Majority-class baseline (always predict non-water)
  2. Otsu threshold on VV dB (per-chip adaptive)
  3. Fixed threshold on VV dB (-16 dB, per paper reference)
  4. Otsu threshold on VH dB (per-chip adaptive)

Reports for val, test, and bolivia SEPARATELY with bootstrap 95% CI by chip.
Metrics: IoU(flood), F1, precision, recall, PR-AUC, mIoU
All metrics IGNORE label == -1 (nodata).

Also prints flood-area prevalence as secondary metric derived from flood-pixel fraction.

Run: python flood_project/ml_training/phase1_baselines.py

Output:
  saved_models/baseline_results.json   -- full results JSON
  Prints formatted table per split
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

BASE    = Path(__file__).resolve().parent
SEG_DIR = BASE / "data" / "seg_dataset"
OUT_DIR = BASE / "saved_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IGNORE_INDEX = -1
DB_CLIP_LOW  = -35.0
DB_CLIP_HIGH =   5.0

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_split_manifest(split: str) -> list:
    path = SEG_DIR / f"{split}_chips.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {path}\n"
            "Run: python scripts/phase0_build_seg_dataset.py"
        )
    with open(path) as f:
        return json.load(f)


def load_chip(entry: dict):
    """
    Load S1 chip and label. Returns (vv_db, vh_db, label_int16).
    vv_db, vh_db: [H, W] float32 in dB, clipped to [-35, 5].
    label_int16:  [H, W] int16 (0/1/-1).
    """
    import rasterio
    with rasterio.open(entry["s1_path"]) as src:
        arr = src.read().astype(np.float32)  # [bands, H, W]
    if arr.shape[0] == 1:
        arr = np.concatenate([arr, arr], axis=0)
    vv = np.clip(arr[0], DB_CLIP_LOW, DB_CLIP_HIGH)
    vh = np.clip(arr[1], DB_CLIP_LOW, DB_CLIP_HIGH)

    with rasterio.open(entry["label_path"]) as src:
        lbl = src.read(1).astype(np.int16)

    return vv, vh, lbl


# ──────────────────────────────────────────────────────────────────────────────
# BASELINE PREDICTORS
# ──────────────────────────────────────────────────────────────────────────────

def predict_majority_class(vv: np.ndarray, vh: np.ndarray, lbl: np.ndarray) -> np.ndarray:
    """Always predict 0 (non-water). Flood class = minority class."""
    return np.zeros_like(vv, dtype=np.int16)


def otsu_threshold(values: np.ndarray) -> float:
    """Compute Otsu's threshold on 1D array of pixel values."""
    # Normalize to [0, 255] for histogram
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        return float(vmin)
    scaled = ((values - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    hist, bin_edges = np.histogram(scaled, bins=256)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return float(vmin)

    best_thresh = 0
    best_var = -1.0
    sum_total = np.dot(np.arange(256), hist)
    sum_bg = 0.0
    n_bg   = 0.0

    for t in range(256):
        n_bg   += hist[t]
        sum_bg += t * hist[t]
        n_fg   = total - n_bg
        if n_bg == 0 or n_fg == 0:
            continue
        mean_bg = sum_bg / n_bg
        mean_fg = (sum_total - sum_bg) / n_fg
        var_between = n_bg * n_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var   = var_between
            best_thresh = t

    # Map threshold back to dB range
    thresh_db = vmin + (best_thresh / 255.0) * (vmax - vmin)
    return float(thresh_db)


def predict_otsu_vv(vv: np.ndarray, vh: np.ndarray, lbl: np.ndarray) -> np.ndarray:
    """Otsu threshold on VV dB. Flood = pixels BELOW threshold (darker = wetter)."""
    valid_mask = lbl >= 0
    valid_vv = vv[valid_mask]
    if len(valid_vv) == 0:
        return np.zeros_like(vv, dtype=np.int16)
    thresh = otsu_threshold(valid_vv)
    pred = (vv <= thresh).astype(np.int16)
    pred[~valid_mask] = 0   # don't predict nodata regions
    return pred


def predict_otsu_vh(vv: np.ndarray, vh: np.ndarray, lbl: np.ndarray) -> np.ndarray:
    """Otsu threshold on VH dB. Flood = pixels BELOW threshold."""
    valid_mask = lbl >= 0
    valid_vh = vh[valid_mask]
    if len(valid_vh) == 0:
        return np.zeros_like(vh, dtype=np.int16)
    thresh = otsu_threshold(valid_vh)
    pred = (vh <= thresh).astype(np.int16)
    pred[~valid_mask] = 0
    return pred


def predict_fixed_thresh_vv(
    vv: np.ndarray, vh: np.ndarray, lbl: np.ndarray,
    thresh_db: float = -16.0,
) -> np.ndarray:
    """Fixed threshold on VV dB (reference value from flood SAR literature)."""
    valid_mask = lbl >= 0
    pred = (vv <= thresh_db).astype(np.int16)
    pred[~valid_mask] = 0
    return pred


def predict_fixed_thresh_vh(
    vv: np.ndarray, vh: np.ndarray, lbl: np.ndarray,
    thresh_db: float = -22.0,
) -> np.ndarray:
    """Fixed threshold on VH dB (empirical SAR flood threshold)."""
    valid_mask = lbl >= 0
    pred = (vh <= thresh_db).astype(np.int16)
    pred[~valid_mask] = 0
    return pred


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

def chip_metrics(pred: np.ndarray, lbl: np.ndarray) -> dict:
    """Compute binary segmentation metrics for a single chip, ignoring -1."""
    mask = lbl >= 0
    if mask.sum() == 0:
        return None

    p = pred[mask].astype(np.int16)
    g = lbl[mask].astype(np.int16)

    eps = 1e-7
    tp = int(((p == 1) & (g == 1)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    tn = int(((p == 0) & (g == 0)).sum())

    iou_flood = tp / max(tp + fp + fn, eps)
    iou_bg    = tn / max(tn + fp + fn, eps)
    miou      = (iou_flood + iou_bg) / 2.0
    precision = tp / max(tp + fp, eps)
    recall    = tp / max(tp + fn, eps)
    f1        = 2 * tp / max(2 * tp + fp + fn, eps)

    flood_frac_pred = float((p == 1).sum()) / max(mask.sum(), 1)
    flood_frac_true = float((g == 1).sum()) / max(mask.sum(), 1)

    return {
        "iou_flood": float(iou_flood),
        "iou_bg":    float(iou_bg),
        "miou":      float(miou),
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "flood_frac_pred": flood_frac_pred,
        "flood_frac_true": flood_frac_true,
    }


def pool_metrics(chip_results: list) -> dict:
    """Pool TP/FP/FN/TN across all chips for global metrics."""
    tp = sum(r["tp"] for r in chip_results)
    fp = sum(r["fp"] for r in chip_results)
    fn = sum(r["fn"] for r in chip_results)
    tn = sum(r["tn"] for r in chip_results)
    eps = 1e-7
    return {
        "iou_flood": tp / max(tp + fp + fn, eps),
        "iou_bg":    tn / max(tn + fp + fn, eps),
        "miou":      (tp / max(tp+fp+fn, eps) + tn / max(tn+fp+fn, eps)) / 2.0,
        "precision": tp / max(tp + fp, eps),
        "recall":    tp / max(tp + fn, eps),
        "f1":        2*tp / max(2*tp + fp + fn, eps),
        "pr_auc":    float("nan"),   # Otsu doesn't produce probabilities
        "flood_frac_pred_mean": float(np.mean([r["flood_frac_pred"] for r in chip_results])),
        "flood_frac_true_mean": float(np.mean([r["flood_frac_true"] for r in chip_results])),
    }


def bootstrap_ci(chip_results: list, n_bootstrap: int = 1000) -> dict:
    """Bootstrap 95% CI resampled by chip."""
    if not chip_results:
        return {}
    keys = ["iou_flood", "miou", "f1", "precision", "recall"]
    rng = np.random.default_rng(42)
    N   = len(chip_results)
    samples = {k: [] for k in keys}
    for _ in range(n_bootstrap):
        idxs = rng.integers(0, N, size=N)
        # Pool TP/FP/FN/TN for each bootstrap sample
        tp = sum(chip_results[i]["tp"] for i in idxs)
        fp = sum(chip_results[i]["fp"] for i in idxs)
        fn = sum(chip_results[i]["fn"] for i in idxs)
        tn = sum(chip_results[i]["tn"] for i in idxs)
        eps = 1e-7
        samples["iou_flood"].append(tp / max(tp+fp+fn, eps))
        samples["iou_bg"]   = samples.get("iou_bg", [])
        samples["iou_bg"].append(tn / max(tn+fp+fn, eps))
        samples["miou"].append((tp/max(tp+fp+fn,eps) + tn/max(tn+fp+fn,eps))/2)
        samples["precision"].append(tp / max(tp+fp, eps))
        samples["recall"].append(tp / max(tp+fn, eps))
        f1 = 2*tp / max(2*tp+fp+fn, eps)
        samples["f1"].append(f1)

    ci = {}
    for k, vals in samples.items():
        if vals:
            ci[k] = {
                "ci95_lo": float(np.percentile(vals, 2.5)),
                "ci95_hi": float(np.percentile(vals, 97.5)),
                "mean":    float(np.mean(vals)),
            }
    return ci


def print_results_table(name: str, global_m: dict, ci: dict, n_chips: int):
    """Print formatted results table."""
    print(f"\n  {'Metric':<18} {'Value':>8}  {'95% CI':>20}  {'n_chips':>7}")
    print(f"  {'-'*18} {'-'*8}  {'-'*20}  {'-'*7}")
    for k in ["iou_flood", "miou", "f1", "precision", "recall", "pr_auc"]:
        v = global_m.get(k, float("nan"))
        if k in ci:
            lo = ci[k]["ci95_lo"]
            hi = ci[k]["ci95_hi"]
            ci_str = f"[{lo:.4f}, {hi:.4f}]"
        else:
            ci_str = "N/A"
        print(f"  {k:<18} {v:>8.4f}  {ci_str:>20}  {n_chips:>7}")
    frac = global_m.get("flood_frac_true_mean", float("nan"))
    print(f"\n  Flood prevalence (true, mean per chip): {frac:.2%}")


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATE ONE BASELINE ON ONE SPLIT
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_baseline(predictor_fn, split_name: str, n_bootstrap: int = 1000) -> dict:
    """
    Evaluate a single baseline predictor on all chips in a split.
    Returns: {global_metrics, bootstrap_ci, per_chip}
    """
    manifest = load_split_manifest(split_name)
    chip_results = []

    for i, entry in enumerate(manifest):
        try:
            vv, vh, lbl = load_chip(entry)
        except Exception as e:
            print(f"    [WARN] {entry['chip_id']}: load failed: {e}")
            continue

        pred = predictor_fn(vv, vh, lbl)
        m = chip_metrics(pred, lbl)
        if m is not None:
            m["chip_id"] = entry["chip_id"]
            chip_results.append(m)

        if (i+1) % 30 == 0:
            print(f"    Processed {i+1}/{len(manifest)} chips...")

    if not chip_results:
        raise RuntimeError(f"No valid chips evaluated for split {split_name!r}!")

    global_m = pool_metrics(chip_results)
    ci = bootstrap_ci(chip_results, n_bootstrap=n_bootstrap)

    return {
        "split": split_name,
        "n_chips": len(chip_results),
        "global_metrics": global_m,
        "bootstrap_ci_by_chip": ci,
        "per_chip": chip_results,
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  PHASE 1: CLASSICAL BASELINES  --  FloodSentinel AI")
    print("  Task: Pixel-wise flood segmentation (ignore_index=-1)")
    print("=" * 70)

    # Verify seg dataset is built
    if not (SEG_DIR / "train_chips.json").exists():
        raise RuntimeError(
            f"Segmentation dataset not found: {SEG_DIR}\n"
            "Run: python scripts/phase0_build_seg_dataset.py"
        )

    baselines = {
        "majority_class": predict_majority_class,
        "otsu_vv":        predict_otsu_vv,
        "otsu_vh":        predict_otsu_vh,
        "fixed_vv_m16":   predict_fixed_thresh_vv,
        "fixed_vh_m22":   predict_fixed_thresh_vh,
    }

    splits_to_eval = ["val", "test", "bolivia"]

    all_results = {}
    for bname, bfn in baselines.items():
        print(f"\n{'='*70}")
        print(f"  BASELINE: {bname}")
        print(f"{'='*70}")
        all_results[bname] = {}

        for split in splits_to_eval:
            print(f"\n  -- Split: {split} --")
            try:
                res = evaluate_baseline(bfn, split)
                all_results[bname][split] = res
                print_results_table(
                    f"{bname}/{split}",
                    res["global_metrics"],
                    res["bootstrap_ci_by_chip"],
                    res["n_chips"],
                )
            except FileNotFoundError as e:
                print(f"  [SKIP] {split}: {e}")
            except Exception as e:
                print(f"  [ERROR] {split}: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()
                raise

    # Save results
    out_path = OUT_DIR / "baseline_results.json"
    # Remove per_chip arrays to keep JSON small (keep global + CI)
    compact = {}
    for bname, splits in all_results.items():
        compact[bname] = {}
        for split, res in splits.items():
            compact[bname][split] = {
                "split":              res["split"],
                "n_chips":            res["n_chips"],
                "global_metrics":     res["global_metrics"],
                "bootstrap_ci_by_chip": res["bootstrap_ci_by_chip"],
            }

    with open(out_path, "w") as f:
        json.dump(compact, f, indent=2)
    print(f"\n\n  [OK] Results saved: {out_path}")

    # Final summary table
    print("\n" + "=" * 70)
    print("  BASELINE SUMMARY: IoU(flood) on val / test / bolivia")
    print("=" * 70)
    print(f"  {'Baseline':<22} {'val':>8} {'test':>8} {'bolivia':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}")
    for bname in baselines:
        row_vals = []
        for split in splits_to_eval:
            v = compact.get(bname, {}).get(split, {}).get("global_metrics", {}).get("iou_flood", float("nan"))
            row_vals.append(f"{v:8.4f}" if not np.isnan(v) else "     N/A")
        print(f"  {bname:<22} {'  '.join(row_vals)}")
    print()
    print("  Expected range: IoU(flood) 0.10-0.50 for classical thresholds on S1.")
    print("  Values near 0 or 1.0 = suspicious, audit split or label.")
    print("=" * 70)


if __name__ == "__main__":
    main()
