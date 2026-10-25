"""
seg_metrics.py  --  FloodSentinel AI  Segmentation Metrics
============================================================
Segmentation evaluation metrics for flood/no-flood binary segmentation.
All metrics IGNORE label == -1 (nodata) per Phase 0 requirement.

Metrics computed:
  - IoU(flood): Intersection over Union for flood class (label=1)
  - IoU(non-water): IoU for non-water class (label=0)
  - mIoU: mean of IoU over valid classes
  - F1 / Dice coefficient for flood class
  - Precision, Recall for flood class
  - PR-AUC (from raw logit/prob predictions)
  - Bootstrap 95% CI resampled by CHIP (not pixel)

Also computes flood-area fraction: secondary metric derived from flood pixel count.

Usage:
  from seg_metrics import SegmentationMetrics
  met = SegmentationMetrics()
  met.update(pred_logits, labels)   # per batch
  results = met.compute()           # final dict
"""

import numpy as np
from typing import Optional, List, Dict, Any


IGNORE_INDEX = -1


def _to_numpy(x) -> np.ndarray:
    """Coerce tensor or ndarray to numpy."""
    if hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    if not isinstance(x, np.ndarray):
        return np.array(x)
    return x


# ──────────────────────────────────────────────────────────────────────────────
# CORE METRIC ACCUMULATORS
# ──────────────────────────────────────────────────────────────────────────────

class SegmentationMetrics:
    """
    Pixel-wise binary segmentation metric accumulator.
    Labels: 0=non-water, 1=flood, -1=ignored (nodata).
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Clear all accumulated state."""
        # Per-chip accumulators for bootstrap CI
        self._chip_results: List[Dict[str, float]] = []
        # Aggregate TP/FP/FN/TN for global metrics
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._tn = 0
        # For PR-AUC: store all predictions and labels
        self._all_probs: List[np.ndarray]  = []
        self._all_labels: List[np.ndarray] = []
        # Flood area fraction tracking
        self._flood_area_fracs: List[float] = []

    def update(
        self,
        logits: np.ndarray,   # [N, 1, H, W] or [N, H, W] raw logits
        labels: np.ndarray,   # [N, H, W] int labels (0/1/-1)
        chip_ids: Optional[List[str]] = None,
    ):
        """
        Update metric state with a batch.
        logits: raw model output (before sigmoid). Shape [N, 1, H, W] or [N, H, W].
        labels: int array with values 0, 1, -1.
        """
        logits = _to_numpy(logits)
        labels = _to_numpy(labels).astype(np.int16)

        if logits.ndim == 4:
            logits = logits[:, 0]   # [N, H, W]
        if logits.ndim == 2:
            logits = logits[np.newaxis]  # [1, H, W]
            labels = labels[np.newaxis]

        probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))   # sigmoid
        preds = (probs >= self.threshold).astype(np.int16)          # [N, H, W]

        N = logits.shape[0]
        for i in range(N):
            pred_i  = preds[i]
            prob_i  = probs[i]
            label_i = labels[i]

            mask = label_i >= 0   # valid pixels
            if mask.sum() == 0:
                continue

            p = pred_i[mask]
            g = label_i[mask].astype(np.int16)
            pr = prob_i[mask]

            tp = int(((p == 1) & (g == 1)).sum())
            fp = int(((p == 1) & (g == 0)).sum())
            fn = int(((p == 0) & (g == 1)).sum())
            tn = int(((p == 0) & (g == 0)).sum())

            self._tp += tp
            self._fp += fp
            self._fn += fn
            self._tn += tn

            self._all_probs.append(pr.ravel())
            self._all_labels.append(g.ravel())

            # Per-chip metrics for bootstrap
            chip_metrics = _compute_from_counts(tp, fp, fn, tn)
            chip_id = chip_ids[i] if (chip_ids and i < len(chip_ids)) else f"chip_{len(self._chip_results)}"
            chip_metrics["chip_id"] = chip_id
            # Flood area fraction (predicted)
            total_valid = mask.sum()
            flood_frac_pred  = float((p == 1).sum()) / max(1, total_valid)
            flood_frac_true  = float((g == 1).sum()) / max(1, total_valid)
            chip_metrics["flood_area_frac_pred"] = flood_frac_pred
            chip_metrics["flood_area_frac_true"] = flood_frac_true
            self._chip_results.append(chip_metrics)
            self._flood_area_fracs.append(flood_frac_true)

    def compute(self, n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Compute final metrics + bootstrap 95% CI by chip.
        Returns dict with all metrics.
        """
        # Global (all pixels pooled)
        global_metrics = _compute_from_counts(self._tp, self._fp, self._fn, self._tn)

        # PR-AUC from pooled predictions
        if self._all_probs:
            all_p = np.concatenate(self._all_probs)
            all_g = np.concatenate(self._all_labels)
            pr_auc = _compute_pr_auc(all_p, all_g)
        else:
            pr_auc = float("nan")
        global_metrics["pr_auc"] = pr_auc

        # Bootstrap 95% CI (resampled by CHIP, not pixel)
        ci = _bootstrap_ci(self._chip_results, n_bootstrap=n_bootstrap)

        # Flood area fraction
        if self._flood_area_fracs:
            global_metrics["flood_area_frac_mean"] = float(np.mean(self._flood_area_fracs))
            global_metrics["flood_area_frac_std"]  = float(np.std(self._flood_area_fracs))

        # Suspicion check
        if global_metrics.get("iou_flood", 0) > 0.90:
            print(
                f"  *** SUSPICIOUS: IoU(flood)={global_metrics['iou_flood']:.4f} > 0.90. "
                "Stop and re-audit split/labels. ***"
            )

        return {
            "global": global_metrics,
            "per_chip": self._chip_results,
            "bootstrap_ci_by_chip": ci,
            "n_chips": len(self._chip_results),
            "threshold": self.threshold,
        }


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def _compute_from_counts(
    tp: int, fp: int, fn: int, tn: int
) -> Dict[str, float]:
    """Compute all metrics from TP/FP/FN/TN counts."""
    eps = 1e-7

    # IoU flood (class 1)
    iou_flood = tp / max(tp + fp + fn, eps)
    # IoU non-water (class 0)
    iou_bg    = tn / max(tn + fp + fn, eps)
    # mIoU
    miou = (iou_flood + iou_bg) / 2.0

    # Precision, Recall, F1
    precision = tp / max(tp + fp, eps)
    recall    = tp / max(tp + fn, eps)
    f1        = 2 * tp / max(2 * tp + fp + fn, eps)

    # Accuracy (pixels)
    total = tp + fp + fn + tn
    acc   = (tp + tn) / max(total, 1)

    return {
        "iou_flood":  float(iou_flood),
        "iou_bg":     float(iou_bg),
        "miou":       float(miou),
        "precision":  float(precision),
        "recall":     float(recall),
        "f1":         float(f1),
        "accuracy":   float(acc),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _compute_pr_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute PR-AUC (area under precision-recall curve)."""
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(labels, probs))
    except Exception:
        # Manual trapezoidal integration fallback
        thresholds = np.linspace(0, 1, 101)[::-1]
        precisions, recalls = [], []
        for t in thresholds:
            p = (probs >= t).astype(int)
            tp = int(((p == 1) & (labels == 1)).sum())
            fp = int(((p == 1) & (labels == 0)).sum())
            fn = int(((p == 0) & (labels == 1)).sum())
            prec = tp / max(tp + fp, 1e-7)
            rec  = tp / max(tp + fn, 1e-7)
            precisions.append(prec)
            recalls.append(rec)
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        return float(trapz_fn(precisions, recalls))


def _bootstrap_ci(
    chip_results: List[Dict[str, float]],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """
    Bootstrap 95% CI on per-chip metrics.
    Resamples chips (not pixels) with replacement.
    """
    if not chip_results:
        return {}

    metric_keys = ["iou_flood", "iou_bg", "miou", "precision", "recall", "f1", "pr_auc"]
    available   = [k for k in metric_keys if k in chip_results[0]]

    rng = np.random.default_rng(42)
    N   = len(chip_results)
    bootstrap_samples = {k: [] for k in available}

    for _ in range(n_bootstrap):
        idxs = rng.integers(0, N, size=N)
        for k in available:
            vals = [chip_results[i].get(k, float("nan")) for i in idxs]
            valid = [v for v in vals if not np.isnan(v)]
            if valid:
                bootstrap_samples[k].append(float(np.mean(valid)))

    alpha = 1.0 - ci_level
    ci = {}
    for k in available:
        if bootstrap_samples[k]:
            bs = sorted(bootstrap_samples[k])
            lo = np.percentile(bs, 100 * alpha / 2)
            hi = np.percentile(bs, 100 * (1 - alpha / 2))
            mean_val = np.mean([r.get(k, float("nan")) for r in chip_results
                                if not np.isnan(r.get(k, float("nan")))])
            ci[k] = {
                "mean":  float(mean_val),
                "ci95_lo": float(lo),
                "ci95_hi": float(hi),
                "ci95_width": float(hi - lo),
            }

    return ci


# ──────────────────────────────────────────────────────────────────────────────
# THRESHOLD CALIBRATION (on VAL only, then frozen)
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_threshold(
    val_logits: np.ndarray,   # [N, H, W]
    val_labels: np.ndarray,   # [N, H, W]
    search_range: tuple = (0.2, 0.8),
    n_steps: int = 30,
) -> float:
    """
    Search for optimal threshold on VAL set that maximizes IoU(flood).
    MUST be called on VAL only -- threshold is then frozen for test.
    Returns best threshold.
    """
    val_logits = _to_numpy(val_logits)
    val_labels = _to_numpy(val_labels).astype(np.int16)

    probs = 1.0 / (1.0 + np.exp(-val_logits.astype(np.float64)))
    best_t = 0.5
    best_iou = -1.0

    thresholds = np.linspace(search_range[0], search_range[1], n_steps)
    for t in thresholds:
        mask = val_labels >= 0
        preds = (probs >= t).astype(np.int16)
        tp = int(((preds == 1) & (val_labels == 1) & mask).sum())
        fp = int(((preds == 1) & (val_labels == 0) & mask).sum())
        fn = int(((preds == 0) & (val_labels == 1) & mask).sum())
        iou = tp / max(tp + fp + fn, 1e-7)
        if iou > best_iou:
            best_iou = iou
            best_t   = float(t)

    print(f"  [threshold] Best VAL threshold: {best_t:.3f}  IoU(flood)={best_iou:.4f}")
    return best_t
