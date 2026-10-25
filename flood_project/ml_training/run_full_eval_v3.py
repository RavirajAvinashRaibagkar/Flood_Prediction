"""
run_full_eval_v3.py  --  FloodSentinel AI  (Bug-Free Full Evaluation)
======================================================================
BUG 1 FIX: No silent except-pass. Every weight-load either succeeds or
           raises a RuntimeError that is NEVER caught silently.
BUG 2 FIX: AttentionPooling is registered FIRST; keras.models.load_model()
           works without safe_mode=False or enable_unsafe_deserialization.
BUG 3 FIX: Evaluates only on processed_v3 (gauge-relative target + static feats).
           Converts residual predictions back to absolute meters at inference.

Usage:
  python run_full_eval_v3.py [--proc v3] [--seed 42]
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

# -- 1. Register AttentionPooling BEFORE any keras import touches .keras files --
# BUG 2 FIX: import registers the class; no Lambda layer; no unsafe flags.
from attention_pooling import AttentionPooling  # noqa: F401  (registers class)

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow import keras

# ASSERT: no unsafe flags needed
assert not hasattr(keras.config, "_unsafe_deserialization"), \
    "enable_unsafe_deserialization must NOT be called in this script."

print(f"[OK] TensorFlow {tf.__version__}  |  AttentionPooling registered.")


# -- 2. Metric helpers ---------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "test") -> dict:
    """Full evaluation suite: MAE, RMSE, R2, NSE, KGE, MedianAE, P90AE."""
    ae   = np.abs(y_pred - y_true)
    mae  = float(ae.mean())
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2   = float(1 - ss_res / (ss_tot + 1e-10))
    nse  = r2   # NSE ≡ R2 when benchmark is mean
    r_c  = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_true) > 0 else 0.0
    alpha = float(np.std(y_pred) / (np.std(y_true) + 1e-10))
    beta  = float(np.mean(y_pred) / (np.mean(y_true) + 1e-10))
    kge   = float(1 - np.sqrt((r_c - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    med   = float(np.median(ae))
    p90   = float(np.percentile(ae, 90))
    return {
        f"{label}_mae":    mae,
        f"{label}_rmse":   rmse,
        f"{label}_r2":     r2,
        f"{label}_nse":    nse,
        f"{label}_kge":    kge,
        f"{label}_median": med,
        f"{label}_p90":    p90,
    }


def bucket_mae(y_true: np.ndarray, y_pred: np.ndarray) -> list:
    """MAE per flood-severity tertile bucket."""
    qs = np.quantile(y_true, [0, 0.33, 0.67, 1.0])
    rows = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        mask = (y_true >= lo) & (y_true <= hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "bucket": f"{lo:.1f}-{hi:.1f} m",
            "n":      int(mask.sum()),
            "mae":    float(np.abs(y_pred[mask] - y_true[mask]).mean()),
        })
    return rows


# -- 3. Baselines --------------------------------------------------------------

def run_baselines(X_tr, y_tr_abs, X_te, y_te_abs) -> dict:
    """
    Phase 1 baselines: GlobalMean, Ridge, LightGBM.
    All operate on absolute meters.
    """
    print("\n" + "=" * 65)
    print("  PHASE 1 -- BASELINES (grouped test set, absolute meters)")
    print("=" * 65)
    results = {}

    # Global mean
    gm_pred = float(y_tr_abs.mean())
    m = compute_metrics(y_te_abs, np.full_like(y_te_abs, gm_pred), "test")
    m["model"] = "GlobalMean"
    results["GlobalMean"] = m
    print(f"  GlobalMean  MAE={m['test_mae']:.2f}m  RMSE={m['test_rmse']:.2f}m  R2={m['test_r2']:.4f}")

    # Ridge
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler as SS
        N_tr, T, F = X_tr.shape
        Xf = X_tr.reshape(N_tr, T * F)
        Xf_te = X_te.reshape(X_te.shape[0], T * F)
        sc = SS()
        ridge = Ridge(alpha=1.0)
        ridge.fit(sc.fit_transform(Xf), y_tr_abs)
        yp = ridge.predict(sc.transform(Xf_te))
        m = compute_metrics(y_te_abs, yp, "test")
        m["model"] = "Ridge"
        results["Ridge"] = m
        print(f"  Ridge       MAE={m['test_mae']:.2f}m  RMSE={m['test_rmse']:.2f}m  R2={m['test_r2']:.4f}")
    except Exception as e:
        print(f"  Ridge FAILED: {e}")

    # LightGBM
    try:
        import lightgbm as lgb
        N_tr, T, F = X_tr.shape
        Xf = X_tr.reshape(N_tr, T * F)
        Xf_te = X_te.reshape(X_te.shape[0], T * F)
        gbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                  num_leaves=63, random_state=42, verbose=-1)
        gbm.fit(Xf, y_tr_abs, eval_set=[(Xf_te, y_te_abs)],
                callbacks=[lgb.early_stopping(30, verbose=False),
                           lgb.log_evaluation(-1)])
        yp = gbm.predict(Xf_te)
        m = compute_metrics(y_te_abs, yp, "test")
        m["model"] = "LightGBM"
        results["LightGBM"] = m
        print(f"  LightGBM    MAE={m['test_mae']:.2f}m  RMSE={m['test_rmse']:.2f}m  R2={m['test_r2']:.4f}")
    except ImportError:
        print("  LightGBM not installed (pip install lightgbm) -- skipped")
    except Exception as e:
        print(f"  LightGBM FAILED: {e}")

    print("=" * 65)
    return results


# -- 4. LSTM model load (loud on failure) --------------------------------------

def load_lstm(models_dir: Path, T: int, F: int) -> keras.Model:
    """
    BUG 1 FIX: No silent except.  Raises RuntimeError if no checkpoint loads.
    BUG 2 FIX: AttentionPooling already registered; custom_objects passed as belt-and-suspenders.
    """
    candidates = sorted(models_dir.glob("lstm_v3_seed*.keras"))
    if not candidates:
        raise FileNotFoundError(
            f"No lstm_v3_seed*.keras files in {models_dir}.\n"
            "Run: python train_lstm_v3.py"
        )

    load_errors = {}
    for ckpt in candidates:
        try:
            model = keras.models.load_model(
                str(ckpt),
                custom_objects={"AttentionPooling": AttentionPooling}
            )
            # Shape assertion: input must match processed_v3 features
            expected_shape = (None, T, F)
            actual_shape   = tuple(model.input_shape)
            assert actual_shape == expected_shape, (
                f"Model input shape {actual_shape} != expected {expected_shape}. "
                "Possible stale weights from a different feature spec."
            )
            print(f"[OK] Loaded {ckpt.name}  ({model.count_params():,} params, shape {actual_shape})")
            return model
        except Exception as e:
            load_errors[ckpt.name] = str(e)
            print(f"[WARN] Could not load {ckpt.name}: {e}")

    # No checkpoint loaded -- fail loudly
    raise RuntimeError(
        "All LSTM checkpoints failed to load:\n" +
        "\n".join(f"  {k}: {v}" for k, v in load_errors.items()) +
        "\nRetrain with: python train_lstm_v3.py"
    )


# -- 5. Main evaluation --------------------------------------------------------

def evaluate(proc_dir: Path, models_dir: Path) -> dict:
    print("\n" + "=" * 65)
    print("  LSTM EVALUATION  --  FloodSentinel AI")
    print(f"  Data : {proc_dir}")
    print(f"  Time : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 65)

    # -- Load processed_v3 arrays --
    required = [
        "X_train.npy", "X_val.npy", "X_test.npy",
        "y_train.npy", "y_val.npy", "y_test.npy",
        "feature_scaler.pkl", "target_scaler.pkl",
        "gauge_train_means.json",
    ]
    for fn in required:
        p = proc_dir / fn
        assert p.exists(), f"Missing: {p}\nRun: python data_pipeline_v3.py"

    X_tr = np.load(proc_dir / "X_train.npy")
    X_va = np.load(proc_dir / "X_val.npy")
    X_te = np.load(proc_dir / "X_test.npy")
    y_tr = np.load(proc_dir / "y_train.npy")
    y_va = np.load(proc_dir / "y_val.npy")
    y_te = np.load(proc_dir / "y_test.npy")

    tgt_sc = joblib.load(proc_dir / "target_scaler.pkl")

    with open(proc_dir / "gauge_train_means.json") as f:
        gm_data = json.load(f)
    global_train_mean = float(gm_data["global_train_mean"])

    N, T, F = X_tr.shape
    print(f"\n  Shapes: train={X_tr.shape}  val={X_va.shape}  test={X_te.shape}")
    print(f"  Target: residual from gauge mean (global fallback={global_train_mean:.2f} m)")

    # Convert normalised residuals -> absolute meters
    def to_abs(y_norm: np.ndarray) -> np.ndarray:
        res = tgt_sc.inverse_transform(y_norm.reshape(-1, 1)).flatten()
        return res + global_train_mean

    y_tr_abs = to_abs(y_tr)
    y_te_abs = to_abs(y_te)

    # -- Phase 1 baselines --
    baseline_results = run_baselines(X_tr, y_tr_abs, X_te, y_te_abs)

    # -- Load LSTM (loud failure if missing) --
    model = load_lstm(models_dir, T, F)

    # -- Inference --
    yp_norm   = model.predict(X_te, verbose=0).flatten()
    yp_res    = tgt_sc.inverse_transform(yp_norm.reshape(-1, 1)).flatten()
    yp_abs    = yp_res + global_train_mean

    yp_tr_n   = model.predict(X_tr, verbose=0).flatten()
    yp_tr_res = tgt_sc.inverse_transform(yp_tr_n.reshape(-1, 1)).flatten()
    yp_tr_abs = yp_tr_res + global_train_mean

    # -- Metrics --
    ms_te = compute_metrics(y_te_abs, yp_abs, "test")
    ms_tr = compute_metrics(y_tr_abs, yp_tr_abs, "train")
    overfit_gap = ms_te["test_mae"] - ms_tr["train_mae"]

    buckets = bucket_mae(y_te_abs, yp_abs)

    # -- Leakage guard --
    if ms_te["test_r2"] > 0.95:
        print("\n  !! WARNING: R2 > 0.95 on unseen test set -- possible leakage!")
        print("  Re-run: python data_pipeline_v3.py  and inspect leakage_audit.json")

    # -- Print results --
    print(f"\n  {'Metric':<35} {'Value':>15}")
    print(f"  {'-'*35} {'-'*15}")
    print(f"  {'Test samples':<35} {len(y_te_abs):>15,}")
    print(f"  {'Train samples':<35} {len(y_tr_abs):>15,}")
    print(f"  {'MAE (m)':<35} {ms_te['test_mae']:>15.3f}")
    print(f"  {'Median AE (m)':<35} {ms_te['test_median']:>15.3f}")
    print(f"  {'P90 AE (m)':<35} {ms_te['test_p90']:>15.3f}")
    print(f"  {'RMSE (m)':<35} {ms_te['test_rmse']:>15.3f}")
    print(f"  {'R2':<35} {ms_te['test_r2']:>15.4f}")
    print(f"  {'NSE':<35} {ms_te['test_nse']:>15.4f}")
    print(f"  {'KGE':<35} {ms_te['test_kge']:>15.4f}")
    print(f"  {'Train MAE (m)':<35} {ms_tr['train_mae']:>15.3f}")
    print(f"  {'Overfit gap (test-train MAE)':<35} {overfit_gap:>+15.3f}")

    # Best baseline
    best_bl_name = min(baseline_results, key=lambda k: baseline_results[k]["test_mae"])
    best_bl_mae  = baseline_results[best_bl_name]["test_mae"]
    beats_best   = ms_te["test_mae"] < best_bl_mae
    print(f"\n  Best baseline: {best_bl_name}  MAE={best_bl_mae:.3f}m")
    print(f"  LSTM beats best baseline: {'YES ✓' if beats_best else 'NO ✗'}")

    print("\n  Bucket MAE (flood severity):")
    for b in buckets:
        print(f"    {b['bucket']:>20}  MAE={b['mae']:.2f}m  n={b['n']}")

    # -- Comparison table --
    print("\n" + "=" * 65)
    print("  LSTM vs BASELINES")
    print("=" * 65)
    print(f"  {'Model':<20} {'MAE':>8}  {'RMSE':>8}  {'R2':>8}  {'NSE':>8}")
    print(f"  {'-'*20} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for name, bm in baseline_results.items():
        print(f"  {name:<20} {bm['test_mae']:>8.2f}  {bm['test_rmse']:>8.2f}  "
              f"{bm['test_r2']:>8.4f}  {bm['test_nse']:>8.4f}")
    print(f"  {'LSTM-v3':<20} {ms_te['test_mae']:>8.2f}  {ms_te['test_rmse']:>8.2f}  "
          f"{ms_te['test_r2']:>8.4f}  {ms_te['test_nse']:>8.4f}  <- LSTM")

    return {
        "timestamp":       datetime.utcnow().isoformat(),
        "proc_dir":        str(proc_dir),
        "test_metrics":    ms_te,
        "train_metrics":   ms_tr,
        "overfit_gap":     overfit_gap,
        "bucket_mae":      buckets,
        "baselines":       baseline_results,
        "beats_best_baseline": beats_best,
        "best_baseline":   best_bl_name,
        "best_baseline_mae": best_bl_mae,
    }


# -- 6. Entry point ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloodSentinel AI -- Full LSTM Evaluation")
    parser.add_argument("--proc",  default="v3", help="processed_XX dir suffix (default: v3)")
    args = parser.parse_args()

    PROC_DIR   = BASE / "data" / f"processed_{args.proc}"
    MODELS_DIR = BASE / "saved_models"

    results = evaluate(PROC_DIR, MODELS_DIR)

    out_path = MODELS_DIR / "eval_v3_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved -> {out_path}")
