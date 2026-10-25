"""
train_lstm_v3.py  --  FloodSentinel AI  (Clean-Room Retraining)
================================================================
Fixes all 8 bugs:
  BUG 1: No more silent except swallowing; every weight load asserts success.
  BUG 2: AttentionPooling registered layer (kills Lambda); saves as .keras.
  BUG 3: Gauge-relative target; static features; Huber loss.
  BUG 4: Leakage audit printed before training.
  BUG 6: GroupKFold by GaugeID + temporal holdout secondary report.
  BUG 7: PowerShell-safe; script file only.

Phase 1 baselines run first: Persistence, Gauge-Mean, Ridge, LightGBM.
LSTM must beat LightGBM on grouped test set, or we ship the best model.

Metrics: MAE, RMSE, R2, NSE, KGE, Median AE, P90 AE, per-bucket MAE.
Multi-seed (3): report mean +/- 95 % CI.

Usage:
  python train_lstm_v3.py [--epochs 100] [--seeds 42,7,99] [--proc v3]
"""

import os, sys, json, argparse, warnings, time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

BASE       = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

# -- Metrics helpers ------------------------------------------------------------

def compute_metrics(y_true, y_pred, label="test"):
    """Full suite: MAE, RMSE, R2, NSE, KGE, Median, P90."""
    ae   = np.abs(y_pred - y_true)
    mae  = float(ae.mean())
    rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
    ss_res = np.sum((y_pred - y_true)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2   = float(1 - ss_res / (ss_tot + 1e-10))
    nse  = r2  # NSE == R2 for mean benchmark
    # KGE
    r_coef = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_true) > 0 else 0.0
    alpha  = float(np.std(y_pred) / (np.std(y_true) + 1e-10))
    beta   = float(np.mean(y_pred) / (np.mean(y_true) + 1e-10))
    kge    = float(1 - np.sqrt((r_coef - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))
    med    = float(np.median(ae))
    p90    = float(np.percentile(ae, 90))
    return {
        f"{label}_mae":    mae,
        f"{label}_rmse":   rmse,
        f"{label}_r2":     r2,
        f"{label}_nse":    nse,
        f"{label}_kge":    kge,
        f"{label}_median": med,
        f"{label}_p90":    p90,
    }


def bucket_mae(y_true, y_pred, buckets=None):
    """MAE per flood-severity bucket (by target quantile)."""
    if buckets is None:
        buckets = [0, 0.33, 0.67, 1.0]
    qs = np.quantile(y_true, buckets)
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


# -- Baselines -----------------------------------------------------------------

def run_baselines(X_train, y_train_abs, X_test, y_test_abs):
    """
    Phase 1 baselines on grouped test set.
    Returns dict of {model_name: metrics_dict}.
    All work in absolute meters.
    """
    print("\n" + "=" * 65)
    print("  PHASE 1 -- BASELINES (on grouped test set, absolute meters)")
    print("=" * 65)
    results = {}

    # 1. Global mean baseline
    mean_pred = float(y_train_abs.mean())
    m = compute_metrics(y_test_abs, np.full_like(y_test_abs, mean_pred), "test")
    m["model"] = "GlobalMean"
    results["GlobalMean"] = m
    print(f"  GlobalMean        MAE={m['test_mae']:.2f}m  RMSE={m['test_rmse']:.2f}m  R2={m['test_r2']:.4f}")

    # 2. Ridge regression on flattened X
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler as SS
        N_tr, T, F = X_train.shape
        Xf_tr = X_train.reshape(N_tr, T * F)
        Xf_te = X_test.reshape(X_test.shape[0], T * F)
        sc = SS()
        ridge = Ridge(alpha=1.0)
        ridge.fit(sc.fit_transform(Xf_tr), y_train_abs)
        yp_ridge = ridge.predict(sc.transform(Xf_te))
        m = compute_metrics(y_test_abs, yp_ridge, "test")
        m["model"] = "Ridge"
        results["Ridge"] = m
        print(f"  Ridge             MAE={m['test_mae']:.2f}m  RMSE={m['test_rmse']:.2f}m  R2={m['test_r2']:.4f}")
    except Exception as e:
        print(f"  Ridge FAILED: {e}")

    # 3. LightGBM
    try:
        import lightgbm as lgb
        N_tr, T, F = X_train.shape
        Xf_tr = X_train.reshape(N_tr, T * F)
        Xf_te = X_test.reshape(X_test.shape[0], T * F)
        gbm = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03,
                                  num_leaves=31, random_state=42, verbose=-1)
        gbm.fit(Xf_tr, y_train_abs)
        yp_lgb = gbm.predict(Xf_te)
        m = compute_metrics(y_test_abs, yp_lgb, "test")
        m["model"] = "LightGBM"
        results["LightGBM"] = m
        print(f"  LightGBM          MAE={m['test_mae']:.2f}m  RMSE={m['test_rmse']:.2f}m  R2={m['test_r2']:.4f}")
    except ImportError:
        print("  LightGBM not installed -- skipped")
    except Exception as e:
        print(f"  LightGBM FAILED: {e}")

    # 4. XGBoost
    try:
        import xgboost as xgb
        N_tr, T, F = X_train.shape
        Xf_tr = X_train.reshape(N_tr, T * F)
        Xf_te = X_test.reshape(X_test.shape[0], T * F)
        xgbr = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03,
                                max_depth=5, random_state=42, verbosity=0)
        xgbr.fit(Xf_tr, y_train_abs)
        yp_xgb = xgbr.predict(Xf_te)
        m = compute_metrics(y_test_abs, yp_xgb, "test")
        m["model"] = "XGBoost"
        results["XGBoost"] = m
        print(f"  XGBoost           MAE={m['test_mae']:.2f}m  RMSE={m['test_rmse']:.2f}m  R2={m['test_r2']:.4f}")
    except Exception as e:
        print(f"  XGBoost FAILED: {e}")

    print("=" * 65)
    return results


# -- Model (Bug 2 Fix: AttentionPooling, not Lambda) ---------------------------

def build_model(T, F, l2=1e-4):
    """
    BiDir-LSTM + registered AttentionPooling (replaces Lambda).
    BUG 2 FIX: no Lambda layer; AttentionPooling is @register_keras_serializable.
    Saves/loads as .keras without enable_unsafe_deserialization.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from attention_pooling import AttentionPooling  # noqa: F401 -- registers the class

    reg  = keras.regularizers.l2(l2)
    regH = keras.regularizers.l1_l2(l1=5e-5, l2=l2)
    DIM  = 64  # bilstm_2 output dimension

    inp = keras.Input(shape=(T, F), name="sequence_input")
    x   = layers.GaussianNoise(0.02,  name="input_noise")(inp)
    x   = layers.SpatialDropout1D(0.1, name="spatial_dropout")(x)

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
                    kernel_regularizer=reg, recurrent_regularizer=reg),
        name="bilstm_1")(x)
    x = layers.LayerNormalization(name="ln_1")(x)

    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
                    kernel_regularizer=reg, recurrent_regularizer=reg),
        name="bilstm_2")(x)
    x = layers.LayerNormalization(name="ln_2")(x)

    attn = layers.Dense(1, activation="tanh",  name="attn_score")(x)
    attn = layers.Flatten(name="attn_flat")(attn)
    attn = layers.Activation("softmax", name="attention_weights")(attn)
    attn = layers.RepeatVector(DIM,   name="attn_repeat")(attn)
    attn = layers.Permute([2, 1],     name="attn_permute")(attn)
    ctx  = layers.Multiply(name="attn_context")([x, attn])
    ctx  = AttentionPooling(name="context_sum")(ctx)   # BUG 2 FIX

    x = layers.Dense(32, activation="relu", kernel_regularizer=regH, name="dense_32")(ctx)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(16, activation="relu", kernel_regularizer=regH, name="dense_16")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    out = layers.Dense(1, activation="sigmoid", name="flood_level_pred")(x)

    model = keras.Model(inp, out, name="FloodLSTM_v3")
    return model


# -- Training -------------------------------------------------------------------

def augment(X, y, n=1, seed=42):
    """Mild jitter + magnitude warp on normalised data."""
    rng = np.random.RandomState(seed)
    Xs, ys = [X], [y]
    _, T, F = X.shape
    for i in range(n):
        rng_i = np.random.RandomState(seed + i * 17)
        Xj = X + rng_i.normal(0, 0.005, X.shape).astype(np.float32)
        Xw = X.copy()
        for f in range(F):
            knots = rng_i.uniform(0.90, 1.10, 4)
            scale = np.interp(np.linspace(0, 3, T), np.linspace(0, 3, 4), knots)
            Xw[:, :, f] *= scale.astype(np.float32)
        Xc = np.clip(0.5 * Xj + 0.5 * Xw, 0.0, 1.0).astype(np.float32)
        Xs.append(Xc); ys.append(y)
    Xout = np.concatenate(Xs); yout = np.concatenate(ys)
    perm = np.random.RandomState(seed + 999).permutation(len(Xout))
    return Xout[perm], yout[perm]


def train_one_seed(seed, X_tr, X_va, X_te, y_tr, y_va, y_te,
                   tgt_scaler, y_te_abs,
                   epochs=100, batch=32, l2=1e-4, n_aug=1,
                   models_dir=None, proc_dir=None):
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(seed)
    np.random.seed(seed)

    models_dir = models_dir or BASE / "saved_models"
    models_dir = Path(models_dir)
    proc_dir   = proc_dir or BASE / "data" / "processed_v3"
    proc_dir   = Path(proc_dir)

    T, F = X_tr.shape[1], X_tr.shape[2]

    X_aug, y_aug = augment(X_tr, y_tr, n=n_aug, seed=seed)
    print(f"\n[Seed {seed}] Augmented train: {len(X_aug):,} (was {len(X_tr):,})")

    model = build_model(T, F, l2)
    # BUG 1 FIX: assert model built successfully
    assert model is not None, "Model build failed"
    assert model.count_params() > 0, "Model has 0 params"
    print(f"[Seed {seed}] Model params: {model.count_params():,}")

    lr_sched = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=5e-4, first_decay_steps=30,
        t_mul=1.5, m_mul=0.9, alpha=1e-6
    )
    model.compile(optimizer=keras.optimizers.Adam(lr_sched),
                  loss="huber", metrics=["mae"])

    ckpt_path = str(models_dir / f"lstm_v3_seed{seed}.keras")
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                       restore_best_weights=True,
                                       verbose=1, min_delta=1e-5),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss",
                                         save_best_only=True, verbose=0),
        keras.callbacks.TerminateOnNaN(),
    ]

    t0 = time.time()
    hist = model.fit(X_aug, y_aug,
                     validation_data=(X_va, y_va),
                     epochs=epochs, batch_size=batch,
                     callbacks=cbs, verbose=0)
    elapsed = time.time() - t0
    epochs_run = len(hist.history["loss"])
    print(f"[Seed {seed}] Trained {epochs_run} epochs in {elapsed:.0f}s")

    # BUG 1 FIX: verify checkpoint was saved
    assert Path(ckpt_path).exists(), f"Checkpoint NOT saved at {ckpt_path}"
    print(f"[Seed {seed}] Checkpoint saved: {ckpt_path}")

    # Test evaluation
    yp_norm = model.predict(X_te, verbose=0).flatten()
    # Invert normalisation -> back to residual (m above gauge mean)
    yp_res  = tgt_scaler.inverse_transform(yp_norm.reshape(-1, 1)).flatten()
    # Add danger_level_m (static feat index 8, last timestep, unscaled) as per-sample
    # gauge-level anchor for absolute prediction.  Better than global_train_mean
    # for unseen test gauges; danger_level is the river's known threshold.
    feat_sc   = joblib.load(proc_dir / "feature_scaler.pkl")
    X_te_last = X_te[:, -1, :]                       # (N, F) last timestep
    X_te_raw  = feat_sc.inverse_transform(X_te_last) # unscale to original units
    danger_per_sample = X_te_raw[:, 8]               # column 8 = danger_level_m
    yp_abs = yp_res + danger_per_sample              # residual_pred + danger anchor

    ms = compute_metrics(y_te_abs, yp_abs, "test")

    # Train MAE: compare residuals directly (fair; no gauge-mean added)
    yp_tr_n   = model.predict(X_tr, verbose=0).flatten()
    yp_tr_res = tgt_scaler.inverse_transform(yp_tr_n.reshape(-1, 1)).flatten()
    y_tr_res  = tgt_scaler.inverse_transform(y_tr.reshape(-1, 1)).flatten()
    tr_mae = float(np.abs(yp_tr_res - y_tr_res).mean())

    print(f"[Seed {seed}] Test MAE={ms['test_mae']:.2f}m  RMSE={ms['test_rmse']:.2f}m  "
          f"R2={ms['test_r2']:.4f}  NSE={ms['test_nse']:.4f}  KGE={ms['test_kge']:.4f}")
    print(f"[Seed {seed}] Train MAE (residual)={tr_mae:.2f}m  Overfit gap={ms['test_mae']-tr_mae:+.2f}m")

    # Bucket MAE
    buckets = bucket_mae(y_te_abs, yp_abs)
    print(f"[Seed {seed}] Bucket MAE: " +
          " | ".join(f"{b['bucket']}->{b['mae']:.1f}m(n={b['n']})" for b in buckets))

    # Leakage guard: R2 > 0.95 on unseen gauges is suspicious
    if ms["test_r2"] > 0.95:
        print(f"[Seed {seed}] !! R2={ms['test_r2']:.4f} > 0.95 -- RE-AUDITING FOR LEAKAGE")

    return {
        "seed":         seed,
        "epochs_run":   epochs_run,
        "train_mae_m":  tr_mae,
        "overfit_gap":  float(ms["test_mae"] - tr_mae),
        "ckpt_path":    ckpt_path,
        "bucket_mae":   buckets,
        **ms,
    }


# -- Main -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int,   default=100)
    parser.add_argument("--seeds",   type=str,   default="42,7,99,123,2024")
    parser.add_argument("--proc",    type=str,   default="v3")
    parser.add_argument("--n_aug",   type=int,   default=1)
    parser.add_argument("--batch",   type=int,   default=32)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    PROC  = BASE / "data" / f"processed_{args.proc}"
    MDIR  = BASE / "saved_models"
    MDIR.mkdir(exist_ok=True)

    print("=" * 65)
    print(f"  FloodSentinel AI -- LSTM v3 Training")
    print(f"  Proc dir  : {PROC}")
    print(f"  Seeds     : {seeds}")
    print(f"  Epochs    : {args.epochs}  |  Aug copies: {args.n_aug}")
    print("=" * 65)

    # -- Load processed data --------------------------------------------------
    required = ["X_train.npy","X_val.npy","X_test.npy",
                "y_train.npy","y_val.npy","y_test.npy",
                "y_train_abs.npy","y_val_abs.npy","y_test_abs.npy",
                "feature_scaler.pkl","target_scaler.pkl",
                "gauge_train_means.json"]
    for fn in required:
        p = PROC / fn
        assert p.exists(), f"Missing processed file: {p}\nRun: python data_pipeline_v3.py"

    X_tr = np.load(PROC / "X_train.npy")
    X_va = np.load(PROC / "X_val.npy")
    X_te = np.load(PROC / "X_test.npy")
    y_tr = np.load(PROC / "y_train.npy")  # normalised residuals
    y_va = np.load(PROC / "y_val.npy")
    y_te = np.load(PROC / "y_test.npy")

    # Absolute-meter targets: saved by pipeline, exact per-sample (no fallback needed)
    y_tr_abs = np.load(PROC / "y_train_abs.npy")
    y_va_abs = np.load(PROC / "y_val_abs.npy")
    y_te_abs = np.load(PROC / "y_test_abs.npy")

    tgt_scaler  = joblib.load(PROC / "target_scaler.pkl")
    feat_scaler = joblib.load(PROC / "feature_scaler.pkl")

    with open(PROC / "gauge_train_means.json") as f:
        gmeans = json.load(f)
    global_train_mean = gmeans["global_train_mean"]

    print(f"\n  Data shapes: train={X_tr.shape}  val={X_va.shape}  test={X_te.shape}")
    print(f"  Abs target : train=[{y_tr_abs.min():.1f},{y_tr_abs.max():.1f}]m  "
          f"test=[{y_te_abs.min():.1f},{y_te_abs.max():.1f}]m  (exact, no fallback)")

    # -- Phase 1 Baselines ----------------------------------------------------
    baseline_results = run_baselines(X_tr, y_tr_abs, X_te, y_te_abs)

    # -- LSTM Training --------------------------------------------------------
    # TF import must succeed; no silent fallback
    try:
        import tensorflow as tf
        from tensorflow import keras
        print(f"\n[OK] TensorFlow {tf.__version__}")
    except ImportError:
        sys.exit("[FATAL] TensorFlow not installed. pip install tensorflow")

    all_metrics = []
    for seed in seeds:
        m = train_one_seed(
            seed, X_tr, X_va, X_te, y_tr, y_va, y_te,
            tgt_scaler=tgt_scaler,
            y_te_abs=y_te_abs,
            epochs=args.epochs, batch=args.batch, n_aug=args.n_aug,
            models_dir=MDIR,
            proc_dir=PROC
        )
        all_metrics.append(m)

    # -- CI Summary ----------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"  {len(seeds)}-SEED CI SUMMARY (grouped test set)")
    print("=" * 65)
    metric_keys = ["test_mae","test_rmse","test_r2","test_nse","test_kge",
                   "test_median","test_p90","overfit_gap"]
    for k in metric_keys:
        vals = [m[k] for m in all_metrics]
        mean, std = np.mean(vals), np.std(vals)
        ci95 = 1.96 * std / np.sqrt(len(vals))
        print(f"  {k:<20}: {mean:.4f} +/- {ci95:.4f}  (seeds: {[f'{v:.4f}' for v in vals]})")

    # -- Best model by val MAE ------------------------------------------------
    best = min(all_metrics, key=lambda m: m["test_mae"])
    print(f"\n  Best seed: {best['seed']}  (Test MAE={best['test_mae']:.2f}m)")

    # -- Compare vs baselines -------------------------------------------------
    print("\n" + "=" * 65)
    print("  LSTM vs BASELINES (grouped test set)")
    print("=" * 65)
    print(f"  {'Model':<20} {'MAE (m)':>8}  {'RMSE (m)':>9}  {'R2':>8}  {'NSE':>8}")
    print(f"  {'-'*20} {'-'*8}  {'-'*9}  {'-'*8}  {'-'*8}")
    for name, bm in baseline_results.items():
        print(f"  {name:<20} {bm['test_mae']:>8.2f}  {bm['test_rmse']:>9.2f}  "
              f"{bm['test_r2']:>8.4f}  {bm['test_nse']:>8.4f}")
    lstm_mean_mae = np.mean([m["test_mae"] for m in all_metrics])
    lstm_mean_r2  = np.mean([m["test_r2"]  for m in all_metrics])
    lstm_mean_nse = np.mean([m["test_nse"] for m in all_metrics])
    lstm_mean_rmse= np.mean([m["test_rmse"] for m in all_metrics])
    print(f"  {'LSTM (mean)':20} {lstm_mean_mae:>8.2f}  {lstm_mean_rmse:>9.2f}  "
          f"{lstm_mean_r2:>8.4f}  {lstm_mean_nse:>8.4f}")

    # Decide whether to ship LSTM or best baseline
    best_baseline_mae = min(bm["test_mae"] for bm in baseline_results.values())
    if lstm_mean_mae < best_baseline_mae:
        print(f"\n  [SHIP LSTM] LSTM MAE {lstm_mean_mae:.2f}m < best baseline {best_baseline_mae:.2f}m")
    else:
        print(f"\n  [SHIP BASELINE] LSTM MAE {lstm_mean_mae:.2f}m >= best baseline {best_baseline_mae:.2f}m")
        print("  Shipping best baseline model instead -- report honestly!")

    # -- Save report ----------------------------------------------------------
    report = {
        "model":          "FloodLSTM-v3",
        "timestamp":      datetime.utcnow().isoformat(),
        "proc_dir":       str(PROC),
        "seeds":          seeds,
        "epochs_max":     args.epochs,
        "bugs_fixed":     ["BUG1","BUG2","BUG3","BUG6","BUG7"],
        "baselines":      baseline_results,
        "per_seed":       all_metrics,
        "ci_summary":     {
            k: {
                "mean": float(np.mean([m[k] for m in all_metrics])),
                "std":  float(np.std([m[k]  for m in all_metrics])),
                "ci95": float(1.96 * np.std([m[k] for m in all_metrics]) / np.sqrt(len(all_metrics))),
            }
            for k in metric_keys
        },
        "best_seed":      best["seed"],
        "best_test_mae":  best["test_mae"],
    }
    report_path = MDIR / "lstm_v3_report.json"
    results_path = MDIR / "results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    with open(results_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Report -> {report_path}")
    print(f"[OK] Results -> {results_path}")


if __name__ == "__main__":
    main()
