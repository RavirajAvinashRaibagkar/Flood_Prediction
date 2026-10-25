"""
eval_v3.py  --  FloodSentinel AI  (Clean-Room Evaluation)
==========================================================
BUG 1 FIX: No silent exception swallowing. Every weight load step is
            explicit and raises loudly on failure.
BUG 2 FIX: Uses AttentionPooling (registered layer) — no Lambda, no
            enable_unsafe_deserialization.
BUG 3 FIX: Reports in absolute meters (inverts residual + gauge mean).

Evaluates:
  - LSTM v3 (best seed checkpoint)
  - vs Phase 1 baselines
  - CNN classifier (status report; real metrics require Sen1Floods11)
  - Backend API health check

Usage:
  python eval_v3.py [--proc v3]
"""

import os, sys, json, argparse, warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    ae   = np.abs(y_pred - y_true)
    mae  = float(ae.mean())
    rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
    ss_r = np.sum((y_pred - y_true)**2)
    ss_t = np.sum((y_true - y_true.mean())**2)
    r2   = float(1 - ss_r / (ss_t + 1e-10))
    r_c  = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_true) > 0 else 0.0
    alpha= float(np.std(y_pred) / (np.std(y_true) + 1e-10))
    beta = float(np.mean(y_pred) / (np.mean(y_true) + 1e-10))
    kge  = float(1 - np.sqrt((r_c-1)**2 + (alpha-1)**2 + (beta-1)**2))
    return dict(mae=mae, rmse=rmse, r2=r2, nse=r2, kge=kge,
                median=float(np.median(ae)), p90=float(np.percentile(ae, 90)))


# ── LSTM model build (same arch as train_lstm_v3.py) ─────────────────────────
def build_lstm_model(T, F, l2=1e-4):
    """
    BUG 1 FIX: Raises on any failure — no silent except.
    BUG 2 FIX: Uses AttentionPooling (registered); no Lambda.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    # Importing registers the custom class in Keras serialization registry
    from attention_pooling import AttentionPooling  # noqa: F401

    reg  = keras.regularizers.l2(l2)
    regH = keras.regularizers.l1_l2(l1=5e-5, l2=l2)
    DIM  = 64

    inp = keras.Input(shape=(T, F), name="sequence_input")
    x   = layers.GaussianNoise(0.02,   name="input_noise")(inp)
    x   = layers.SpatialDropout1D(0.1, name="spatial_dropout")(x)
    x   = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
                        kernel_regularizer=reg, recurrent_regularizer=reg),
            name="bilstm_1")(x)
    x   = layers.LayerNormalization(name="ln_1")(x)
    x   = layers.Bidirectional(
            layers.LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
                        kernel_regularizer=reg, recurrent_regularizer=reg),
            name="bilstm_2")(x)
    x   = layers.LayerNormalization(name="ln_2")(x)

    attn = layers.Dense(1, activation="tanh",  name="attn_score")(x)
    attn = layers.Flatten(name="attn_flat")(attn)
    attn = layers.Activation("softmax",        name="attention_weights")(attn)
    attn = layers.RepeatVector(DIM,            name="attn_repeat")(attn)
    attn = layers.Permute([2, 1],              name="attn_permute")(attn)
    ctx  = layers.Multiply(name="attn_context")([x, attn])
    ctx  = AttentionPooling(name="context_sum")(ctx)   # BUG 2 FIX

    x    = layers.Dense(32, activation="relu", kernel_regularizer=regH, name="dense_32")(ctx)
    x    = layers.Dropout(0.3, name="dropout_1")(x)
    x    = layers.Dense(16, activation="relu", kernel_regularizer=regH, name="dense_16")(x)
    x    = layers.Dropout(0.2, name="dropout_2")(x)
    out  = layers.Dense(1, activation="sigmoid", name="flood_level_pred")(x)
    return keras.Model(inp, out, name="FloodLSTM_v3")


def load_lstm_model(T, F, models_dir):
    """
    BUG 1 FIX: Explicit load with clear error propagation.
    1. Try each v3 seed checkpoint via keras.models.load_model (no unsafe mode).
    2. Raise descriptively if nothing loads.
    """
    import tensorflow as tf
    from tensorflow import keras
    from attention_pooling import AttentionPooling  # registers class

    candidates = sorted(models_dir.glob("lstm_v3_seed*.keras"))
    if not candidates:
        raise FileNotFoundError(
            f"No lstm_v3_seed*.keras found in {models_dir}.\n"
            "Run: python train_lstm_v3.py  to produce the checkpoints."
        )

    last_error = None
    for ckpt in candidates:
        try:
            # Direct load — works because AttentionPooling is registered
            model = keras.models.load_model(
                str(ckpt),
                custom_objects={"AttentionPooling": AttentionPooling}
            )
            # Shape assertion (BUG 1 FIX)
            expected_in  = (None, T, F)
            actual_in    = tuple(model.input_shape)
            if actual_in != expected_in:
                raise ValueError(
                    f"Input shape mismatch: checkpoint has {actual_in}, expected {expected_in}."
                )
            print(f"  Loaded: {ckpt.name}  params={model.count_params():,}")
            return model, ckpt.name
        except Exception as e:
            last_error = e
            print(f"  [WARN] {ckpt.name} failed: {e}")
            continue

    # Nothing loaded
    raise RuntimeError(
        f"All LSTM checkpoints failed to load.\nLast error: {last_error}\n"
        "Re-run: python train_lstm_v3.py"
    )


# ── Main eval ─────────────────────────────────────────────────────────────────
def evaluate_lstm(proc_dir, models_dir):
    print("\n" + "=" * 65)
    print("  LSTM v3 -- EVALUATION (BUG 1+2+3 fixes applied)")
    print("=" * 65)

    X_te  = np.load(proc_dir / "X_test.npy")
    X_tr  = np.load(proc_dir / "X_train.npy")
    y_te  = np.load(proc_dir / "y_test.npy")
    y_tr  = np.load(proc_dir / "y_train.npy")
    tgt_s = joblib.load(proc_dir / "target_scaler.pkl")

    with open(proc_dir / "gauge_train_means.json") as f:
        gm = json.load(f)
    global_mean = gm["global_train_mean"]

    T, F = X_te.shape[1], X_te.shape[2]
    print(f"  Test shape : {X_te.shape}  |  Features: {F}")

    model, src = load_lstm_model(T, F, models_dir)

    feat_sc = joblib.load(proc_dir / "feature_scaler.pkl")
    X_te_last = X_te[:, -1, :]
    X_te_raw = feat_sc.inverse_transform(X_te_last)
    danger_per_sample = X_te_raw[:, 8]

    # Inference -> absolute meters
    yp_norm = model.predict(X_te, verbose=0).flatten()
    yp_res  = tgt_s.inverse_transform(yp_norm.reshape(-1, 1)).flatten()
    yp_abs  = yp_res + danger_per_sample

    if (proc_dir / "y_test_abs.npy").exists():
        yt_abs = np.load(proc_dir / "y_test_abs.npy")
    else:
        yt_res  = tgt_s.inverse_transform(y_te.reshape(-1, 1)).flatten()
        yt_abs  = yt_res + danger_per_sample

    ms = compute_metrics(yt_abs, yp_abs)

    # Naive baseline
    if (proc_dir / "y_train_abs.npy").exists():
        yt_tr_abs = np.load(proc_dir / "y_train_abs.npy")
        naive_mae = float(np.abs(yt_abs - yt_tr_abs.mean()).mean())
    else:
        naive_mae = float(np.abs(yt_abs - global_mean).mean())

    # Train MAE for overfit gap
    yp_tr_n  = model.predict(X_tr, verbose=0).flatten()
    yp_tr_r  = tgt_s.inverse_transform(yp_tr_n.reshape(-1, 1)).flatten()
    yt_tr_r  = tgt_s.inverse_transform(y_tr.reshape(-1, 1)).flatten()
    tr_mae   = float(np.abs(yp_tr_r - yt_tr_r).mean())

    print()
    print(f"  {'Metric':<30} {'Value':>15}")
    print(f"  {'-'*30} {'-'*15}")
    print(f"  {'Loaded from':<30} {src:>15}")
    print(f"  {'Test samples':<30} {len(y_te):>15,}")
    print(f"  {'Features':<30} {F:>15}")
    print(f"  {'MAE (m)':<30} {ms['mae']:>15.3f}")
    print(f"  {'Median AE (m)':<30} {ms['median']:>15.3f}")
    print(f"  {'P90 AE (m)':<30} {ms['p90']:>15.3f}")
    print(f"  {'RMSE (m)':<30} {ms['rmse']:>15.3f}")
    print(f"  {'R2':<30} {ms['r2']:>15.4f}")
    print(f"  {'NSE':<30} {ms['nse']:>15.4f}")
    print(f"  {'KGE':<30} {ms['kge']:>15.4f}")
    print(f"  {'Naive baseline MAE (m)':<30} {naive_mae:>15.3f}")
    print(f"  {'Beats baseline?':<30} {'YES' if ms['mae'] < naive_mae else 'NO':>15}")
    print(f"  {'Train MAE (m)':<30} {tr_mae:>15.3f}")
    print(f"  {'Overfit gap (test-train)':<30} {ms['mae']-tr_mae:>+15.3f}")

    # Leakage guard
    if ms["r2"] > 0.95:
        print("\n  [WARN] R2 > 0.95 on UNSEEN GAUGES -- STOP and re-audit leakage.")

    return ms, naive_mae, src


def evaluate_cnn(models_dir):
    print("\n" + "=" * 65)
    print("  CNN CLASSIFIER - Status")
    print("=" * 65)
    rp = models_dir / "cnn_phase_training_report.json"
    if rp.exists():
        with open(rp) as f:
            r = json.load(f)
        for ph in r.get("phases", []):
            note = " <- SYNTHETIC MEMORIZATION" if ph["best_val_acc"] >= 0.99 else ""
            print(f"  {ph['phase']}: val_acc={ph['best_val_acc']:.4f}{note}")
        print()
    print("  BUG 5: CNN trained on synthetic data -> 100% accuracy is memorization.")
    print("  Real metrics require Sen1Floods11 (~4 GB SAR data).")
    print("  Download: python scripts/download_sen1floods11.py")
    print("  Retrain : python train_cnn_v3.py  (after download)")


def check_backend():
    print("\n" + "=" * 65)
    print("  BACKEND API - Health")
    print("=" * 65)
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:8000/health", timeout=3) as r:
            print(f"  Status: RUNNING  ({r.getcode()})")
    except Exception as e:
        try:
            import urllib.request
            with urllib.request.urlopen("http://localhost:8000/", timeout=3) as r:
                print(f"  Status: RUNNING  ({r.getcode()})")
        except Exception:
            print(f"  Status: NOT RUNNING  ({type(e).__name__}: {e})")
            print("  Start : cd flood_project/backend && uvicorn main:app --port 8000")


# -- Entry ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc", default="v3")
    args = parser.parse_args()

    proc_dir   = BASE / "data" / f"processed_{args.proc}"
    models_dir = BASE / "saved_models"

    if not proc_dir.exists():
        sys.exit(f"[FATAL] Processed dir not found: {proc_dir}\nRun: python data_pipeline_v3.py")

    try:
        import tensorflow as tf
        from tensorflow import keras
        print(f"[OK] TensorFlow {tf.__version__}")
    except ImportError:
        sys.exit("[FATAL] TensorFlow not installed.")

    lstm_ms, naive_mae, src = evaluate_lstm(proc_dir, models_dir)
    evaluate_cnn(models_dir)
    check_backend()

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  LSTM v3 | MAE={lstm_ms['mae']:.2f}m  RMSE={lstm_ms['rmse']:.2f}m  "
          f"R2={lstm_ms['r2']:.3f}  NSE={lstm_ms['nse']:.3f}  KGE={lstm_ms['kge']:.3f}")
    print(f"  Naive   | MAE={naive_mae:.2f}m")
    print(f"  Beats   | {'YES [OK]' if lstm_ms['mae'] < naive_mae else 'NO [X]'}")
    print(f"  Loaded  | {src}")
    print()
    print("  Next: python train_lstm_v3.py  (if not already run)")
    print("        python scripts/download_sen1floods11.py  (CNN real data)")


if __name__ == "__main__":
    main()
