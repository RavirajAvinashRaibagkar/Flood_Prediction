"""
run_full_eval.py -- FloodSentinel AI Full Evaluation
=====================================================
Strategy: Rebuild LSTM from architecture, copy weights layer-by-layer.
The saved .keras models use a Lambda(tf.reduce_sum) whose output_shape
cannot be inferred by Keras 3 during model loading, so we must rebuild
the architecture (with output_shape specified) and transfer weights.
"""

import os, sys, json, warnings
import numpy as np
import joblib

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

BASE       = os.path.dirname(os.path.abspath(__file__))
PROC_V2    = os.path.join(BASE, "data", "processed_v2")
MODELS_DIR = os.path.join(BASE, "saved_models")
sys.path.insert(0, os.path.join(BASE, "..", "backend"))

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow import keras
from tensorflow.keras import layers
keras.config.enable_unsafe_deserialization()

# ─── Architecture ────────────────────────────────────────────────────────────

def build_lstm_model(T=10, F=6, l2=1e-4):
    """Rebuild BiDir-LSTM+Attention exactly as in train_lstm_v2.py"""
    reg = keras.regularizers.l2(l2)
    rh  = keras.regularizers.l1_l2(l1=5e-5, l2=l2)
    DIM = 64  # bilstm_2 output per direction

    inp = keras.Input(shape=(T, F), name="sequence_input")
    x = layers.GaussianNoise(0.02, name="input_noise")(inp)
    x = layers.SpatialDropout1D(0.1, name="spatial_dropout")(x)

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

    attn = layers.Dense(1, activation="tanh", name="attn_score")(x)
    attn = layers.Flatten(name="attn_flat")(attn)
    attn = layers.Activation("softmax", name="attention_weights")(attn)
    attn = layers.RepeatVector(DIM, name="attn_repeat")(attn)
    attn = layers.Permute([2, 1], name="attn_permute")(attn)
    ctx  = layers.Multiply(name="attn_context")([x, attn])
    # Lambda WITH output_shape so Keras 3 can infer it
    ctx  = layers.Lambda(
        lambda v: tf.reduce_sum(v, axis=1),
        output_shape=(DIM,),
        name="context_sum"
    )(ctx)

    x = layers.Dense(32, activation="relu", kernel_regularizer=rh, name="dense_32")(ctx)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(16, activation="relu", kernel_regularizer=rh, name="dense_16")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    out = layers.Dense(1, activation="sigmoid", name="flood_level_pred")(x)
    return keras.Model(inp, out, name="FloodLSTM_v2")


# ─── Weight Transfer ─────────────────────────────────────────────────────────

def transfer_weights(src_model, dst_model):
    """Transfer weights from src to dst by layer name."""
    transferred = 0
    for dst_layer in dst_model.layers:
        try:
            src_layer = src_model.get_layer(dst_layer.name)
            w = src_layer.get_weights()
            if w:
                dst_layer.set_weights(w)
                transferred += 1
        except (ValueError, Exception):
            pass
    return transferred


# ─── Load saved LSTM .h5 (old format, more compatible) ───────────────────────

def load_lstm_weights_h5(model):
    """Try the .h5 file which may serialize differently."""
    h5_path = os.path.join(MODELS_DIR, "lstm_flood_forecaster.h5")
    if not os.path.exists(h5_path):
        return None
    try:
        # Load as TF SavedModel weights
        model.load_weights(h5_path)
        return "lstm_flood_forecaster.h5 (direct weights)"
    except Exception as e:
        # Try loading as full model and transferring
        try:
            src = tf.keras.models.load_model(h5_path, compile=False)
            n = transfer_weights(src, model)
            return f"lstm_flood_forecaster.h5 ({n} layers transferred)"
        except Exception as e2:
            return None


# ─── LSTM Evaluation ─────────────────────────────────────────────────────────

def evaluate_lstm():
    print()
    print("=" * 62)
    print("  LSTM FLOOD FORECASTER -- Live Evaluation")
    print("=" * 62)

    X_train = np.load(os.path.join(PROC_V2, "X_train.npy"))
    X_val   = np.load(os.path.join(PROC_V2, "X_val.npy"))
    X_test  = np.load(os.path.join(PROC_V2, "X_test.npy"))
    y_train = np.load(os.path.join(PROC_V2, "y_train.npy"))
    y_val   = np.load(os.path.join(PROC_V2, "y_val.npy"))
    y_test  = np.load(os.path.join(PROC_V2, "y_test.npy"))
    tgt_sc  = joblib.load(os.path.join(PROC_V2, "target_scaler.pkl"))

    N, T, F = X_train.shape
    print(f"  Data: train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")

    model = build_lstm_model(T, F)
    loaded_from = None

    # Strategy 1: Try .keras files via full model load + weight transfer
    for name in ["lstm_v2_seed42.keras", "lstm_best.keras", "lstm_flood_real.keras"]:
        p = os.path.join(MODELS_DIR, name)
        if not os.path.exists(p):
            continue
        try:
            src = keras.models.load_model(p, safe_mode=False)
            n = transfer_weights(src, model)
            loaded_from = f"{name} ({n} layers by name)"
            print(f"  Loaded: {loaded_from}")
            break
        except Exception as e:
            pass  # will try h5 fallback

    # Strategy 2: .h5 direct weight load
    if loaded_from is None:
        result = load_lstm_weights_h5(model)
        if result:
            loaded_from = result
            print(f"  Loaded: {loaded_from}")

    if loaded_from is None:
        print("  [WARN] Could not load saved weights.")
        print("         Re-training now with 30 epochs for evaluation...")
        model.compile(optimizer=keras.optimizers.Adam(5e-4), loss="huber", metrics=["mae"])
        cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                            restore_best_weights=True, verbose=0)]
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=30, batch_size=32, callbacks=cb, verbose=1)
        loaded_from = "freshly trained (30 epochs)"

    print(f"  Parameters: {model.count_params():,}")
    print(f"  Architecture: BiDir-LSTM(64,32) + Attention + Dense(32,16,1)")

    # Inference on test set
    yp_n = model.predict(X_test, verbose=0).flatten()
    yp_m = tgt_sc.inverse_transform(yp_n.reshape(-1, 1)).flatten()
    yt_m = tgt_sc.inverse_transform(y_test.reshape(-1, 1)).flatten()

    ae   = np.abs(yp_m - yt_m)
    mae  = float(ae.mean())
    rmse = float(np.sqrt(np.mean((yp_m - yt_m) ** 2)))
    r2   = float(1 - np.sum((yp_m - yt_m) ** 2) / np.sum((yt_m - yt_m.mean()) ** 2))
    med  = float(np.median(ae))
    p90  = float(np.percentile(ae, 90))

    mp = float(tgt_sc.inverse_transform([[y_train.mean()]])[0][0])
    bl = float(np.abs(yt_m - mp).mean())

    tr_p  = model.predict(X_train, verbose=0).flatten()
    tr_m  = tgt_sc.inverse_transform(tr_p.reshape(-1, 1)).flatten()
    yt_tr = tgt_sc.inverse_transform(y_train.reshape(-1, 1)).flatten()
    tr_mae = float(np.abs(tr_m - yt_tr).mean())
    overfit = mae - tr_mae

    print()
    print(f"  {'Metric':<35} {'Value':>15}")
    print(f"  {'-'*35} {'-'*15}")
    print(f"  {'Test samples':<35} {len(y_test):>15,}")
    print(f"  {'Train samples':<35} {len(y_train):>15,}")
    print(f"  {'MAE (meters)':<35} {mae:>15.2f}")
    print(f"  {'Median Error (m)':<35} {med:>15.2f}")
    print(f"  {'P90 Error (m)':<35} {p90:>15.2f}")
    print(f"  {'RMSE (meters)':<35} {rmse:>15.2f}")
    print(f"  {'R2 Score':<35} {r2:>15.4f}")
    print(f"  {'Naive Baseline MAE (m)':<35} {bl:>15.2f}")
    beats = mae < bl
    print(f"  {'Beats Naive Baseline?':<35} {'YES' if beats else 'NO':>15}")
    print(f"  {'Train MAE (m)':<35} {tr_mae:>15.2f}")
    print(f"  {'Overfit Gap (test - train MAE)':<35} {overfit:>+15.2f}")
    print(f"  {'Model Parameters':<35} {model.count_params():>15,}")

    if mae < bl * 0.8:
        verdict = "STRONG generalization -- well below naive baseline"
    elif mae < bl:
        verdict = "Marginal -- beats baseline, needs improvement"
    else:
        verdict = "FAILS to beat naive baseline"
    print(f"\n  Verdict: {verdict}")

    return dict(mae_m=mae, rmse_m=rmse, r2=r2, med_m=med, p90_m=p90,
                baseline_m=bl, beats=beats, overfit=overfit,
                loaded_from=loaded_from)


# ─── CNN Evaluation ───────────────────────────────────────────────────────────

def evaluate_cnn():
    print()
    print("=" * 62)
    print("  CNN FLOOD CLASSIFIER -- Status & Metrics")
    print("=" * 62)

    rp = os.path.join(MODELS_DIR, "cnn_phase_training_report.json")
    if os.path.exists(rp):
        with open(rp) as f:
            r = json.load(f)
        print(f"  Model    : {r.get('model', 'N/A')}")
        print(f"  Strategy : {r.get('strategy', 'N/A')}")
        print(f"  Train samples: {r.get('train_samples')}  Val: {r.get('val_samples')}")
        print()
        print(f"  {'Phase':<40} {'Val Acc':>9}  {'Val Loss':>12}")
        print(f"  {'-'*40} {'-'*9}  {'-'*12}")
        for ph in r.get("phases", []):
            print(f"  {ph['phase']:<40} {ph['best_val_acc']:>9.4f}  {ph['best_val_loss']:>12.2e}")
        print()
        print(f"  Final Val Accuracy : {r['final_val_acc']:.4f}")
        print(f"  Final Val Loss     : {r['final_val_loss']:.2e}")
        print()
        print("  NOTE: 100% accuracy = synthetic data memorization (BUG4).")
        print("  To get real-world metrics run: python train_cnn_v2.py --data real")
        print("  (requires downloading the Sen1Floods11 SAR dataset ~4GB)")

    cnn_files = [f for f in os.listdir(MODELS_DIR) if "cnn" in f.lower()]
    sizes = {f: os.path.getsize(os.path.join(MODELS_DIR, f)) // 1024 // 1024
             for f in cnn_files}
    print()
    print("  Saved CNN model files:")
    for f, sz in sizes.items():
        print(f"    {f}  ({sz} MB)")


# ─── Before vs After ──────────────────────────────────────────────────────────

def compare_before_after(lstm_after):
    print()
    print("=" * 62)
    print("  BEFORE vs AFTER LEAKAGE FIX")
    print("=" * 62)

    op = os.path.join(MODELS_DIR, "lstm_training_report.json")
    if os.path.exists(op):
        with open(op) as f:
            old = json.load(f)
        print()
        print("  LSTM Forecaster:")
        print(f"  {'Metric':<30} {'BEFORE':>12}  {'AFTER':>12}  Notes")
        print(f"  {'-'*30} {'-'*12}  {'-'*12}  {'-'*24}")
        print(f"  {'Test MAE (m)':<30} {old['test_mae_meters']:>12.1f}  {lstm_after['mae_m']:>12.1f}  BUG1,2,3 removed")
        print(f"  {'Test RMSE (m)':<30} {old['test_rmse_meters']:>12.1f}  {lstm_after['rmse_m']:>12.1f}  GroupKFold split")
        print(f"  {'Test R2':<30} {old['test_r2']:>12.4f}  {lstm_after['r2']:>12.4f}  BUG2 spatial fix")
        print(f"  {'Overfit gap':<30} {old['overfit_gap']:>+12.6f}  {lstm_after['overfit']:>+12.2f}  BUG7 augmentation")
        print(f"  {'Feature 4':<30} {'danger_lvl':>12}  {'excess_above':>12}  BUG3 leakage fix")
        print(f"  {'Scaler fit':<30} {'full data':>12}  {'train only':>12}  BUG1 fix")
        print(f"  {'Split strategy':<30} {'random evt':>12}  {'GroupKFold':>12}  BUG2 fix")

    print()
    print("  CNN Classifier:")
    print(f"  {'Metric':<30} {'BEFORE':>20}  AFTER")
    print(f"  {'-'*30} {'-'*20}  {'-'*25}")
    print(f"  {'Val Accuracy':<30} {'100% (synthetic)':>20}  Pending Sen1Floods11 data")
    print(f"  {'Data source':<30} {'400 synthetic imgs':>20}  Sen1Floods11 real SAR")
    print(f"  {'Split':<30} {'sequential slice':>20}  stratified split (BUG5)")
    print(f"  {'Augmentation':<30} {'val+test polluted':>20}  train-only (BUG5 fix)")


# ─── Backend API ──────────────────────────────────────────────────────────────

def check_backend():
    print()
    print("=" * 62)
    print("  BACKEND API -- Health Check")
    print("=" * 62)
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:8000/", timeout=3) as resp:
            data = json.loads(resp.read())
            print(f"  Status : RUNNING  v{data.get('version','?')}")
            for k, v in data.get("endpoints", {}).items():
                print(f"    {k:<22}: {v}")
    except Exception as e:
        print(f"  Status : NOT RUNNING")
        print(f"  Error  : {type(e).__name__}: {e}")
        print("  Start  : cd backend && uvicorn main:app --reload --port 8000")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 62)
    print("  FloodSentinel AI -- Full Model & Project Evaluation")
    print(f"  TensorFlow {tf.__version__}")
    print("=" * 62)

    lstm_res = evaluate_lstm()
    evaluate_cnn()
    compare_before_after(lstm_res)
    check_backend()

    print()
    print("=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    r = lstm_res
    print(f"  LSTM | MAE={r['mae_m']:.1f}m  RMSE={r['rmse_m']:.1f}m  R2={r['r2']:.3f}  Overfit={r['overfit']:+.1f}m")
    print(f"  LSTM | Beats naive baseline: {r['beats']}")
    print(f"  CNN  | 100% val acc on synthetic data (real metrics need Sen1Floods11)")
    print()
    print("  Next steps:")
    print("  1. Run  python train_lstm_v2.py    (3-seed CI training)")
    print("  2. Run  python train_cnn_v2.py     (needs Sen1Floods11)")
    print("  3. Start backend:  uvicorn main:app --reload  (from backend/)")
    print("  4. Start frontend: npm run dev  (from frontend/)")
    print()
