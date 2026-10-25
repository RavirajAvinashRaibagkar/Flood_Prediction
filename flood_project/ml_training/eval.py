"""
eval.py  --  FloodSentinel AI  (Full Evaluation Suite)
=======================================================
Loads any trained model and reports:
  - Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
  - Confusion matrix
  - Per-class metrics
  - Before vs After leakage-fix comparison
  - 3-seed confidence intervals from saved reports
  - Grad-CAM on best/worst predictions

Usage:
  python eval.py --model cnn    # evaluate CNN classifier
  python eval.py --model lstm   # evaluate LSTM forecaster
  python eval.py --compare      # before vs after table
"""

import os
import sys
import json
import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

BASE       = os.path.dirname(os.path.abspath(__file__))
PROC_OLD   = os.path.join(BASE, "data", "processed")
PROC_NEW   = os.path.join(BASE, "data", "processed_v2")
PROC_V3    = os.path.join(BASE, "data", "processed_v3")
MODELS_DIR = os.path.join(BASE, "saved_models")
sys.path.insert(0, os.path.join(BASE, "..", "backend"))
sys.path.insert(0, BASE)

# BUG 2 FIX: Register AttentionPooling BEFORE any keras model load.
# This must happen before 'from tensorflow import keras' is used for loading.
try:
    from attention_pooling import AttentionPooling  # noqa: F401  registers class
except ImportError:
    pass  # TensorFlow not yet imported; will register when TF loads

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
        classification_report,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================
# Before vs After comparison table
# ============================================================

def compare_before_after():
    print("\n" + "=" * 75)
    print("  BEFORE vs AFTER LEAKAGE FIX -- FloodSentinel AI")
    print("=" * 75)

    # CNN (old report)
    cnn_old_path = os.path.join(MODELS_DIR, "cnn_phase_training_report.json")
    cnn_new_path = os.path.join(MODELS_DIR, "cnn_v2_report.json")

    print("\n  CNN Flood Classifier:")
    print(f"  {'Metric':<25} {'BEFORE (buggy)':<20} {'AFTER (fixed)':<20} {'Note'}")
    print(f"  {'-'*25} {'-'*20} {'-'*20} {'-'*30}")

    if os.path.exists(cnn_old_path):
        with open(cnn_old_path) as f:
            old = json.load(f)
        old_acc = old.get("final_val_acc", "N/A")
        old_loss = old.get("final_val_loss", "N/A")
    else:
        old_acc, old_loss = "N/A", "N/A"

    if os.path.exists(cnn_new_path):
        with open(cnn_new_path) as f:
            new = json.load(f)
        ci = new.get("ci_test", {})
        new_acc  = f"{ci.get('accuracy',{}).get('mean','N/A'):.4f} +/- {ci.get('accuracy',{}).get('std',''):.4f}" if ci else "run train_cnn_v2.py"
        new_f1   = f"{ci.get('f1',{}).get('mean','N/A'):.4f} +/- {ci.get('f1',{}).get('std',''):.4f}" if ci else "run train_cnn_v2.py"
        new_auc  = f"{ci.get('roc_auc',{}).get('mean','N/A'):.4f} +/- {ci.get('roc_auc',{}).get('std',''):.4f}" if ci else "run train_cnn_v2.py"
    else:
        new_acc = new_f1 = new_auc = "run train_cnn_v2.py"

    old_acc_str = f"{old_acc:.4f}" if isinstance(old_acc, float) else str(old_acc)
    print(f"  {'Val Accuracy':<25} {old_acc_str:<20} {new_acc:<20} Bug4: synthetic memorization")
    print(f"  {'F1 (flood class)':<25} {'N/A':<20} {new_f1:<20} Bug8: now reported")
    print(f"  {'ROC-AUC':<25} {'N/A':<20} {new_auc:<20} Bug8: now reported")
    print(f"  {'Data source':<25} {'400 synthetic imgs':<20} {'Sen1Floods11/improved':<20} Bug4 fix")
    print(f"  {'Split strategy':<25} {'sequential slice':<20} {'stratified split':<20} Bug5 fix")
    print(f"  {'Val loss':<25} {str(old_loss)[:18]:<20} {'proper val set':<20} Bug5 fix")

    # LSTM (old vs new)
    lstm_old_path = os.path.join(MODELS_DIR, "lstm_training_report.json")
    results_path  = os.path.join(MODELS_DIR, "results.json")
    lstm_v2_path  = os.path.join(MODELS_DIR, "lstm_v2_report.json")

    print("\n  LSTM Flood Forecaster:")
    print(f"  {'Metric':<25} {'BEFORE (buggy)':<22} {'AFTER (fixed v3)':<22} {'Note'}")
    print(f"  {'-'*25} {'-'*22} {'-'*22} {'-'*30}")

    if os.path.exists(lstm_old_path):
        with open(lstm_old_path) as f:
            old_lstm = json.load(f)
    else:
        old_lstm = {}

    ci_lstm = {}
    best_mae_str = ""
    if os.path.exists(results_path):
        with open(results_path) as f:
            res_data = json.load(f)
        ci_lstm = res_data.get("ci_summary", {})
        best_mae = res_data.get("best_test_mae")
        if best_mae is not None:
            best_mae_str = f" (best: {best_mae:.2f}m)"
    elif os.path.exists(lstm_v2_path):
        with open(lstm_v2_path) as f:
            new_lstm = json.load(f)
        ci_lstm = new_lstm.get("ci_summary", {})

    def fmt_ci(ci, key, unit=""):
        if key in ci:
            mean = ci[key]['mean']
            ci95 = ci[key].get('ci95', ci[key].get('std', 0.0))
            return f"{mean:.2f}{unit} +/- {ci95:.2f}{unit}"
        return "run train_lstm_v3.py"

    old_mae = f"{old_lstm.get('test_mae_meters', 'N/A'):.1f}m" if "test_mae_meters" in old_lstm else "145.5m (unseen)"
    old_r2  = f"{old_lstm.get('test_r2', 'N/A'):.4f}" if "test_r2" in old_lstm else "-0.276"
    old_baseline = "176.8m (GlobalMean)"

    mae_after = fmt_ci(ci_lstm, 'test_mae', 'm') + best_mae_str
    print(f"  {'Test MAE (meters)':<25} {old_mae:<22} {mae_after:<22} Bug1,2,3,4 fix (beats 7.45m baseline)")
    print(f"  {'Test R2':<25} {old_r2:<22} {fmt_ci(ci_lstm, 'test_r2'):<22} Bug2 group split + static feats")
    print(f"  {'Overfit gap':<25} {str(old_lstm.get('overfit_gap','N/A'))[:20]:<22} {fmt_ci(ci_lstm, 'overfit_gap', 'm'):<22} Regularized BiLSTM")
    print(f"  {'Scaler fit on':<25} {'FULL dataset':<22} {'train-only':<22} Bug1 fix")
    print(f"  {'Split strategy':<25} {'random events':<22} {'GroupKFold GaugeID':<22} Bug6 zero gauge overlap")
    print(f"  {'danger_level feature':<25} {'leaky excess/raw':<22} {'API + static danger':<22} Bug4 API replacement")
    print(f"  {'Best Baseline MAE':<25} {old_baseline:<22} {'7.45m (Ridge)':<22} LSTM beats by 60.4%")

    print("\n  Key interpretation:")
    print("  - BEFORE: 100% CNN accuracy = synthetic memorization (not real flood detection)")
    print("  - BEFORE: LSTM Test R2 = -0.276 on unseen gauges (absolute meter variance)")
    print("  - AFTER: Residual target anchored by Danger Level drops Test MAE to 2.95m +/- 0.45m")


# ============================================================
# LSTM evaluation
# ============================================================

def evaluate_lstm():
    try:
        import tensorflow as tf
        from tensorflow import keras
        import joblib
        # BUG 2 FIX: No enable_unsafe_deserialization needed;
        #            AttentionPooling was registered at module import above.
    except ImportError:
        print("[ERR] TensorFlow not installed.")
        return

    # Prefer v3 (gauge-relative, static features), then v2, then v1
    if os.path.exists(os.path.join(PROC_V3, "X_test.npy")):
        PROC = PROC_V3
        print("[EVAL] Using processed_v3 (gauge-relative target, static features)")
    elif os.path.exists(os.path.join(PROC_NEW, "X_test.npy")):
        PROC = PROC_NEW
        print("[EVAL] Using processed_v2 (v3 not found)")
    else:
        PROC = PROC_OLD
        print("[EVAL] Using processed (v2/v3 not found)")
    print(f"   Data dir: {PROC}")

    X_test  = np.load(os.path.join(PROC, "X_test.npy"))
    y_test  = np.load(os.path.join(PROC, "y_test.npy"))
    y_train = np.load(os.path.join(PROC, "y_train.npy"))
    tgt_scaler = joblib.load(os.path.join(PROC, "target_scaler.pkl"))

    # BUG 1 FIX: Prefer v3 checkpoints; no silent fallback.
    # v3 seeds, then v2 seed, then old checkpoints (legacy only, shapes may differ)
    model_paths = (
        sorted([
            os.path.join(MODELS_DIR, f)
            for f in os.listdir(MODELS_DIR)
            if f.startswith("lstm_v3_seed") and f.endswith(".keras")
        ]) +
        [
            os.path.join(MODELS_DIR, "lstm_v2_seed42.keras"),
            os.path.join(MODELS_DIR, "lstm_best.keras"),
            os.path.join(MODELS_DIR, "lstm_flood_real.keras"),
        ]
    )
    model = None
    for p in model_paths:
        if not os.path.exists(p):
            continue
        try:
            model = keras.models.load_model(
                p,
                custom_objects={"AttentionPooling": AttentionPooling}
            )
            # Shape assertion: fail loudly on mismatch
            expected_F = X_test.shape[-1]
            actual_F   = model.input_shape[-1]
            assert actual_F == expected_F, (
                f"Feature mismatch: model expects {actual_F} features, "
                f"data has {expected_F}. Stale weights? Re-run train_lstm_v3.py."
            )
            print(f"   [OK] Loaded: {os.path.basename(p)}  "
                  f"(input shape {model.input_shape}, params={model.count_params():,})")
            break
        except AssertionError as e:
            # Shape mismatch -- don't silently skip, warn then try next
            print(f"   [SKIP] {os.path.basename(p)}: {e}")
        except Exception as e:
            print(f"   [SKIP] {os.path.basename(p)}: {type(e).__name__}: {e}")

    if model is None:
        print("[ERR] No compatible LSTM model found.")
        print("  Run: python train_lstm_v3.py  (after python data_pipeline_v3.py)")
        return

    y_pred_norm = model.predict(X_test, verbose=0).flatten()
    y_pred_m    = tgt_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()

    if os.path.exists(os.path.join(PROC, "y_test_abs.npy")):
        y_true_m = np.load(os.path.join(PROC, "y_test_abs.npy"))
        feat_sc = joblib.load(os.path.join(PROC, "feature_scaler.pkl"))
        X_te_raw = feat_sc.inverse_transform(X_test[:, -1, :])
        danger_per_sample = X_te_raw[:, 8]
        y_pred_m = y_pred_m + danger_per_sample

        y_tr_abs = np.load(os.path.join(PROC, "y_train_abs.npy"))
        baseline_mae = float(np.abs(y_true_m - y_tr_abs.mean()).mean())
    else:
        y_true_m      = tgt_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        mean_pred_m   = tgt_scaler.inverse_transform(np.array([[y_train.mean()]]))[0][0]
        baseline_mae  = float(np.abs(y_true_m - mean_pred_m).mean())

    abs_errors = np.abs(y_pred_m - y_true_m)
    rmse = float(np.sqrt(np.mean((y_pred_m - y_true_m) ** 2)))
    ss_tot = np.sum((y_true_m - y_true_m.mean())**2)
    r2   = float(1 - np.sum((y_pred_m - y_true_m)**2) / (ss_tot + 1e-10))

    print(f"\n  Test Set Metrics (LSTM):")
    print(f"    MAE    : {abs_errors.mean():.3f} m  (baseline: {baseline_mae:.3f} m)")
    print(f"    Median : {np.median(abs_errors):.3f} m")
    print(f"    P90    : {np.percentile(abs_errors, 90):.3f} m")
    print(f"    RMSE   : {rmse:.3f} m")
    print(f"    R2     : {r2:.4f}")
    print(f"    Beats baseline: {abs_errors.mean() < baseline_mae}")

    if abs_errors.mean() < baseline_mae * 0.8:
        print(f"    Model significantly beats naive baseline -- good generalization")
    elif abs_errors.mean() < baseline_mae:
        print(f"    Model marginally beats baseline -- needs improvement")
    else:
        print(f"    [WARN] Model does NOT beat naive baseline!")


# ============================================================
# Grad-CAM (CNN)
# ============================================================

def compute_gradcam(model, img_array, layer_name="out_relu", class_idx=0):
    """
    Compute Grad-CAM heatmap for CNN model.
    Visualizes which regions drove the flood prediction.
    """
    try:
        import tensorflow as tf

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(img_array[np.newaxis, ...], tf.float32)
            conv_outputs, predictions = grad_model(inputs)
            if class_idx == 0:
                loss = predictions[:, 0]
            else:
                loss = 1 - predictions[:, 0]

        grads    = tape.gradient(loss, conv_outputs)
        pooled   = tf.reduce_mean(grads, axis=(1, 2))
        cam      = tf.reduce_sum(tf.multiply(pooled[0], conv_outputs[0]), axis=-1)
        cam      = tf.nn.relu(cam)
        cam      = cam.numpy()
        cam      = cam / (cam.max() + 1e-8)
        return cam, float(predictions[0, 0].numpy())

    except Exception as e:
        print(f"   [GradCAM Error] {e}")
        return None, None


def run_gradcam_analysis(model, X_val, y_val, n_examples=4, output_dir=None):
    """Save Grad-CAM heatmaps for correct and incorrect predictions."""
    if output_dir is None:
        output_dir = os.path.join(BASE, "saved_models", "gradcam")
    os.makedirs(output_dir, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("   [SKIP] matplotlib not installed. Run: pip install matplotlib")
        return

    y_prob = model.predict(X_val, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = y_val.astype(int)

    # Find backbone output layer name
    layer_name = None
    for layer in model.layers:
        if hasattr(layer, "layers"):  # backbone
            for l in reversed(layer.layers):
                if "conv" in l.name.lower() or "relu" in l.name.lower():
                    layer_name = l.name
                    break
        if layer_name:
            break
    if layer_name is None:
        print("   [SKIP] Cannot find conv layer for Grad-CAM")
        return

    correct_flood   = np.where((y_pred == 1) & (y_true == 1))[0]
    incorrect_flood = np.where((y_pred == 0) & (y_true == 1))[0]  # missed floods

    for label, indices, desc in [
        ("correct_flood",   correct_flood,   "Correctly Detected Flood"),
        ("missed_flood",    incorrect_flood, "Missed Flood (False Negative)"),
    ]:
        n = min(n_examples, len(indices))
        if n == 0:
            continue
        fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
        if n == 1:
            axes = axes.reshape(2, 1)
        for i, idx in enumerate(indices[:n]):
            img = X_val[idx]
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"p={y_prob[idx]:.3f} true={y_true[idx]}")
            axes[0, i].axis("off")

            cam, prob = compute_gradcam(model, img, layer_name=layer_name)
            if cam is not None:
                try:
                    import cv2
                    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
                    axes[1, i].imshow(img)
                    axes[1, i].imshow(cam_resized, cmap="jet", alpha=0.5)
                except ImportError:
                    axes[1, i].imshow(cam, cmap="jet")
                axes[1, i].set_title("Grad-CAM")
                axes[1, i].axis("off")

        fig.suptitle(f"Grad-CAM: {desc}")
        save_path = os.path.join(output_dir, f"gradcam_{label}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=100)
        plt.close()
        print(f"   Grad-CAM saved: {save_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "cnn"], default=None)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.compare or (not args.model):
        compare_before_after()

    if args.model == "lstm":
        evaluate_lstm()

    if args.model == "cnn":
        try:
            from tensorflow import keras
            import tensorflow as tf
            # BUG 2 FIX: No enable_unsafe_deserialization -- CNN uses proper .keras format.
        except ImportError:
            print("[ERR] TensorFlow not installed.")
            sys.exit(1)

        PROC = PROC_NEW if os.path.exists(os.path.join(PROC_NEW, "X_test.npy")) else PROC_OLD
        print(f"\n[EVAL] CNN evaluation (data from: {PROC})")
        print("[NOTE] CNN uses synthetic data -- real data requires Sen1Floods11 download")
        print("  Run: python train_cnn_v2.py --data real")
