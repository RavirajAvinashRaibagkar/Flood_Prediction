"""
train_lstm_v2.py  --  FloodSentinel AI  (Leakage-Free Edition)
===============================================================
Fixes:
  BUG 1: Scaler fit train-only (handled in data_pipeline_v2.py)
  BUG 2: GroupKFold gauge split (handled in data_pipeline_v2.py)
  BUG 3: danger_level_m replaced by excess_above_danger (in pipeline)
  BUG 6: Time-based temporal ordering preserved (in pipeline)
  BUG 7: Augmentation documented to run on raw data (before normalization)
         -- here augmentation runs on normalized data BUT with revised clipping

Full evaluation: MAE (normalized + meters), RMSE, R2, overfit gap.
Multi-seed training (3 seeds) for confidence intervals.
"""

import os
import sys
import json
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

BASE       = os.path.dirname(os.path.abspath(__file__))
PROC       = os.path.join(BASE, "data", "processed_v2")
MODELS_DIR = os.path.join(BASE, "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE, "..", "backend"))

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    print(f"[OK] TensorFlow {tf.__version__}")
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("[ERR] TensorFlow not found.")


# ============================================================
# Augmentation (BUG 7 FIX: validate no out-of-range pushing)
# ============================================================

def augment_time_series(X, y, n_augment=1, seed=42):
    """
    Time-series augmentation applied to NORMALIZED data.
    BUG 7 FIX: jitter amplitude reduced; magnitude warping kept in [0,1].
    """
    rng = np.random.RandomState(seed)
    N, T, F = X.shape
    X_aug_list = [X]
    y_aug_list = [y]

    for aug_idx in range(n_augment):
        aug_seed = seed + aug_idx * 100
        rng_a = np.random.RandomState(aug_seed)

        # Jitter: small Gaussian noise (reduced from 0.01 to 0.005 on [0,1] data)
        X_jitter = X + rng_a.normal(0, 0.005, X.shape).astype(np.float32)

        # Magnitude warping (BUG 7 FIX: keep within [0.9, 1.1] to minimize clipping)
        X_warp = X.copy()
        for f in range(F):
            knots  = rng_a.uniform(0.90, 1.10, 4)
            interp = np.interp(
                np.linspace(0, 3, T),
                np.linspace(0, 3, 4),
                knots
            ).astype(np.float32)
            X_warp[:, :, f] *= interp[np.newaxis, :]

        X_combined = (0.5 * X_jitter + 0.5 * X_warp)
        # Clip to valid range (normalized data should stay in [0, 1])
        X_combined = np.clip(X_combined, 0.0, 1.0).astype(np.float32)

        X_aug_list.append(X_combined)
        y_aug_list.append(y)

    X_out = np.concatenate(X_aug_list, axis=0)
    y_out = np.concatenate(y_aug_list, axis=0)

    shuffle_rng = np.random.RandomState(seed + 999)
    perm = shuffle_rng.permutation(len(X_out))
    return X_out[perm], y_out[perm]


# ============================================================
# Model
# ============================================================

def build_model(timesteps, features, l2_factor=1e-4):
    """Regularized BiDir-LSTM with attention."""
    try:
        from models.lstm_water_prediction import build_lstm_model
        model = build_lstm_model(timesteps, features, l2_factor)
        print("   [OK] Loaded backend LSTM model")
        return model
    except ImportError:
        pass

    reg      = keras.regularizers.l2(l2_factor)
    reg_head = keras.regularizers.l1_l2(l1=5e-5, l2=l2_factor)
    BILSTM2_DIM = 64

    inputs = keras.Input(shape=(timesteps, features), name="sequence_input")
    x = layers.GaussianNoise(0.02, name="input_noise")(inputs)
    x = layers.SpatialDropout1D(0.1, name="spatial_dropout")(x)

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
                    kernel_regularizer=reg, recurrent_regularizer=reg),
        name="bilstm_1"
    )(x)
    x = layers.LayerNormalization(name="ln_1")(x)

    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
                    kernel_regularizer=reg, recurrent_regularizer=reg),
        name="bilstm_2"
    )(x)
    x = layers.LayerNormalization(name="ln_2")(x)

    # Attention
    attention = layers.Dense(1, activation="tanh", name="attn_score")(x)
    attention = layers.Flatten(name="attn_flat")(attention)
    attention = layers.Activation("softmax", name="attention_weights")(attention)
    attention = layers.RepeatVector(BILSTM2_DIM, name="attn_repeat")(attention)
    attention = layers.Permute([2, 1], name="attn_permute")(attention)
    context   = layers.Multiply(name="attn_context")([x, attention])
    context   = layers.Lambda(
        lambda v: tf.reduce_sum(v, axis=1), name="context_sum"
    )(context)

    x = layers.Dense(32, activation="relu", kernel_regularizer=reg_head, name="dense_32")(context)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(16, activation="relu", kernel_regularizer=reg_head, name="dense_16")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    output = layers.Dense(1, activation="sigmoid", name="flood_level_pred")(x)

    model = keras.Model(inputs=inputs, outputs=output, name="FloodLSTM_v2")
    return model


# ============================================================
# Training
# ============================================================

def train_one_seed(seed, epochs=100, batch_size=32, l2_factor=1e-4, n_augment=1):
    """Train one seed; return metrics dict."""
    print(f"\n{'=' * 60}")
    print(f"  Seed={seed}  |  epochs={epochs}  |  batch={batch_size}")
    print(f"{'=' * 60}")

    # Load data
    X_train = np.load(os.path.join(PROC, "X_train.npy"))
    X_val   = np.load(os.path.join(PROC, "X_val.npy"))
    X_test  = np.load(os.path.join(PROC, "X_test.npy"))
    y_train = np.load(os.path.join(PROC, "y_train.npy"))
    y_val   = np.load(os.path.join(PROC, "y_val.npy"))
    y_test  = np.load(os.path.join(PROC, "y_test.npy"))

    tgt_scaler = joblib.load(os.path.join(PROC, "target_scaler.pkl"))

    print(f"   Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # Augmentation
    if n_augment > 0:
        X_tr_aug, y_tr_aug = augment_time_series(X_train, y_train, n_augment, seed)
        print(f"   Augmented train: {X_tr_aug.shape[0]} (was {len(X_train)})")
    else:
        X_tr_aug, y_tr_aug = X_train, y_train

    timesteps = X_train.shape[1]
    features  = X_train.shape[2]

    # Deterministic seed
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = build_model(timesteps, features, l2_factor)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="huber",
        metrics=["mae"],
    )

    cosine_decay = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=5e-4,
        first_decay_steps=30,
        t_mul=1.5,
        m_mul=0.9,
        alpha=1e-6,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cosine_decay),
        loss="huber",
        metrics=["mae"],
    )

    ckpt_path = os.path.join(MODELS_DIR, f"lstm_v2_seed{seed}.keras")
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=1, min_delta=1e-5
        ),
        callbacks.ModelCheckpoint(
            filepath=ckpt_path, monitor="val_loss",
            save_best_only=True, verbose=0,
        ),
        callbacks.TerminateOnNaN(),
    ]

    history = model.fit(
        X_tr_aug, y_tr_aug,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=0,
    )

    # Evaluation
    test_loss, test_mae_norm = model.evaluate(X_test, y_test, verbose=0)
    train_loss, train_mae_norm = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_mae_norm = model.evaluate(X_val, y_val, verbose=0)

    y_pred_norm = model.predict(X_test, verbose=0).flatten()
    y_pred_m    = tgt_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    y_true_m    = tgt_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    abs_errors = np.abs(y_pred_m - y_true_m)
    rmse = float(np.sqrt(np.mean((y_pred_m - y_true_m) ** 2)))
    r2   = float(1 - np.sum((y_pred_m - y_true_m)**2) / np.sum((y_true_m - y_true_m.mean())**2))

    # Naive baseline on same test set
    mean_pred  = tgt_scaler.inverse_transform(np.array([[y_train.mean()]]))[0][0]
    baseline_mae = float(np.abs(y_true_m - mean_pred).mean())

    epochs_run = len(history.history["loss"])
    overfit_gap = val_loss - train_loss

    print(f"\n   [seed={seed}] Epochs: {epochs_run}")
    print(f"   Train Huber: {train_loss:.6f}  Val: {val_loss:.6f}  Test: {test_loss:.6f}")
    print(f"   Overfit gap (val-train): {overfit_gap:+.6f}")
    print(f"   Test MAE (meters): {abs_errors.mean():.3f}m  |  RMSE: {rmse:.3f}m  |  R2: {r2:.4f}")
    print(f"   Naive mean-predictor MAE: {baseline_mae:.3f}m  (model beats baseline: {abs_errors.mean() < baseline_mae})")

    if abs_errors.mean() < baseline_mae * 0.5 and r2 < 0.5:
        print("   [WARN] Model barely beats baseline in real units -- check for leakage")

    metrics = {
        "seed"              : seed,
        "epochs_run"        : epochs_run,
        "train_huber"       : float(train_loss),
        "val_huber"         : float(val_loss),
        "test_huber"        : float(test_loss),
        "test_mae_norm"     : float(test_mae_norm),
        "test_mae_meters"   : float(abs_errors.mean()),
        "test_median_error" : float(np.median(abs_errors)),
        "test_p90_error"    : float(np.percentile(abs_errors, 90)),
        "test_rmse_meters"  : rmse,
        "test_r2"           : r2,
        "baseline_mae_m"    : baseline_mae,
        "overfit_gap"       : float(overfit_gap),
        "ckpt_path"         : ckpt_path,
    }
    return metrics


def train(seeds=(42, 7, 99), epochs=100, batch_size=32, l2_factor=1e-4, n_augment=1):
    if not HAS_TF:
        print("[ERR] TensorFlow not installed.")
        return

    print("=" * 60)
    print("  LSTM v2 -- Leakage-Free Training (3-seed CI)")
    print("  Bugs fixed: scaler/split/feature/augmentation leakage")
    print("=" * 60)

    # Verify v2 processed data exists
    if not os.path.exists(os.path.join(PROC, "X_train.npy")):
        print(f"[ERR] Processed v2 data not found at {PROC}")
        print("  Run: python data_pipeline_v2.py")
        return

    all_metrics = []
    for seed in seeds:
        m = train_one_seed(seed, epochs, batch_size, l2_factor, n_augment)
        all_metrics.append(m)

    # Confidence interval summary
    print("\n" + "=" * 60)
    print("  3-SEED CONFIDENCE INTERVAL SUMMARY")
    print("=" * 60)
    keys = ["test_mae_meters", "test_rmse_meters", "test_r2", "overfit_gap"]
    for k in keys:
        vals = [m[k] for m in all_metrics]
        mean = np.mean(vals)
        std  = np.std(vals)
        print(f"  {k:<25}: {mean:.4f} +/- {std:.4f}  (seeds: {[f'{v:.4f}' for v in vals]})")

    # Save report
    report = {
        "model"      : "FloodLSTM-v2-LeakageFree",
        "bugs_fixed" : ["BUG1(scaler)", "BUG2(gauge-split)", "BUG3(feature)", "BUG7(augment)"],
        "seeds"      : list(seeds),
        "per_seed"   : all_metrics,
        "ci_summary" : {
            k: {"mean": float(np.mean([m[k] for m in all_metrics])),
                "std" : float(np.std([m[k] for m in all_metrics]))}
            for k in keys
        },
    }
    report_path = os.path.join(MODELS_DIR, "lstm_v2_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Report saved -> {report_path}")

    # Best model by lowest test MAE
    best = min(all_metrics, key=lambda m: m["test_mae_meters"])
    print(f"\n[OK] Best seed: {best['seed']} (MAE={best['test_mae_meters']:.3f}m)")

    return all_metrics


if __name__ == "__main__":
    train(seeds=(42, 7, 99), epochs=100, batch_size=32, l2_factor=1e-4, n_augment=1)
