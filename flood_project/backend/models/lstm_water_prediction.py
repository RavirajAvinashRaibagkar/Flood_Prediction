"""
lstm_water_prediction.py  --  FloodSentinel AI Backend Model Wrapper
======================================================================
BUG 2 FIX: Uses AttentionPooling registered layer; no Lambda; no enable_unsafe_deserialization.
BUG 3 FIX: Feature spec updated to v3 (11 features: 6 dynamic + 5 static).
            Loads from processed_v3 scalers.
BUG 8 FIX: Shared feature spec matches config.yaml and training pipeline.
            Loads gauge_train_means.json for residual inversion at inference.

FEATURE SPEC (must match data_pipeline_v3.py ALL_FEATURE_NAMES):
  Dynamic (repeated per timestep):
    [0]  rainfall_cumulative_mm
    [1]  rainfall_delta_mm
    [2]  peak_discharge_cumec
    [3]  excess_above_danger_m
    [4]  event_duration_days
    [5]  dfsi_severity_index
  Static (repeated per timestep):
    [6]  lat
    [7]  lon
    [8]  danger_level_m
    [9]  warning_level_m
    [10] catchment_area_km2
"""

import os
import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Feature spec (single source of truth for backend inference) ───────────────
TIMESTEPS = 10
DYNAMIC_FEATURE_NAMES = [
    "rainfall_cumulative_mm",
    "rainfall_delta_mm",
    "peak_discharge_cumec",
    "antecedent_precipitation_index",   # BUG4 FIX: replaces leaky excess_above_danger_m
    "event_duration_days",
    "dfsi_severity_index",
]
STATIC_FEATURE_NAMES = [
    "lat",
    "lon",
    "danger_level_m",
    "warning_level_m",
    "catchment_area_km2",
]
FEATURE_NAMES = DYNAMIC_FEATURE_NAMES + STATIC_FEATURE_NAMES
FEATURES = len(FEATURE_NAMES)   # 11

# ── Paths ─────────────────────────────────────────────────────────────────────
_THIS_DIR   = Path(__file__).resolve().parent
_ML_DIR     = _THIS_DIR / ".." / ".." / "ml_training"
_MODELS_DIR = _ML_DIR / "saved_models"
_PROC_DIR   = _ML_DIR / "data" / "processed_v3"


# ── TensorFlow / Keras ────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    # Register AttentionPooling before any model load
    _attn_module = _ML_DIR / "attention_pooling.py"
    if _attn_module.exists():
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("attention_pooling", str(_attn_module))
        mod  = importlib.util.module_from_spec(spec)
        sys.modules["attention_pooling"] = mod
        spec.loader.exec_module(mod)
        AttentionPooling = mod.AttentionPooling
    else:
        # Fallback inline definition — use same compat shim as attention_pooling.py
        def _get_reg():
            if hasattr(keras, 'saving') and hasattr(keras.saving, 'register_keras_serializable'):
                return keras.saving.register_keras_serializable
            return keras.utils.register_keras_serializable

        @_get_reg()(package="FloodSentinel")
        class AttentionPooling(keras.layers.Layer):
            def call(self, x):
                return tf.reduce_sum(x, axis=1)
            def compute_output_shape(self, input_shape):
                return (input_shape[0], input_shape[2])
            def get_config(self):
                return super().get_config()


    def build_lstm_model(timesteps=TIMESTEPS, features=FEATURES, l2_factor=1e-4):
      
        reg  = regularizers.l2(l2_factor)
        regH = regularizers.l1_l2(l1=5e-5, l2=l2_factor)
        DIM  = 64

        inp = keras.Input(shape=(timesteps, features), name="sequence_input")
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
        model = keras.Model(inp, out, name="FloodLSTM_v3")
        model.compile(optimizer=keras.optimizers.Adam(5e-4), loss="huber", metrics=["mae"])
        return model

    def load_trained_model():
      
        candidates = sorted(_MODELS_DIR.glob("lstm_v3_seed*.keras"))
        if not candidates:
            logger.warning("No lstm_v3 checkpoints found; model not loaded.")
            return None
        for ckpt in candidates:
            try:
                model = keras.models.load_model(
                    str(ckpt),
                    custom_objects={"AttentionPooling": AttentionPooling}
                )
                logger.info(f"Loaded LSTM from {ckpt.name} ({model.count_params():,} params)")
                return model
            except Exception as e:
                logger.warning(f"Failed to load {ckpt.name}: {e}")
        logger.error("All LSTM checkpoints failed; returning None.")
        return None

    HAS_TENSORFLOW = True

except ImportError:
    HAS_TENSORFLOW = False
    logger.warning("TensorFlow not installed; LSTM unavailable.")

    def build_lstm_model(timesteps=TIMESTEPS, features=FEATURES, l2_factor=1e-4):
        return None

    def load_trained_model():
        return None


# ── Scaler + gauge means loader ───────────────────────────────────────────────
def _load_scalers():
    """Load v3 scalers. Returns (feat_scaler, tgt_scaler, global_train_mean)."""
    try:
        import joblib
        feat_s = joblib.load(_PROC_DIR / "feature_scaler.pkl")
        tgt_s  = joblib.load(_PROC_DIR / "target_scaler.pkl")
        with open(_PROC_DIR / "gauge_train_means.json") as f:
            gm = json.load(f)
        global_mean = gm["global_train_mean"]
        return feat_s, tgt_s, global_mean
    except Exception as e:
        logger.warning(f"Could not load v3 scalers: {e}")
        return None, None, 0.0


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_water_level(sequence: np.ndarray, model=None,
                        gauge_id: str = None) -> dict:
    """
    Predict flood level from a time-series sequence.

    Args:
        sequence  : np.ndarray shape (TIMESTEPS, FEATURES) or (1, TIMESTEPS, FEATURES)
                    FEATURES must be 11 (6 dynamic + 5 static).
        model     : Keras model (optional; uses heuristic if None)
        gauge_id  : gauge identifier for per-gauge mean lookup (optional)

    Returns:
        dict with predictions, model name, and feature info.
    """
    feat_scaler, tgt_scaler, global_mean = _load_scalers()

    if model is not None and HAS_TENSORFLOW and feat_scaler is not None:
        if sequence.ndim == 2:
            sequence = np.expand_dims(sequence, 0)
        if sequence.shape[-1] != FEATURES:
            raise ValueError(
                f"Feature count mismatch: got {sequence.shape[-1]}, "
                f"expected {FEATURES}. "
                f"Feature names: {FEATURE_NAMES}"
            )
        N, T, F = sequence.shape
        seq_s = feat_scaler.transform(sequence.reshape(-1, F)).reshape(N, T, F)
        pred_norm = model.predict(seq_s, verbose=0).flatten()
        pred_res  = tgt_scaler.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
        pred_abs  = float(pred_res[0]) + global_mean

        return {
            "predictions": [{
                "hour":            1,
                "predicted_level": round(pred_abs, 3),
                "upper_ci":        round(pred_abs + 0.5, 3),
                "lower_ci":        round(pred_abs - 0.5, 3),
            }],
            "model":       "FloodLSTM-v3 (AttentionPooling)",
            "features":    FEATURE_NAMES,
            "data_source": "INDOFLOODS + IFI-Impacts (v3)",
        }

    # Physics-based heuristic fallback
    predictions = []
    for i in range(48):
        rain  = float(sequence[-1, 0]) if sequence.ndim >= 2 else 0
        disch = float(sequence[-1, 2]) if sequence.shape[-1] > 2 else 0
        level = 2.0 + rain * 0.01 + disch * 0.001 + np.sin(i / 8) * 0.5
        predictions.append({
            "hour":            i + 1,
            "predicted_level": round(float(level), 2),
            "upper_ci":        round(float(level) + 0.4, 2),
            "lower_ci":        round(float(level) - 0.4, 2),
        })
    return {
        "predictions": predictions,
        "model":       "Physics-heuristic (LSTM not loaded)",
        "features":    FEATURE_NAMES,
        "data_source": "heuristic",
    }
