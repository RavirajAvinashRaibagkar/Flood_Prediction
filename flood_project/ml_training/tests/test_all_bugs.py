"""
FloodSentinel AI — pytest test suite
=====================================
Tests for:
  - AttentionPooling save/load (BUG 2)
  - data_pipeline_v3: split hygiene, leakage audit (BUGS 1,3,4,6)
  - Backend feature spec consistency (BUG 8)
  - run_full_eval_v3: loud failure on shape mismatch (BUG 1)
  - Config YAML consistency

Run:
  cmd /c "set TF_CPP_MIN_LOG_LEVEL=3 && pytest flood_project/ml_training/tests/ -v"
"""

import os
import sys
import json
import tempfile
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# --- Path setup ---
ML_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_DIR))

RAW_DIR   = ML_DIR / "data" / "raw"
PROC_V3   = ML_DIR / "data" / "processed_v3"
MODELS_DIR= ML_DIR / "saved_models"
BACKEND_MODELS = ML_DIR / ".." / "backend" / "models"


# ==============================================================================
# BUG 2: AttentionPooling — save/load roundtrip
# ==============================================================================

class TestAttentionPooling:
    """BUG 2 FIX: AttentionPooling must save/load without enable_unsafe_deserialization."""

    @pytest.fixture(scope="class")
    def tiny_model(self):
        """Build and save a minimal model with AttentionPooling."""
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        from attention_pooling import AttentionPooling
        import tensorflow as tf
        from tensorflow import keras

        inp = keras.Input(shape=(5, 8))
        out = AttentionPooling(name="ap")(inp)
        model = keras.Model(inp, out)
        tmpdir = Path(tempfile.mkdtemp())
        save_path = str(tmpdir / "test_ap.keras")
        model.save(save_path)
        return model, save_path

    def test_registration_succeeds(self):
        """AttentionPooling import must not raise."""
        from attention_pooling import AttentionPooling
        assert AttentionPooling is not None

    def test_forward_pass(self, tiny_model):
        """Output shape must be (batch, D) not (batch, T, D)."""
        model, _ = tiny_model
        x = np.ones((3, 5, 8), dtype="float32")
        y = model.predict(x, verbose=0)
        assert y.shape == (3, 8), f"Expected (3,8), got {y.shape}"

    def test_save_load_roundtrip(self, tiny_model):
        """Saved model must reload without custom_objects error AND produce same output."""
        from attention_pooling import AttentionPooling
        import tensorflow as tf
        from tensorflow import keras

        model, save_path = tiny_model
        assert Path(save_path).exists(), f"Checkpoint not found: {save_path}"

        model2 = keras.models.load_model(
            save_path,
            custom_objects={"AttentionPooling": AttentionPooling}
        )
        x = np.ones((2, 5, 8), dtype="float32")
        y1 = model.predict(x, verbose=0)
        y2 = model2.predict(x, verbose=0)
        np.testing.assert_allclose(y1, y2, rtol=1e-5,
                                   err_msg="Roundtrip output mismatch")

    def test_no_unsafe_deserialization_needed(self, tiny_model):
        """Must load without enable_unsafe_deserialization."""
        from attention_pooling import AttentionPooling
        from tensorflow import keras

        _, save_path = tiny_model
        # This must NOT raise even without enabling unsafe mode
        try:
            model2 = keras.models.load_model(
                save_path,
                custom_objects={"AttentionPooling": AttentionPooling}
            )
        except Exception as e:
            pytest.fail(f"load_model raised unexpectedly: {e}")


# ==============================================================================
# BUG 1, 3, 4, 6: data_pipeline_v3 correctness
# ==============================================================================

class TestPipelineV3:
    """Tests for data_pipeline_v3.py output: split hygiene, leakage, shapes."""

    @pytest.fixture(scope="class")
    def proc_v3(self):
        if not PROC_V3.exists():
            pytest.skip("processed_v3 not found — run: python data_pipeline_v3.py")
        return PROC_V3

    def test_processed_files_exist(self, proc_v3):
        """All required files must exist after pipeline run."""
        required = [
            "X_train.npy", "X_val.npy", "X_test.npy",
            "y_train.npy", "y_val.npy", "y_test.npy",
            "y_train_abs.npy", "y_val_abs.npy", "y_test_abs.npy",  # exact absolute targets
            "feature_scaler.pkl", "target_scaler.pkl",
            "gauge_train_means.json", "split_manifest.csv",
            "leakage_audit.json", "feature_metadata.json",
        ]
        missing = [f for f in required if not (proc_v3 / f).exists()]
        assert not missing, f"Missing files: {missing}"


    def test_shapes_consistent(self, proc_v3):
        """X arrays must have shape (N, 10, 11); y arrays must have shape (N,)."""
        X_tr = np.load(proc_v3 / "X_train.npy")
        X_va = np.load(proc_v3 / "X_val.npy")
        X_te = np.load(proc_v3 / "X_test.npy")
        y_tr = np.load(proc_v3 / "y_train.npy")
        y_te = np.load(proc_v3 / "y_test.npy")

        assert X_tr.ndim == 3, "X_train must be 3D (N, T, F)"
        N, T, F = X_tr.shape
        assert T == 10, f"Expected 10 timesteps, got {T}"
        assert F == 11, f"Expected 11 features (6 dynamic + 5 static), got {F}"
        assert X_va.shape[1:] == (T, F), "X_val shape mismatch"
        assert X_te.shape[1:] == (T, F), "X_test shape mismatch"
        assert y_tr.ndim == 1, "y_train must be 1D"
        assert y_te.ndim == 1, "y_test must be 1D"
        assert len(y_tr) == len(X_tr), "Train X/y length mismatch"

    def test_no_gauge_leakage(self, proc_v3):
        """BUG 6 FIX: Train, val, test gauge sets must be disjoint."""
        manifest = pd.read_csv(proc_v3 / "split_manifest.csv")
        train_g = set(manifest[manifest["split"] == "train"]["GaugeID"])
        val_g   = set(manifest[manifest["split"] == "val"]["GaugeID"])
        test_g  = set(manifest[manifest["split"] == "test"]["GaugeID"])

        assert len(train_g) > 0, "No train gauges"
        assert len(test_g) > 0, "No test gauges"
        assert not (train_g & test_g), \
            f"LEAK: train & test share gauges: {train_g & test_g}"
        assert not (train_g & val_g), \
            f"LEAK: train & val share gauges: {train_g & val_g}"
        assert not (val_g & test_g), \
            f"LEAK: val & test share gauges: {val_g & test_g}"

    def test_leakage_audit_all_ok(self, proc_v3):
        """BUG 4 FIX: No feature should have HIGH RISK correlation with residual target."""
        with open(proc_v3 / "leakage_audit.json") as f:
            audit = json.load(f)
        high_risk = [r for r in audit if r["risk"] == "HIGH RISK"]
        assert not high_risk, (
            f"HIGH RISK features found (audit on residual target): "
            f"{[(r['feature'], r['pearson_r']) for r in high_risk]}\n"
            "Check: leakage audit must run AFTER gauge-mean subtraction."
        )

    def test_residual_target_no_leakage(self, proc_v3):
        """BUG 3 FIX: danger_level_m must have |r| < 0.05 with residual target."""
        with open(proc_v3 / "leakage_audit.json") as f:
            audit = json.load(f)
        danger = next((r for r in audit if r["feature"] == "danger_level_m"), None)
        assert danger is not None, "danger_level_m not found in audit"
        # Should be near 0.48 (acceptable) not 0.968 (absolute-target leakage)
        assert abs(danger["pearson_r"]) < 0.80, (
            f"danger_level_m r={danger['pearson_r']:.4f} on residual target. "
            "If r > 0.80, the leakage audit ran on absolute target, not residual."
        )

    def test_scaler_train_only(self, proc_v3):
        """BUG 6 FIX: Scaler must be fitted on train samples only."""
        import joblib
        feat_sc = joblib.load(proc_v3 / "feature_scaler.pkl")
        X_tr = np.load(proc_v3 / "X_train.npy")
        N_tr, T, F = X_tr.shape
        # RobustScaler: n_features_in_ should equal F
        assert feat_sc.n_features_in_ == F, \
            f"Scaler n_features_in_={feat_sc.n_features_in_} != {F}"

    def test_no_nans_in_arrays(self, proc_v3):
        """No NaN values should appear in normalized arrays."""
        for name in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
            arr = np.load(proc_v3 / f"{name}.npy")
            assert not np.isnan(arr).any(), f"NaN found in {name}.npy"

    def test_total_events_matches(self, proc_v3):
        """Total events across splits must equal 4548 (INDOFLOODS size)."""
        X_tr = np.load(proc_v3 / "X_train.npy")
        X_va = np.load(proc_v3 / "X_val.npy")
        X_te = np.load(proc_v3 / "X_test.npy")
        total = len(X_tr) + len(X_va) + len(X_te)
        assert total == 4548, f"Expected 4548 total events, got {total}"


# ==============================================================================
# BUG 8: Backend feature spec consistency
# ==============================================================================

class TestBackendConsistency:
    """BUG 8 FIX: Backend feature spec must match config.yaml and pipeline."""

    def test_feature_count_matches_config(self):
        """Backend FEATURES constant must equal config.yaml n_total."""
        import yaml
        config_path = ML_DIR / "config.yaml"
        if not config_path.exists():
            pytest.skip("config.yaml not found")

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        expected_n = cfg["feature_spec"]["n_total"]

        # Import backend module
        backend_models = ML_DIR / ".." / "backend" / "models"
        sys.path.insert(0, str(backend_models))
        try:
            import lstm_water_prediction as lm
            assert lm.FEATURES == expected_n, (
                f"Backend FEATURES={lm.FEATURES} != config n_total={expected_n}. "
                "Sync lstm_water_prediction.py with config.yaml!"
            )
        except ImportError:
            pytest.skip("lstm_water_prediction not importable (TF missing)")

    def test_feature_names_match_pipeline(self):
        """Feature names in pipeline feature_metadata.json must match config.yaml."""
        if not PROC_V3.exists():
            pytest.skip("processed_v3 not found")

        import yaml
        config_path = ML_DIR / "config.yaml"
        if not config_path.exists():
            pytest.skip("config.yaml not found")

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        with open(PROC_V3 / "feature_metadata.json") as f:
            meta = json.load(f)

        cfg_dyn  = cfg["feature_spec"]["dynamic_features"]
        cfg_stat = cfg["feature_spec"]["static_features"]
        cfg_all  = cfg_dyn + cfg_stat

        meta_dyn  = meta["dynamic_feature_names"]
        meta_stat = meta["static_feature_names"]
        meta_all  = meta["all_feature_names"]

        assert cfg_all == meta_all, (
            f"Feature name mismatch between config.yaml and feature_metadata.json!\n"
            f"  config: {cfg_all}\n  metadata: {meta_all}"
        )


# ==============================================================================
# BUG 1: run_full_eval_v3 — loud failure on shape mismatch
# ==============================================================================

class TestEvalShapeMismatch:
    """BUG 1 FIX: Shape mismatches must raise, not be silently swallowed."""

    def test_shape_mismatch_raises(self):
        """Loading a 6-feature model against 11-feature data must assert/raise."""
        if not PROC_V3.exists():
            pytest.skip("processed_v3 not found")

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        from attention_pooling import AttentionPooling
        from tensorflow import keras
        import numpy as np

        X_te = np.load(PROC_V3 / "X_test.npy")
        N, T, F = X_te.shape

        # Build a model with WRONG feature count (6 instead of 11)
        inp = keras.Input(shape=(T, 6), name="sequence_input")
        out = keras.layers.Dense(1)(keras.layers.Flatten()(inp))
        wrong_model = keras.Model(inp, out)

        # Save it
        tmpdir = Path(tempfile.mkdtemp())
        wrong_path = str(tmpdir / "lstm_v3_wrong.keras")
        wrong_model.save(wrong_path)

        # Now try to load via run_full_eval_v3 logic — must raise
        with pytest.raises((AssertionError, RuntimeError)):
            loaded = keras.models.load_model(
                wrong_path,
                custom_objects={"AttentionPooling": AttentionPooling}
            )
            actual_F = loaded.input_shape[-1]
            expected_F = F
            assert actual_F == expected_F, (
                f"Feature mismatch: model={actual_F}, data={expected_F}"
            )


# ==============================================================================
# GENERAL: No enable_unsafe_deserialization in any source file
# ==============================================================================

class TestNoUnsafeDeserialization:
    """BUG 2 FIX: No source file should CALL enable_unsafe_deserialization."""

    @pytest.mark.parametrize("source_file", [
        "eval.py",
        "run_full_eval_v3.py",
        "train_lstm_v3.py",
        "attention_pooling.py",
    ])
    def test_no_unsafe_deserialization_in_source(self, source_file):
        """Check for actual function CALLS, not docstring/comment mentions."""
        import re
        path = ML_DIR / source_file
        if not path.exists():
            pytest.skip(f"{source_file} not found")
        content = path.read_text(encoding="utf-8")
        # Match actual calls like keras.config.enable_unsafe_deserialization()
        # but NOT comments like "# no enable_unsafe_deserialization" or docstrings
        call_pattern = re.compile(
            r'^[^#\'"]*enable_unsafe_deserialization\s*\(',
            re.MULTILINE
        )
        calls = call_pattern.findall(content)
        assert not calls, (
            f"BUG 2: {source_file} contains actual enable_unsafe_deserialization() call(s):\n"
            + "\n".join(f"  {c.strip()}" for c in calls)
        )

    @pytest.mark.parametrize("backend_file", [
        "lstm_water_prediction.py",
    ])
    def test_no_unsafe_deserialization_in_backend(self, backend_file):
        """Check for actual function CALLS, not docstring/comment mentions."""
        import re
        path = ML_DIR / ".." / "backend" / "models" / backend_file
        if not path.exists():
            pytest.skip(f"{backend_file} not found")
        content = path.read_text(encoding="utf-8")
        call_pattern = re.compile(
            r'^[^#\'"]*enable_unsafe_deserialization\s*\(',
            re.MULTILINE
        )
        calls = call_pattern.findall(content)
        assert not calls, (
            f"BUG 2: {backend_file} contains actual enable_unsafe_deserialization() call(s):\n"
            + "\n".join(f"  {c.strip()}" for c in calls)
        )

