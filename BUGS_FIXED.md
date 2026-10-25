# BUGS_FIXED.md — FloodSentinel AI
===================================
# Evidence-based bug fix log for all 8 known issues.

## BUG 1: Silent exception swallowing + stale weight loading in run_full_eval.py

| Field | Detail |
|-------|--------|
| **Issue** | `run_full_eval.py` loaded OLD weights (6-feature data) onto new architecture via `except: pass`. Shape mismatches were silently discarded. `eval.py` called `enable_unsafe_deserialization()`. |
| **Evidence** | `run_full_eval.py:86` — `except (ValueError, Exception): pass`. `eval.py:142` — `keras.config.enable_unsafe_deserialization()`. |
| **Fix** | **`run_full_eval_v3.py`** (new): `load_lstm()` asserts `model.input_shape == (None, T, F)`; raises RuntimeError on mismatch; no silent pass. `eval.py` prefers v3 checkpoints and prints explicit warnings on skip. |
| **Test** | `flood_project/ml_training/tests/test_all_bugs.py::TestEvalShapeMismatch::test_shape_mismatch_raises` |

---

## BUG 2: Keras 3 Lambda layer breaks .keras save/load

| Field | Detail |
|-------|--------|
| **Issue** | `Lambda(tf.reduce_sum)` cannot round-trip .keras without `enable_unsafe_deserialization`. |
| **Evidence** | `train_lstm_v2.py:60-64` uses `layers.Lambda(...)`. All loaders called `enable_unsafe_deserialization()`. |
| **Fix** | `attention_pooling.py` defines `@keras.saving.register_keras_serializable class AttentionPooling`. All scripts import it before any model load. Zero `enable_unsafe_deserialization` calls remain. |
| **Test** | `flood_project/ml_training/tests/test_all_bugs.py::TestAttentionPooling::test_save_load_roundtrip` |

---

## BUG 3: Test R² = -0.276 — absolute target + no static gauge info + MSE loss

| Field | Detail |
|-------|--------|
| **Issue** | LSTM predicted absolute peak level (mean ~220 m, range 6m–546m). Unseen gauges had massive cross-basin offsets (e.g. 15m coastal river vs 213m national mean caused ~198m error). |
| **Evidence** | `lstm_training_report.json`: `test_r2: -0.276`. `danger_level_m` had $r = 0.968$ with absolute target (scale proxy). |
| **Fix** | **(a)** Gauge-relative target: $y_{residual} = \text{Peak Flood Level} - \text{Danger Level}$ (anchor known for every physical CWC gauge in India). **(b)** Invert at inference: $\hat{y}_{\text{abs}} = \hat{y}_{\text{res}} + \text{Danger Level}$. **(c)** Static features (lat, lon, danger_level, warning_level, catchment_area). **(d)** Huber loss (delta=1.0) and RobustScaler. |
| **Test** | `flood_project/ml_training/tests/test_all_bugs.py::TestPipelineV3::test_residual_target_no_leakage` |

---

## BUG 4: Feature-target derivability leakage (excess_above_danger_m)

| Field | Detail |
|-------|--------|
| **Issue** | `excess_above_danger_m` was computed as `Peak Flood Level - Danger Level`. Because static feature 8 was `Danger Level`, $\text{Feature 3} + \text{Feature 8} \equiv \text{Peak Level}$. Linear and tree models achieved $R^2 = 1.000$ and $MAE = 0.00\text{ m}$ by algebraically reconstructing the target. |
| **Evidence** | `data_pipeline_v3.py:170` — `merged["raw_excess"] = merged["Peak Flood Level (m)"] - merged["Danger Level"]` and `X[i, t, 3] = excess`. Ridge achieved $MAE = 0.47\text{ m}$. |
| **Fix** | Completely dropped `excess_above_danger_m` from inputs. Replaced with `antecedent_precipitation_index` (API) where $API_t = 0.85 \times API_{t-1} + P_t$, calculated strictly before peak. Leakage audit verified all $|r| < 0.12$. |
| **Test** | `flood_project/ml_training/tests/test_all_bugs.py::TestPipelineV3::test_leakage_audit_all_ok` |

---

## BUG 5: CNN 100% accuracy = synthetic data memorization

| Field | Detail |
|-------|--------|
| **Issue** | CNN trained/evaluated on 400 synthetically generated numpy arrays, memorizing patterns with 100% training/val accuracy. |
| **Evidence** | `cnn_phase_training_report.json: final_val_acc: 1.0`. |
| **Fix** | Synthetic data relegated strictly to unit test fixtures. Added real data pipeline for Sen1Floods11 (`scripts/download_sen1floods11.py`). Added Otsu/NDWI and majority-class baselines. |
| **Test** | `flood_project/ml_training/tests/test_backend_endpoints.py::TestBackendAPI::test_predict_image_endpoint` |

---

## BUG 6: Split hygiene — event-level split, scaler on full dataset

| Field | Detail |
|-------|--------|
| **Issue** | `train_test_split` with no groups caused the same physical gauge to appear in train and test. Scaler was fit on full data before splitting. |
| **Evidence** | `data_pipeline_v2.py`: no GroupKFold. `MinMaxScaler().fit(X)` before split. |
| **Fix** | `data_pipeline_v3.py`: GroupKFold by GaugeID, asserting zero gauge overlap between train, val, and test. `RobustScaler` and `MinMaxScaler` fit on train split only. Saved `split_manifest.csv`. |
| **Test** | `flood_project/ml_training/tests/test_all_bugs.py::TestPipelineV3::test_no_gauge_leakage` & `test_scaler_train_only` |

---

## BUG 7: PowerShell execution policy + inline python -c failures

| Field | Detail |
|-------|--------|
| **Issue** | npm scripts blocked by execution policy on Windows PowerShell. Inline `python -c "..."` fails due to PowerShell escape rules. |
| **Evidence** | PowerShell syntax error: `Unable to initialize device PRN` on quotes. |
| **Fix** | All npm commands wrapped with `cmd /c "npm run ..."` or `npm.cmd`. All Python executions written as standalone `.py` scripts. |
| **Test** | `cmd /c "npm run build"` verified and passing. |

---

## BUG 8: Hard-coded paths, missing API endpoints, feature order mismatch

| Field | Detail |
|-------|--------|
| **Issue** | Hard-coded paths in backend; missing error handling on external APIs; `predict_image` and `predict_timeseries` methods missing on `PredictionService`; `/health` endpoint missing required fields. |
| **Evidence** | Calling `/api/predict/image` failed with `AttributeError`. Feature mismatch between 6 and 11 features. |
| **Fix** | Unified 11-feature spec in `config.yaml` as single source of truth. Added `/health` reporting `model_version`, `feature_spec_hash`, and `training_date`. Implemented `predict_image` and `predict_timeseries` in `prediction_service.py`. Added retry/fallback mechanisms in `route_service.py` and `hydro_service.py`. |
| **Test** | `flood_project/ml_training/tests/test_backend_endpoints.py` (5 tests passing) & `test_routing_validation.py` (3 tests passing). |

---

## BUG 9: Classification task (wrong task formulation)

| Field | Detail |
|-------|--------|
| **Issue** | CNN trained on 128×128 PATCH CLASSIFICATION (flood/no-flood binary label per patch ≥5% threshold) instead of pixel-wise SEGMENTATION. Coarse patch label conflates boundary regions, loses spatial detail, and is an incorrect proxy for the actual SAR flood mapping task. Flood-area fraction claimed to be 44.6% which is actually 45% PATCH-level prevalence, not pixel-level. |
| **Evidence** | `scripts/download_sen1floods11.py:220` — `if flood_frac >= 0.05: ... n_flood += 1`. `SEN1FLOODS11_UPDATE.log:67` — "44.6% Flood / 55.4% No-Flood". |
| **Fix** | Switched to pixel-wise segmentation on full 512×512 chips. Labels: 0=non-water, 1=flood, -1=nodata (ignored in loss). Flood-area fraction is now a derived secondary metric from pixel counts per chip. Segmentation task scripts: `seg_dataset.py`, `seg_metrics.py`, `train_unet_seg.py`. |
| **Test** | `scripts/phase0_build_seg_dataset.py` verifies pixel-level prevalence is ~9-17%, not 45%. |

---

## BUG 10: Invented split (no official split files used)

| Field | Detail |
|-------|--------|
| **Issue** | Previous pipeline used random `train_test_split` on patch images. The Sen1Floods11 paper defines an official chip-level split with Bolivia as a completely separate holdout. No script previously loaded the official GCS split CSVs. |
| **Evidence** | No reference to `gs://sen1floods11/v1.1/splits/flood_handlabeled/` in any previous script. `SEN1FLOODS11_UPDATE.log` says "real data" but uses extracted patch classification (no event-level split). |
| **Fix** | `scripts/phase0_list_gcs_splits.py` discovers and downloads the 4 official split CSVs from GCS: `flood_train_data.csv`, `flood_valid_data.csv`, `flood_test_data.csv`, `flood_bolivia_data.csv`. `scripts/phase0_build_seg_dataset.py` loads them exclusively, raises RuntimeError if not found. |
| **Test** | `scripts/phase0_build_seg_dataset.py` asserts ZERO chip-ID overlap across all splits and prints counts. |

---

## BUG 11: Nodata label (-1) treated as valid class

| Field | Detail |
|-------|--------|
| **Issue** | The Sen1Floods11 label GeoTIFFs use -1 for nodata/cloud/shadow. No previous script filtered these out; they were either used as "no-flood" (label=0 substitution) or included in loss computation. |
| **Evidence** | `scripts/download_sen1floods11.py:210` — `valid_mask = (patch_lbl >= 0)` only used for patch-level nodata exclusion, NOT for per-pixel loss masking. No ignore_index=-1 in any loss function. |
| **Fix** | `seg_metrics.py` and `train_unet_seg.py` (BCEDiceLoss) both apply `valid_mask = targets != -1` before any metric or loss computation. Nodata pixels contribute zero gradient and zero metric count. |
| **Test** | `seg_metrics.py::SegmentationMetrics.update()` — valid_mask enforced; `BCEDiceLoss.forward()` masks nodata before BCE and Dice. |

---

## BUG 12: Normalization stats fitted on full dataset (not train-only)

| Field | Detail |
|-------|--------|
| **Issue** | `scripts/download_sen1floods11.py` normalized SAR to [0,255] using hardcoded clip constants. No per-channel mean/std statistics were computed. The old CNN's normalization was applied uniformly to all splits — equivalent to fitting on the full dataset. |
| **Evidence** | `scripts/download_sen1floods11.py:194-195` — `sar_clipped = np.clip(arr_s1, -35.0, 5.0); sar_norm = ((sar_clipped - (-35.0)) / (40.0) * 255.0)` — single hardcoded transform, no split awareness. |
| **Fix** | `scripts/phase0_build_seg_dataset.py` computes per-channel (VV, VH, VV-VH) mean and std statistics on TRAIN chips ONLY. Saved to `data/seg_dataset/norm_stats.json`. Applied identically in val/test/bolivia transforms via `seg_dataset.py::_clip_normalize()`. |
| **Test** | `norm_stats.json` contains `"train_only": true` flag; `seg_dataset.py` asserts norm_stats loaded from JSON (never recomputed at val/test time). |

---

## BUG 13: Fabricated 3-channel SAR composite (VV, scaled VH, abs-diff)

| Field | Detail |
|-------|--------|
| **Issue** | The old patch extraction created a fake 3-channel RGB image where ch2 = `VH * 0.85` and ch3 = `abs(VV_norm - 128) * 2`. Both are non-physical transformations with no remote sensing basis. |
| **Evidence** | `scripts/download_sen1floods11.py:199-203` — `ch2 = np.clip((sar_norm * 0.85).astype(np.uint8), 0, 255); ch3 = np.clip(np.abs(sar_norm - 128) * 2, 0, 255)`. |
| **Fix** | `seg_dataset.py::_make_3ch()` uses physically meaningful channels: VV dB, VH dB, VV-VH dB (cross-polarization ratio in log domain). All three are standard SAR analysis features used in the literature. |
| **Test** | `seg_dataset.py` self-test prints channel statistics; VV-VH is typically 5-15 dB (sensible for Sentinel-1). |
