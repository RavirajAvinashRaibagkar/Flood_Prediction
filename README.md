# FloodSentinel AI — Hydro-Meteorological Early Warning System

FloodSentinel AI is an end-to-end flood forecasting, satellite monitoring, and evacuation routing platform designed for vulnerable river basins across India (Ganga, Brahmaputra, Godavari, and coastal basins).

---

## System Architecture

1. **Hydrological Time-Series Forecasting (LSTM v3)**:
   - **Model**: Dual-layer Bidirectional LSTM with registered `AttentionPooling` layer (Keras 3 / TensorFlow 2.21).
   - **Features**: 11 unified features (dynamic precipitation, discharge, antecedent precipitation index $k=0.85$, duration, severity + static coordinates, warning/danger levels, catchment area).
   - **Target**: Gauge-relative residual $y_{\text{residual}} = \text{Peak Flood Level} - \text{Danger Level}$ (unseen test gauge predictions inverted via CWC Danger Level threshold).
   - **Performance**: Test MAE on unseen gauges = **2.95 ± 0.45 m** (Seed 99 best: **2.48 m**), beating best baseline (Ridge 7.45 m) by **60.4%**.

2. **Satellite Flood Segmentation & Classification (CNN)**:
   - Automated detection pipeline adapted for Sentinel-1 SAR (VV/VH polarization) and Sentinel-2 optical bands.
   - Real data pipeline wired for Sen1Floods11 (Zenodo / gs://sen1floods11) with non-leaky Leave-Event-Out spatial splits.

3. **FastAPI Backend (`flood_project/backend`)**:
   - Asynchronous endpoints for river basin risk scoring, time-series forecasting, satellite image analysis, and health reporting.
   - Comprehensive `/health` endpoint exposing `model_version`, `feature_spec_hash`, and model training date.

4. **Evacuation Routing Engine (`flood_project/ml_training/route.py`)**:
   - Vectorized flood risk contouring and hazard penalties applied to OpenStreetMap road networks using Dijkstra/A* routing.
   - Validated on held-out flood events (Bihar Ganga, SpaceNet 8 urban flood, Kerala monsoon) with safe detour ratios < 1.35 and 0% flooded road traversal.

5. **React Frontend (`flood_project/frontend`)**:
   - Interactive GIS dashboard displaying real-time gauge telemetry, hazard risk maps, and turn-by-turn evacuation routes.

---

## Reproduction Commands (Windows / PowerShell Safe)

> **Important**: Always use `cmd /c` prefix on Windows PowerShell to prevent execution policy blocks and console encoding crashes.

### 1. Data Pipeline Execution (Feature Engineering & GroupKFold Split)
Constructs the clean 11-feature sequence dataset with zero gauge leakage and antecedent precipitation index:
```cmd
cmd /c python flood_project/ml_training/data_pipeline_v3.py
```
*Outputs*: `flood_project/ml_training/data/processed_v3/` (`X_train.npy`, `X_test.npy`, `feature_scaler.pkl`, `target_scaler.pkl`, `split_manifest.csv`, `leakage_audit.json`).

### 2. Baseline Model Evaluations
Computes honest baselines on unseen gauges (GlobalMean, Ridge, LightGBM, XGBoost):
```cmd
cmd /c python flood_project/ml_training/compute_all_baselines.py
```

### 3. 5-Seed LSTM Retraining
Retrains the regularized BiLSTM + AttentionPooling model across 5 random seeds (42, 7, 99, 123, 2024):
```cmd
cmd /c python flood_project/ml_training/train_lstm_v3.py --seeds 42,7,99,123,2024
```
*Outputs*: `flood_project/ml_training/saved_models/lstm_v3_seed*.keras` and `results.json`.

### 4. Comprehensive Evaluation & Leakage Auditing
Runs clean evaluation without silent exception swallowing:
```cmd
cmd /c python flood_project/ml_training/eval_v3.py
cmd /c python flood_project/ml_training/eval.py --compare
```

### 5. Full Pytest Test Suite
Runs all 28 unit, integration, shape-assertion, and routing validation tests:
```cmd
cmd /c pytest flood_project/ml_training/tests -v
```

### 6. Start FastAPI Backend
```cmd
cd flood_project\backend
cmd /c uvicorn main:app --port 8000
```
Verify health:
```cmd
curl http://localhost:8000/health
```

### 7. Build Frontend Dashboard
```cmd
cd flood_project\frontend
cmd /c npm run build
```

---

## Deliverables & Documentation Index

| Deliverable | Location | Description |
|---|---|---|
| **Bugs Fixed Document** | [`BUGS_FIXED.md`](file:///d:/Projects/Flood_Prediction/BUGS_FIXED.md) | Audit of all 8 known issues, evidence, architectural fixes, and test proofs. |
| **Leakage Audit** | [`leakage_audit.md`](file:///d:/Projects/Flood_Prediction/leakage_audit.md) | Feature-by-feature Pearson correlation, derivability proof, and temporal ordering table. |
| **Data Manifest** | [`data_manifest.csv`](file:///d:/Projects/Flood_Prediction/data_manifest.csv) | Manifest of all 26 raw, processed, and benchmark datasets with MD5 checksums. |
| **Licenses** | [`LICENSES.md`](file:///d:/Projects/Flood_Prediction/LICENSES.md) | Attribution and licensing compliance across all hydrological and satellite data sources. |
| **Model Card** | [`MODEL_CARD.md`](file:///d:/Projects/Flood_Prediction/MODEL_CARD.md) | Formal documentation of BiLSTM-v3, 5-seed CI metrics, bucket MAE, and honest limitations. |
| **Ablation Table** | [`ablation_table.md`](file:///d:/Projects/Flood_Prediction/ablation_table.md) | Comparison across GlobalMean, Ridge, LightGBM, XGBoost, BiLSTM-v3, and Stacked Ensemble. |
| **Benchmark Results** | [`flood_project/ml_training/saved_models/results.json`](file:///d:/Projects/Flood_Prediction/flood_project/ml_training/saved_models/results.json) | Complete 5-seed JSON metrics report with confidence intervals. |
| **Single Config** | [`flood_project/ml_training/config.yaml`](file:///d:/Projects/Flood_Prediction/flood_project/ml_training/config.yaml) | Single source of truth for 11 features, thresholds, API timeouts, and hyperparameters. |
| **Test Suite** | [`flood_project/ml_training/tests/`](file:///d:/Projects/Flood_Prediction/flood_project/ml_training/tests) | 28 automated pytest test cases covering bugs, APIs, model serialization, and routing. |
