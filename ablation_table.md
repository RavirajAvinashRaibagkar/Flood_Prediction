# ablation_table.md — FloodSentinel AI

Ablation study evaluated on the **exact same untouched real grouped test set** (336 events, 24 unseen gauges, zero overlap).

| Configuration | Category | Test MAE (m) | RMSE (m) | R² | KGE | Notes |
|---|---|---:|---:|---:|---:|---|
| 1. Baseline (Global Mean) | Baseline | **176.82** | 185.77 | -0.1280 | nan | Predicts training global mean |
| 2. Linear / Ridge Regression | Baseline | **7.37** | 11.32 | 0.9958 | 0.9539 | L2 regularized linear model on flattened sequence |
| 3. LightGBM Regressor | Tree-based Baseline | **10.43** | 18.12 | 0.9893 | 0.9448 | 300 trees, leaf-wise depth |
| 4. XGBoost Regressor | Tree-based Baseline | **8.06** | 14.64 | 0.9930 | 0.9726 | 300 trees, depth 5 |
| 5. BiLSTM + AttentionPooling (Full v3) | Deep Sequence Architecture | **2.51** | 8.18 | 0.9978 | 0.9840 | BiDir-LSTM(64,32) + AttentionPooling + Augmentation |
| 6. Stacked Ensemble (LSTM + XGB + Ridge) | Ensemble | **4.12** | 8.75 | 0.9975 | 0.9809 | 50% BiLSTM-v3 + 30% XGBoost + 20% Ridge |
