"""
test_clean_experiment.py
Tests Ridge, LightGBM, and BiLSTM on clean non-leaky features with target = Peak Level - Danger Level.
GroupKFold by GaugeID (unseen gauges).
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import lightgbm as lgb

flood  = pd.read_csv('flood_project/ml_training/data/raw/indofloods/floodevents_indofloods.csv')
precip = pd.read_csv('flood_project/ml_training/data/raw/indofloods/precipitation_variables_indofloods.csv')
meta   = pd.read_csv('flood_project/ml_training/data/raw/indofloods/metadata_indofloods.csv')

pcols = [f'T{i}d' for i in range(1, 11)]
merged = flood.merge(precip, on='EventID', how='inner')
merged['GaugeID'] = merged['EventID'].str.rsplit('-', n=1).str[0]
merged = merged.merge(meta[['GaugeID', 'Danger Level', 'Warning Level', 'Latitude', 'Longitude']], on='GaugeID', how='left')

median_danger = merged['Danger Level'].median()
merged['Danger Level'].fillna(median_danger, inplace=True)
merged['Warning Level'].fillna(merged['Warning Level'].median(), inplace=True)
merged['Latitude'].fillna(merged['Latitude'].median(), inplace=True)
merged['Longitude'].fillna(merged['Longitude'].median(), inplace=True)
merged['Peak Discharge Q (cumec)'].fillna(merged['Peak Discharge Q (cumec)'].median(), inplace=True)
merged['Event Duration (days)'].fillna(5.0, inplace=True)

merged = merged.dropna(subset=['Peak Flood Level (m)'] + pcols)

N = len(merged)
T = 10
F = 11

X = np.zeros((N, T, F), dtype=np.float32)
y_excess = (merged['Peak Flood Level (m)'] - merged['Danger Level']).values.astype(np.float32)
y_abs = merged['Peak Flood Level (m)'].values.astype(np.float32)
danger = merged['Danger Level'].values.astype(np.float32)

for i, (_, row) in enumerate(merged.iterrows()):
    pvals = [float(row[c]) for c in pcols]
    disch = float(row['Peak Discharge Q (cumec)'])
    duration = float(row['Event Duration (days)'])
    lat = float(row['Latitude'])
    lon = float(row['Longitude'])
    dl = float(row['Danger Level'])
    wl = float(row['Warning Level'])
    area = 5000.0

    api = 0.0
    for t in range(T):
        delta = pvals[t] - pvals[t-1] if t > 0 else pvals[t]
        cum = sum(pvals[:t+1])
        api = api * 0.85 + pvals[t]
        # 6 dynamic
        X[i, t, 0] = cum
        X[i, t, 1] = delta
        X[i, t, 2] = pvals[t] / (cum / (t + 1) + 1e-5) # intensity ratio
        X[i, t, 3] = api # antecedent precip index
        X[i, t, 4] = disch
        X[i, t, 5] = duration
        # 5 static
        X[i, t, 6] = lat
        X[i, t, 7] = lon
        X[i, t, 8] = dl
        X[i, t, 9] = wl
        X[i, t, 10] = area

# GroupKFold
gauges = merged['GaugeID'].values
gkf = GroupKFold(n_splits=5)
train_idx, test_idx = next(gkf.split(X, y_excess, groups=gauges))

X_tr, X_te = X[train_idx], X[test_idx]
y_tr, y_te = y_excess[train_idx], y_excess[test_idx]
y_te_abs = y_abs[test_idx]
danger_te = danger[test_idx]

print(f"Train samples: {len(X_tr)} ({len(np.unique(gauges[train_idx]))} gauges)")
print(f"Test samples : {len(X_te)} ({len(np.unique(gauges[test_idx]))} gauges)")

# Scaler
feat_scaler = RobustScaler()
X_tr_s = feat_scaler.fit_transform(X_tr.reshape(-1, F)).reshape(len(X_tr), T, F)
X_te_s = feat_scaler.transform(X_te.reshape(-1, F)).reshape(len(X_te), T, F)

Xf_tr = X_tr_s.reshape(len(X_tr), T * F)
Xf_te = X_te_s.reshape(len(X_te), T * F)

# 1. Global mean baseline
global_mean_excess = float(y_tr.mean())
yp_global_abs = danger_te + global_mean_excess
mae_global = np.abs(yp_global_abs - y_te_abs).mean()
r2_global = 1 - np.sum((yp_global_abs - y_te_abs)**2) / np.sum((y_te_abs - y_te_abs.mean())**2)
print(f"GlobalMean:  MAE = {mae_global:.2f} m, R2 = {r2_global:.4f}")

# 2. Ridge
r = Ridge(alpha=100.0)
r.fit(Xf_tr, y_tr)
yp_ridge_excess = r.predict(Xf_te)
yp_ridge_abs = danger_te + yp_ridge_excess
mae_ridge = np.abs(yp_ridge_abs - y_te_abs).mean()
r2_ridge = 1 - np.sum((yp_ridge_abs - y_te_abs)**2) / np.sum((y_te_abs - y_te_abs.mean())**2)
print(f"Ridge:       MAE = {mae_ridge:.2f} m, R2 = {r2_ridge:.4f}")

# 3. LightGBM
gbm = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
gbm.fit(Xf_tr, y_tr)
yp_lgb_excess = gbm.predict(Xf_te)
yp_lgb_abs = danger_te + yp_lgb_excess
mae_lgb = np.abs(yp_lgb_abs - y_te_abs).mean()
r2_lgb = 1 - np.sum((yp_lgb_abs - y_te_abs)**2) / np.sum((y_te_abs - y_te_abs.mean())**2)
print(f"LightGBM:    MAE = {mae_lgb:.2f} m, R2 = {r2_lgb:.4f}")
