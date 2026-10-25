"""
check_r2_excess.py
Calculates R2 both on residual excess (without danger level elevation inflation)
and on absolute meters.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb

flood  = pd.read_csv('flood_project/ml_training/data/raw/indofloods/floodevents_indofloods.csv')
precip = pd.read_csv('flood_project/ml_training/data/raw/indofloods/precipitation_variables_indofloods.csv')
meta   = pd.read_csv('flood_project/ml_training/data/raw/indofloods/metadata_indofloods.csv')

pcols = [f'T{i}d' for i in range(1, 11)]
merged = flood.merge(precip, on='EventID', how='inner')
merged['GaugeID'] = merged['EventID'].str.rsplit('-', n=1).str[0]
merged = merged.merge(meta[['GaugeID', 'Danger Level', 'Warning Level', 'Latitude', 'Longitude']], on='GaugeID', how='left')

median_danger = merged['Danger Level'].median()
merged['Danger Level'] = merged['Danger Level'].fillna(median_danger)
merged['Warning Level'] = merged['Warning Level'].fillna(merged['Warning Level'].median())
merged['Latitude'] = merged['Latitude'].fillna(merged['Latitude'].median())
merged['Longitude'] = merged['Longitude'].fillna(merged['Longitude'].median())
merged['Peak Discharge Q (cumec)'] = merged['Peak Discharge Q (cumec)'].fillna(merged['Peak Discharge Q (cumec)'].median())
merged['Event Duration (days)'] = merged['Event Duration (days)'].fillna(5.0)

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
        X[i, t, 0] = cum
        X[i, t, 1] = delta
        X[i, t, 2] = pvals[t] / (cum / (t + 1) + 1e-5)
        X[i, t, 3] = api
        X[i, t, 4] = disch
        X[i, t, 5] = duration
        X[i, t, 6] = lat
        X[i, t, 7] = lon
        X[i, t, 8] = dl
        X[i, t, 9] = wl
        X[i, t, 10] = area

gauges = merged['GaugeID'].values
gkf = GroupKFold(n_splits=5)
train_idx, test_idx = next(gkf.split(X, y_excess, groups=gauges))

X_tr, X_te = X[train_idx], X[test_idx]
y_tr, y_te = y_excess[train_idx], y_excess[test_idx]
y_te_abs = y_abs[test_idx]
danger_te = danger[test_idx]

feat_scaler = RobustScaler()
X_tr_s = feat_scaler.fit_transform(X_tr.reshape(-1, F)).reshape(len(X_tr), T, F)
X_te_s = feat_scaler.transform(X_te.reshape(-1, F)).reshape(len(X_te), T, F)

Xf_tr = X_tr_s.reshape(len(X_tr), T * F)
Xf_te = X_te_s.reshape(len(X_te), T * F)

gbm = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
gbm.fit(Xf_tr, y_tr)
yp_excess = gbm.predict(Xf_te)

# R2 on excess (residual)
ss_res_excess = np.sum((yp_excess - y_te)**2)
ss_tot_excess = np.sum((y_te - y_te.mean())**2)
r2_excess = 1 - ss_res_excess / ss_tot_excess
mae_excess = np.abs(yp_excess - y_te).mean()

# R2 on absolute meters
yp_abs = danger_te + yp_excess
ss_res_abs = np.sum((yp_abs - y_te_abs)**2)
ss_tot_abs = np.sum((y_te_abs - y_te_abs.mean())**2)
r2_abs = 1 - ss_res_abs / ss_tot_abs
mae_abs = np.abs(yp_abs - y_te_abs).mean()

print(f"Excess (residual) target: MAE = {mae_excess:.2f} m, R2 = {r2_excess:.4f}")
print(f"Absolute meters target  : MAE = {mae_abs:.2f} m, R2 = {r2_abs:.4f}")
print(f"Mean excess in test     : {y_te.mean():.2f} m, Std excess: {y_te.std():.2f} m")
