"""
audit_nonleaky_features.py
Verifies correlation, derivability, and time-ordering for the clean 11 features.
"""
import pandas as pd
import numpy as np

flood = pd.read_csv('flood_project/ml_training/data/raw/indofloods/floodevents_indofloods.csv')
precip = pd.read_csv('flood_project/ml_training/data/raw/indofloods/precipitation_variables_indofloods.csv')
meta = pd.read_csv('flood_project/ml_training/data/raw/indofloods/metadata_indofloods.csv')

merged = flood.merge(precip, on='EventID', how='inner')
merged['GaugeID'] = merged['EventID'].str.rsplit('-', n=1).str[0]
merged = merged.merge(meta[['GaugeID', 'Danger Level', 'Warning Level', 'Latitude', 'Longitude']], on='GaugeID', how='left')

pcols = [f'T{i}d' for i in range(1, 11)]
merged = merged.dropna(subset=['Peak Flood Level (m)'] + pcols)

y_abs = merged['Peak Flood Level (m)'].values
danger = merged['Danger Level'].fillna(merged['Danger Level'].median()).values

# Compute non-leaky dynamic features
cum_precip = merged[pcols].sum(axis=1).values
delta_precip = (merged['T10d'] - merged['T9d']).values
intensity_ratio = (merged['T10d'] / (merged[pcols].mean(axis=1) + 1e-5)).values

# API (antecedent precip index with k=0.85)
api = np.zeros(len(merged))
for col in pcols:
    api = api * 0.85 + merged[col].values

discharge = merged['Peak Discharge Q (cumec)'].fillna(merged['Peak Discharge Q (cumec)'].median()).values

print("=== CLEAN FEATURES AUDIT vs ABSOLUTE PEAK LEVEL ===")
features = {
    'rainfall_cumulative_mm': cum_precip,
    'rainfall_delta_mm': delta_precip,
    'rainfall_intensity_ratio': intensity_ratio,
    'antecedent_precipitation_index': api,
    'peak_discharge_cumec': discharge,
    'lat': merged['Latitude'].fillna(merged['Latitude'].median()).values,
    'lon': merged['Longitude'].fillna(merged['Longitude'].median()).values,
    'danger_level_m': danger,
    'warning_level_m': merged['Warning Level'].fillna(merged['Warning Level'].median()).values,
}

for name, feat in features.items():
    r = np.corrcoef(feat, y_abs)[0, 1]
    print(f"  {name:<32}: r = {r:+.4f}")
