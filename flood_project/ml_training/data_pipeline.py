"""
Data Pipeline — FloodSentinel AI (v2)
======================================
Ingests INDOFLOODS + IFI-Impacts datasets and produces:
  - LSTM training sequences: (N, timesteps=10, features=6)
  - Saves processed arrays + scalers to data/processed/
  - Exports a danger_thresholds.json for the hydro_service calibration
"""

import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
RAW   = os.path.join(BASE, "data", "raw")
PROC  = os.path.join(BASE, "data", "processed")
os.makedirs(PROC, exist_ok=True)

INDOFLOODS_DIR = os.path.join(RAW, "indofloods")
IFI_DIR        = os.path.join(RAW, "ifi_impacts")

# ── Config ─────────────────────────────────────────────────────────────
TIMESTEPS = 10   # 10-day cumulative rainfall window (T1d..T10d)
FEATURES  = 6    # features per timestep


def load_all():
    print("[DATA] Loading datasets...")
    flood   = pd.read_csv(os.path.join(INDOFLOODS_DIR, "floodevents_indofloods.csv"))
    precip  = pd.read_csv(os.path.join(INDOFLOODS_DIR, "precipitation_variables_indofloods.csv"))
    meta    = pd.read_csv(os.path.join(INDOFLOODS_DIR, "metadata_indofloods.csv"))
    dfsi    = pd.read_csv(os.path.join(IFI_DIR, "DFSI.csv"))
    print(f"   Flood events : {flood.shape[0]:,}")
    print(f"   Precipitation: {precip.shape[0]:,}")
    print(f"   Gauge meta   : {meta.shape[0]:,}")
    print(f"   DFSI         : {dfsi.shape[0]:,}")

    return flood, precip, meta, dfsi


def build_dfsi_lookup(dfsi: pd.DataFrame) -> dict:
    """State-level DFSI severity lookup (normalized 0-1)."""
    # DFSI has District and State_Name columns
    col = "State_Name" if "State_Name" in dfsi.columns else dfsi.columns[1]
    dfsi_agg = dfsi.groupby(col)["DFSI"].mean().reset_index()
    mn, mx = dfsi_agg["DFSI"].min(), dfsi_agg["DFSI"].max()
    dfsi_agg["DFSI_norm"] = (dfsi_agg["DFSI"] - mn) / (mx - mn + 1e-8)
    return dict(zip(dfsi_agg[col].str.upper().str.strip(), dfsi_agg["DFSI_norm"]))


def build_threshold_export(meta: pd.DataFrame) -> dict:
    """
    Build gauge → danger_threshold mapping for hydro_service calibration.
    Uses actual CWC 'Danger Level' values from INDOFLOODS metadata.
    """
    thresholds = {}
    for _, row in meta.iterrows():
        gid      = str(row["GaugeID"])
        danger   = row.get("Danger Level", np.nan)
        warning  = row.get("Warning Level", np.nan)
        lat      = row.get("Latitude", None)
        lng      = row.get("Longitude", None)
        state    = row.get("State", "")
        river    = row.get("River Name/ Tributory/ SubTributory", "")
        if pd.notna(danger):
            thresholds[gid] = {
                "danger_level_m": float(danger),
                "warning_level_m": float(warning) if pd.notna(warning) else float(danger) * 0.9,
                "lat": float(lat) if pd.notna(lat) else None,
                "lng": float(lng) if pd.notna(lng) else None,
                "state": state,
                "river": river,
            }
    return thresholds


def build_sequences(flood, precip, meta, dfsi_lookup):
    """
    Build LSTM sequences. Each event → one (TIMESTEPS, FEATURES) sample.

    Feature layout per timestep t (t=0 is 1-day antecedent, t=9 is 10-day):
      [0] rainfall_cum_t      — cumulative precip at day t (mm)
      [1] rainfall_delta_t    — day-over-day change in precip (mm)
      [2] peak_discharge      — peak discharge m³/s (static per event)
      [3] danger_level        — CWC danger level (m) from metadata (static)
      [4] event_duration      — duration in days (static)
      [5] severity_index      — normalized DFSI for that state (static)

    Target:
      Peak Flood Level (m) — water level at peak of flood event
    """
    print("\n[PROC] Building sequences...")

    precip_cols = [f"T{i}d" for i in range(1, TIMESTEPS + 1)]

    # Merge on EventID
    merged = flood.merge(precip, on="EventID", how="inner")

    # Extract gauge code from EventID (e.g. INDOFLOODS-gauge-394-1 → INDOFLOODS-gauge-394)
    merged["GaugeID"] = merged["EventID"].str.rsplit("-", n=1).str[0]

    # Join metadata (danger level, state, coords)
    meta_slim = meta[["GaugeID", "Danger Level", "Warning Level", "State",
                       "Latitude", "Longitude"]].copy()
    merged = merged.merge(meta_slim, on="GaugeID", how="left")

    # Map state → DFSI severity
    merged["severity_index"] = merged["State"].str.upper().str.strip().map(dfsi_lookup).fillna(0.5)

    # Fill remaining NaN
    median_discharge = merged["Peak Discharge Q (cumec)"].median()
    median_danger    = merged["Danger Level"].median()
    merged["Peak Discharge Q (cumec)"].fillna(median_discharge, inplace=True)
    merged["Danger Level"].fillna(median_danger, inplace=True)
    merged["Event Duration (days)"].fillna(5.0, inplace=True)

    # Drop rows where target (Peak Flood Level) or precip is NaN
    merged = merged.dropna(subset=["Peak Flood Level (m)"] + precip_cols)
    print(f"   Usable events after cleaning: {len(merged):,}")

    # Build arrays
    N = len(merged)
    X = np.zeros((N, TIMESTEPS, FEATURES), dtype=np.float32)
    y = np.zeros(N, dtype=np.float32)

    for i, (_, row) in enumerate(merged.iterrows()):
        precip_vals = [float(row[c]) for c in precip_cols]
        discharge   = float(row["Peak Discharge Q (cumec)"])
        danger_lv   = float(row["Danger Level"])
        duration    = float(row["Event Duration (days)"])
        severity    = float(row["severity_index"])

        for t in range(TIMESTEPS):
            delta = precip_vals[t] - precip_vals[t-1] if t > 0 else precip_vals[t]
            X[i, t, 0] = precip_vals[t]   # cumulative rainfall
            X[i, t, 1] = delta             # delta rainfall
            X[i, t, 2] = discharge         # peak discharge
            X[i, t, 3] = danger_lv         # danger threshold
            X[i, t, 4] = duration          # event duration
            X[i, t, 5] = severity          # DFSI severity

        y[i] = float(row["Peak Flood Level (m)"])

    print(f"   X: {X.shape}  y: {y.shape}")
    print(f"   Target range: {y.min():.2f}m – {y.max():.2f}m  (mean: {y.mean():.2f}m)")
    return X, y, merged


def normalize_and_split(X, y):
    """MinMax normalize and split 70/15/15."""
    N, T, F = X.shape
    X_flat = X.reshape(-1, F)
    feat_scaler = MinMaxScaler()
    X_norm = feat_scaler.fit_transform(X_flat).reshape(N, T, F).astype(np.float32)

    tgt_scaler = MinMaxScaler()
    y_norm = tgt_scaler.fit_transform(y.reshape(-1, 1)).flatten().astype(np.float32)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_norm, y_norm, test_size=0.30, random_state=42)
    X_va, X_te, y_va, y_te   = train_test_split(X_tmp,  y_tmp,  test_size=0.50, random_state=42)

    return X_tr, X_va, X_te, y_tr, y_va, y_te, feat_scaler, tgt_scaler


def save_all(X_tr, X_va, X_te, y_tr, y_va, y_te, feat_scaler, tgt_scaler, thresholds):
    print("\n[SAVE] Saving processed outputs...")
    for name, arr in [("X_train", X_tr), ("X_val", X_va), ("X_test", X_te),
                      ("y_train", y_tr), ("y_val", y_va), ("y_test", y_te)]:
        np.save(os.path.join(PROC, f"{name}.npy"), arr)

    joblib.dump(feat_scaler, os.path.join(PROC, "feature_scaler.pkl"))
    joblib.dump(tgt_scaler,  os.path.join(PROC, "target_scaler.pkl"))

    with open(os.path.join(PROC, "danger_thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    # Save feature metadata
    feature_meta = {
        "timesteps": TIMESTEPS,
        "features": FEATURES,
        "feature_names": [
            "rainfall_cumulative_mm",
            "rainfall_delta_mm",
            "peak_discharge_cumec",
            "danger_level_m",
            "event_duration_days",
            "dfsi_severity_index"
        ],
        "target": "peak_flood_level_m",
        "train_size": len(X_tr),
        "val_size":   len(X_va),
        "test_size":  len(X_te),
        "source_datasets": ["INDOFLOODS (Zenodo 14584655)", "IFI-Impacts DFSI (Zenodo 16994648)"]
    }
    with open(os.path.join(PROC, "feature_metadata.json"), "w") as f:
        json.dump(feature_meta, f, indent=2)

    print(f"   Train: {len(X_tr):,} | Val: {len(X_va):,} | Test: {len(X_te):,}")
    print(f"   Danger thresholds exported: {len(thresholds):,} gauges")
    print(f"   Saved to: {PROC}")


def run():
    print("=" * 65)
    print("  FloodSentinel AI - Real Data Pipeline")
    print("  Sources: INDOFLOODS + IFI-Impacts DFSI (IIT Delhi)")
    print("=" * 65)

    flood, precip, meta, dfsi = load_all()
    dfsi_lookup = build_dfsi_lookup(dfsi)
    thresholds  = build_threshold_export(meta)

    X, y, _ = build_sequences(flood, precip, meta, dfsi_lookup)
    X_tr, X_va, X_te, y_tr, y_va, y_te, feat_scaler, tgt_scaler = normalize_and_split(X, y)
    save_all(X_tr, X_va, X_te, y_tr, y_va, y_te, feat_scaler, tgt_scaler, thresholds)

    print("\n[OK] Pipeline complete!")


if __name__ == "__main__":
    run()
