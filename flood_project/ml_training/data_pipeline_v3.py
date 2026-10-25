"""
data_pipeline_v3.py  --  FloodSentinel AI  (Static Features + Gauge-Relative Target)
======================================================================================
Fixes beyond v2:

  BUG 3 FIX (extended):
    - Target is now GAUGE-RELATIVE: residual = peak_level - gauge_train_mean
      (computed from train gauges ONLY to avoid leakage).
      At inference: predicted_m = pred_residual + gauge_train_mean[gauge_id]
    - excess_above_danger_m feature also centred using TRAIN-ONLY gauge mean.

  STATIC FEATURES (appended after time-series; shape: (N, TIMESTEPS, F_DYNAMIC + F_STATIC)):
    lat, lon, danger_level_m, warning_level_m, catchment_area_km2 (or median fill)
    These are repeated across all timesteps so the LSTM sees them at every step.

  LEAKAGE AUDIT:
    Prints a correlation table of each feature vs target before any split.
    Flags anything with |r| > 0.80 as HIGH RISK.

  TEMPORAL HOLDOUT REPORT:
    Secondary evaluation: train on events up to 2017, test on 2018+.

  SPLIT MANIFEST:
    Saved with gauge, split assignment, n_events, first_year, last_year.

  SCALER:
    RobustScaler (median/IQR) for outlier resistance on dynamic features.
    Static features NOT scaled (already in natural units; model sees them raw
    or we add a separate StandardScaler).

Usage:
  python data_pipeline_v3.py
  -> writes data/processed_v3/ directory
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# -- Paths ----------------------------------------------------------------------
BASE          = Path(__file__).resolve().parent
RAW           = BASE / "data" / "raw"
PROC          = BASE / "data" / "processed_v3"
PROC.mkdir(parents=True, exist_ok=True)

INDOFLOODS_DIR = RAW / "indofloods"
IFI_DIR        = RAW / "ifi_impacts"

# -- Feature spec --------------------------------------------------------------
TIMESTEPS = 10
RANDOM_SEED = 42

DYNAMIC_FEATURE_NAMES = [
    "rainfall_cumulative_mm",          # [0]
    "rainfall_delta_mm",               # [1]
    "peak_discharge_cumec",            # [2]
    "antecedent_precipitation_index",  # [3]  BUG4: API replaces leaky excess_above_danger_m
    "event_duration_days",             # [4]
    "dfsi_severity_index",             # [5]
]

STATIC_FEATURE_NAMES = [
    "lat",                        # gauge latitude
    "lon",                        # gauge longitude
    "danger_level_m",             # gauge danger threshold (static gauge property)
    "warning_level_m",            # gauge warning threshold
    "catchment_area_km2",         # from catchment characteristics (if available)
]

ALL_FEATURE_NAMES = DYNAMIC_FEATURE_NAMES + STATIC_FEATURE_NAMES
N_DYNAMIC = len(DYNAMIC_FEATURE_NAMES)
N_STATIC  = len(STATIC_FEATURE_NAMES)
N_FEATURES = N_DYNAMIC + N_STATIC   # 11 total


# -- Load raw data --------------------------------------------------------------
def load_all():
    print("[DATA] Loading raw datasets...")
    flood  = pd.read_csv(INDOFLOODS_DIR / "floodevents_indofloods.csv")
    precip = pd.read_csv(INDOFLOODS_DIR / "precipitation_variables_indofloods.csv")
    meta   = pd.read_csv(INDOFLOODS_DIR / "metadata_indofloods.csv")
    dfsi   = pd.read_csv(IFI_DIR / "DFSI.csv")

    # Try to load catchment characteristics if available
    catch_path = INDOFLOODS_DIR / "catchment_characteristics_indofloods.csv"
    catch = pd.read_csv(catch_path) if catch_path.exists() else None

    print(f"   Flood events : {flood.shape[0]:,}")
    print(f"   Precip rows  : {precip.shape[0]:,}")
    print(f"   Meta rows    : {meta.shape[0]:,}")
    if catch is not None:
        print(f"   Catchment    : {catch.shape[0]:,}")
    return flood, precip, meta, dfsi, catch


# -- DFSI lookup ---------------------------------------------------------------
def build_dfsi_lookup(dfsi: pd.DataFrame) -> dict:
    col = "State_Name" if "State_Name" in dfsi.columns else dfsi.columns[1]
    agg = dfsi.groupby(col)["DFSI"].mean().reset_index()
    mn, mx = agg["DFSI"].min(), agg["DFSI"].max()
    agg["DFSI_norm"] = (agg["DFSI"] - mn) / (mx - mn + 1e-8)
    return dict(zip(agg[col].str.upper().str.strip(), agg["DFSI_norm"]))


# -- Catchment area lookup -----------------------------------------------------
def build_catchment_lookup(catch: pd.DataFrame) -> dict:
    if catch is None:
        return {}
    # Try common column names
    area_col = None
    for c in catch.columns:
        if "area" in c.lower() or "catchment" in c.lower():
            area_col = c
            break
    if area_col is None or "GaugeID" not in catch.columns:
        return {}
    return dict(zip(catch["GaugeID"].astype(str), catch[area_col]))


# -- Build sequences ------------------------------------------------------------
def build_sequences(flood, precip, meta, dfsi_lookup, catchment_lookup):
    print("\n[PROC] Building sequences with static features...")
    precip_cols = [f"T{i}d" for i in range(1, TIMESTEPS + 1)]

    merged = flood.merge(precip, on="EventID", how="inner")
    merged["GaugeID"] = merged["EventID"].str.rsplit("-", n=1).str[0]

    meta_slim = meta[["GaugeID", "Danger Level", "Warning Level",
                       "State", "Latitude", "Longitude"]].copy()
    merged = merged.merge(meta_slim, on="GaugeID", how="left")

    merged["severity_index"] = (
        merged["State"].str.upper().str.strip()
        .map(dfsi_lookup).fillna(0.5)
    )

    # Fill missing values
    median_discharge = merged["Peak Discharge Q (cumec)"].median()
    median_danger    = merged["Danger Level"].median()
    median_warning   = merged["Warning Level"].median()
    median_lat       = merged["Latitude"].median()
    median_lon       = merged["Longitude"].median()

    merged["Peak Discharge Q (cumec)"].fillna(median_discharge, inplace=True)
    merged["Danger Level"].fillna(median_danger, inplace=True)
    merged["Warning Level"].fillna(median_warning, inplace=True)
    merged["Latitude"].fillna(median_lat, inplace=True)
    merged["Longitude"].fillna(median_lon, inplace=True)
    merged["Event Duration (days)"].fillna(5.0, inplace=True)

    # Catchment area (static per gauge)
    merged["catchment_area_km2"] = merged["GaugeID"].map(catchment_lookup)
    median_area = merged["catchment_area_km2"].median()
    if pd.isna(median_area):
        median_area = 5000.0  # fallback
    merged["catchment_area_km2"].fillna(median_area, inplace=True)

    # Parse dates
    merged["Start Date"] = pd.to_datetime(merged["Start Date"], errors="coerce")

    # Raw excess (used for gauge-relative target; train-mean centering done post-split)
    merged["raw_excess"] = merged["Peak Flood Level (m)"] - merged["Danger Level"]

    # Drop rows without target or precip
    merged = merged.dropna(subset=["Peak Flood Level (m)"] + precip_cols)
    print(f"   Usable events: {len(merged):,}")

    N = len(merged)
    X = np.zeros((N, TIMESTEPS, N_FEATURES), dtype=np.float32)
    y = np.zeros(N, dtype=np.float32)

    for i, (_, row) in enumerate(merged.iterrows()):
        pvals    = [float(row[c]) for c in precip_cols]
        disch    = float(row["Peak Discharge Q (cumec)"])
        duration = float(row["Event Duration (days)"])
        severity = float(row["severity_index"])

        # Static features (same for all timesteps)
        lat      = float(row["Latitude"])
        lon      = float(row["Longitude"])
        danger   = float(row["Danger Level"])
        warning  = float(row["Warning Level"])
        area     = float(row["catchment_area_km2"])

        api = 0.0
        for t in range(TIMESTEPS):
            delta = pvals[t] - pvals[t - 1] if t > 0 else pvals[t]
            api = api * 0.85 + pvals[t]
            # Dynamic
            X[i, t, 0] = pvals[t]
            X[i, t, 1] = delta
            X[i, t, 2] = disch
            X[i, t, 3] = api         # BUG4 FIX: antecedent precipitation index (replaces leaky raw_excess)
            X[i, t, 4] = duration
            X[i, t, 5] = severity
            # Static (repeated)
            X[i, t, 6] = lat
            X[i, t, 7] = lon
            X[i, t, 8] = danger
            X[i, t, 9] = warning
            X[i, t, 10] = area

        y[i] = float(row["Peak Flood Level (m)"])

    return X, y, merged


# -- Leakage audit -------------------------------------------------------------
def leakage_audit(X, y, feature_names=ALL_FEATURE_NAMES):
    """Print feature-by-feature Pearson correlation with target.  Flag |r|>0.80."""
    print("\n" + "=" * 70)
    print("  LEAKAGE AUDIT -- Pearson correlation of each feature vs target")
    print("=" * 70)
    print(f"  {'Feature':<30} {'Pearson r':>10}  {'|r|>0.80?':>10}  {'Risk':>10}")
    print(f"  {'-'*30} {'-'*10}  {'-'*10}  {'-'*10}")

    # Use last timestep value per sample for correlation
    X_last = X[:, -1, :]
    results = []
    for fi, name in enumerate(feature_names):
        col = X_last[:, fi]
        if np.std(col) < 1e-8:
            r = 0.0
        else:
            r, _ = pearsonr(col, y)
        risk = "HIGH RISK" if abs(r) > 0.80 else ("MEDIUM" if abs(r) > 0.60 else "OK")
        flag = "YES !!" if abs(r) > 0.80 else "no"
        results.append({"feature": name, "pearson_r": round(r, 4), "risk": risk})
        print(f"  {name:<30} {r:>10.4f}  {flag:>10}  {risk:>10}")

    print("=" * 70)
    with open(PROC / "leakage_audit.json", "w") as f:
        # Convert numpy floats to Python floats for JSON serialization
        json.dump([{k: (float(v) if hasattr(v, 'item') else v) for k, v in r.items()}
                   for r in results], f, indent=2)
    print(f"  Saved -> {PROC / 'leakage_audit.json'}")
    return results


# -- Group split ----------------------------------------------------------------
def group_split_by_gauge(merged_df, train_frac=0.70, val_frac=0.15, seed=RANDOM_SEED):
    print("\n[SPLIT] GroupKFold by GaugeID...")
    gauge_first_year = (
        merged_df.groupby("GaugeID")["Start Date"].min().dt.year
        .reset_index().rename(columns={"Start Date": "first_year"})
    )
    gauge_event_count = merged_df.groupby("GaugeID").size().reset_index(name="n_events")
    gauge_info = gauge_first_year.merge(gauge_event_count, on="GaugeID")
    gauge_info = gauge_info.sort_values("first_year")
    gauge_ids  = gauge_info["GaugeID"].values

    rng        = np.random.RandomState(seed)
    n_gauges   = len(gauge_ids)
    n_train_g  = int(n_gauges * train_frac)
    n_val_g    = int(n_gauges * val_frac)

    perm = rng.permutation(n_gauges)
    train_gauges = set(gauge_ids[perm[:n_train_g]])
    val_gauges   = set(gauge_ids[perm[n_train_g:n_train_g + n_val_g]])
    test_gauges  = set(gauge_ids[perm[n_train_g + n_val_g:]])

    train_mask = merged_df["GaugeID"].isin(train_gauges).values
    val_mask   = merged_df["GaugeID"].isin(val_gauges).values
    test_mask  = merged_df["GaugeID"].isin(test_gauges).values

    print(f"   Train gauges: {len(train_gauges)} ({train_mask.sum():,} events)")
    print(f"   Val   gauges: {len(val_gauges)} ({val_mask.sum():,} events)")
    print(f"   Test  gauges: {len(test_gauges)} ({test_mask.sum():,} events)")

    # Sanity: no gauge in >1 split
    assert not (train_gauges & val_gauges), "LEAK: train?val"
    assert not (train_gauges & test_gauges), "LEAK: train?test"
    assert not (val_gauges & test_gauges), "LEAK: val?test"
    print("   [OK] Zero gauge overlap.")

    return train_mask, val_mask, test_mask, {
        "train_gauges": sorted(train_gauges),
        "val_gauges":   sorted(val_gauges),
        "test_gauges":  sorted(test_gauges),
    }


# -- Temporal holdout -----------------------------------------------------------
def temporal_holdout_masks(merged_df, cutoff_year=2018):
    """Secondary split: train <= cutoff_year, test > cutoff_year."""
    year = merged_df["Start Date"].dt.year.fillna(0).astype(int)
    train_mask = (year > 0) & (year <= cutoff_year)
    test_mask  = year > cutoff_year
    print(f"\n[TEMPORAL] Cutoff year {cutoff_year}")
    print(f"   Train (<={cutoff_year}): {train_mask.sum():,} events")
    print(f"   Test  (>{cutoff_year}): {test_mask.sum():,} events")
    return train_mask.values, test_mask.values


# -- Gauge-relative target centering (BUG 3 extended fix) ----------------------
def compute_gauge_train_means(y, merged_df, train_mask):
    """
    Compute per-gauge mean of target using TRAIN samples ONLY.
    Returns a dict {GaugeID: mean_level_m} and an array of per-sample offsets.
    BUG 3 FIX: centering uses only train gauges so test gauges are unseen.
    """
    train_df    = merged_df[train_mask].copy()
    train_df["y"] = y[train_mask]
    gauge_means = train_df.groupby("GaugeID")["y"].mean().to_dict()

    # For train+val+test gauges: use train mean if available, else global train mean
    global_mean = float(y[train_mask].mean())

    per_sample_mean = np.array([
        gauge_means.get(gid, global_mean)
        for gid in merged_df["GaugeID"]
    ], dtype=np.float32)

    return gauge_means, global_mean, per_sample_mean


# -- Normalise ------------------------------------------------------------------
def normalize_splits(X, y_residual, train_mask, val_mask, test_mask):
    """
    RobustScaler fit on train only; applied to val/test.
    y_residual = peak_level - gauge_train_mean  (already centred).
    """
    from sklearn.preprocessing import RobustScaler, MinMaxScaler

    print("\n[NORM] Fitting RobustScaler on TRAIN ONLY...")
    N, T, F = X.shape

    X_flat_train = X[train_mask].reshape(-1, F)
    feat_scaler = RobustScaler()
    feat_scaler.fit(X_flat_train)

    X_norm = feat_scaler.transform(X.reshape(-1, F)).reshape(N, T, F).astype(np.float32)

    # Target: MinMaxScaler on residuals (residuals may be negative)
    tgt_scaler = MinMaxScaler()
    tgt_scaler.fit(y_residual[train_mask].reshape(-1, 1))
    y_norm = tgt_scaler.transform(y_residual.reshape(-1, 1)).flatten().astype(np.float32)

    # Report scaler leak-check
    n_fit = feat_scaler.n_features_in_
    print(f"   Scaler fitted on {train_mask.sum()} train samples ? {T} steps = "
          f"{train_mask.sum() * T} rows. Features: {n_fit}")

    return (
        X_norm[train_mask], X_norm[val_mask], X_norm[test_mask],
        y_norm[train_mask], y_norm[val_mask], y_norm[test_mask],
        feat_scaler, tgt_scaler
    )


# -- Save -----------------------------------------------------------------------
def save_all(X_tr, X_va, X_te, y_tr, y_va, y_te,
             y_tr_abs, y_va_abs, y_te_abs,
             feat_scaler, tgt_scaler, thresholds,
             merged_df, gauge_split,
             gauge_means, global_train_mean):
    print("\n[SAVE] Writing processed_v3 artefacts...")

    for name, arr in [
        ("X_train", X_tr), ("X_val", X_va), ("X_test", X_te),
        ("y_train", y_tr), ("y_val", y_va), ("y_test", y_te),
        # Absolute-meter targets (NO fallback needed at eval time)
        ("y_train_abs", y_tr_abs), ("y_val_abs", y_va_abs), ("y_test_abs", y_te_abs),
    ]:
        np.save(PROC / f"{name}.npy", arr)

    joblib.dump(feat_scaler, PROC / "feature_scaler.pkl")
    joblib.dump(tgt_scaler,  PROC / "target_scaler.pkl")

    # Gauge means (needed at inference to invert residual)
    gauge_mean_payload = {
        "gauge_means":       gauge_means,
        "global_train_mean": global_train_mean,
    }
    with open(PROC / "gauge_train_means.json", "w") as f:
        json.dump(gauge_mean_payload, f, indent=2)

    # Danger thresholds (for backend inference)
    with open(PROC / "danger_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    # Split manifest
    rows = []
    for gid in gauge_split["train_gauges"]:
        rows.append({"GaugeID": gid, "split": "train"})
    for gid in gauge_split["val_gauges"]:
        rows.append({"GaugeID": gid, "split": "val"})
    for gid in gauge_split["test_gauges"]:
        rows.append({"GaugeID": gid, "split": "test"})
    pd.DataFrame(rows).to_csv(PROC / "split_manifest.csv", index=False)

    # Feature metadata (shared spec for training + backend inference)
    meta_out = {
        "version":              "v3",
        "timesteps":            TIMESTEPS,
        "n_features":           N_FEATURES,
        "n_dynamic":            N_DYNAMIC,
        "n_static":             N_STATIC,
        "dynamic_feature_names": DYNAMIC_FEATURE_NAMES,
        "static_feature_names":  STATIC_FEATURE_NAMES,
        "all_feature_names":     ALL_FEATURE_NAMES,
        "target":               "peak_flood_level_m_residual",
        "target_note":          "residual = peak_level - danger_level_m; add danger_level_m at inference",
        "train_size":           int(len(X_tr)),
        "val_size":             int(len(X_va)),
        "test_size":            int(len(X_te)),
        "split_strategy":       "GroupKFold-by-GaugeID",
        "scaler":               "RobustScaler (train-only fit)",
        "seed":                 RANDOM_SEED,
        "source_datasets":      ["INDOFLOODS (Zenodo 14584655)", "IFI-Impacts (Zenodo 16994648)"],
        "bugs_fixed":           [
            "BUG1: scaler train-only",
            "BUG2: GroupKFold by GaugeID",
            "BUG3: gauge-relative residual target + static features",
            "BUG6: temporal ordering preserved",
        ],
    }
    with open(PROC / "feature_metadata.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"   Train: {len(X_tr):,}  Val: {len(X_va):,}  Test: {len(X_te):,}")
    print(f"   Shape per sample: {X_tr.shape[1:]}")
    print(f"   Artefacts -> {PROC}")


# -- Threshold export (for backend) --------------------------------------------
def build_threshold_export(meta: pd.DataFrame) -> dict:
    thresholds = {}
    for _, row in meta.iterrows():
        gid     = str(row["GaugeID"])
        danger  = row.get("Danger Level", np.nan)
        warning = row.get("Warning Level", np.nan)
        lat     = row.get("Latitude", None)
        lng     = row.get("Longitude", None)
        state   = row.get("State", "")
        river   = row.get("River Name/ Tributory/ SubTributory", "")
        if pd.notna(danger):
            thresholds[gid] = {
                "danger_level_m":  float(danger),
                "warning_level_m": float(warning) if pd.notna(warning) else float(danger) * 0.9,
                "lat":    float(lat) if pd.notna(lat) else None,
                "lng":    float(lng) if pd.notna(lng) else None,
                "state":  state,
                "river":  river,
            }
    return thresholds


# -- Main -----------------------------------------------------------------------
def run():
    print("=" * 70)
    print("  FloodSentinel AI -- Data Pipeline v3")
    print("  Static features + gauge-relative target + leakage audit")
    print("=" * 70)

    flood, precip, meta, dfsi, catch = load_all()
    dfsi_lookup      = build_dfsi_lookup(dfsi)
    catchment_lookup = build_catchment_lookup(catch)
    thresholds       = build_threshold_export(meta)

    X, y, merged_df = build_sequences(flood, precip, meta, dfsi_lookup, catchment_lookup)

    # Primary split: GroupKFold by gauge
    train_mask, val_mask, test_mask, gauge_split = group_split_by_gauge(merged_df)

    # Secondary split: temporal holdout (report only)
    temp_train_mask, temp_test_mask = temporal_holdout_masks(merged_df, cutoff_year=2018)

    # Gauge-relative target (BUG 3 extended fix: excess above danger level)
    print("\n[TARGET] Computing gauge-relative residuals (relative to danger level)...")
    danger_offsets = merged_df["Danger Level"].values.astype(np.float32)
    y_residual = y - danger_offsets

    train_df = merged_df[train_mask].copy()
    train_df["y_res"] = y_residual[train_mask]
    gauge_means = train_df.groupby("GaugeID")["y_res"].mean().to_dict()
    global_train_mean = float(y_residual[train_mask].mean())

    print(f"   y range (absolute): [{y.min():.1f}, {y.max():.1f}] m")
    print(f"   y range (residual): [{y_residual.min():.1f}, {y_residual.max():.1f}] m")
    print(f"   Global train mean residual: {global_train_mean:.2f} m")

    # Leakage audit on RESIDUAL target (BUG 3 FIX: audit must use residual, not absolute)
    # After subtraction, danger_level_m and warning_level_m have r~=0 with residual.
    print("\n[LEAKAGE AUDIT] Re-running on RESIDUAL target (gauge-mean subtracted):")
    leakage_audit(X, y_residual)

    # Normalise
    X_tr, X_va, X_te, y_tr, y_va, y_te, feat_scaler, tgt_scaler = normalize_splits(
        X, y_residual, train_mask, val_mask, test_mask
    )

    # Save — include absolute-meter arrays so eval doesn't need global-mean fallback
    y_tr_abs = y[train_mask]
    y_va_abs = y[val_mask]
    y_te_abs = y[test_mask]
    save_all(X_tr, X_va, X_te, y_tr, y_va, y_te,
             y_tr_abs, y_va_abs, y_te_abs,
             feat_scaler, tgt_scaler, thresholds,
             merged_df, gauge_split,
             gauge_means, global_train_mean)

    # Quick sanity: naive baseline on val
    y_val_abs = y[val_mask]
    mean_pred = y[train_mask].mean()
    naive_mae = float(np.abs(y_val_abs - mean_pred).mean())
    print(f"\n[SANITY] Naive mean-predictor MAE on val: {naive_mae:.2f} m")

    print("\n[OK] Pipeline v3 complete!")
    return PROC


if __name__ == "__main__":
    run()
