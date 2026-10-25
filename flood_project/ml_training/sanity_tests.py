"""
sanity_tests.py  --  FloodSentinel AI  (Phase 1 Audit: Sanity Tests)
=====================================================================
Test A: Train on shuffled labels -> accuracy must drop to chance (~50%)
Test B: Train on tiny subset, evaluate on truly held-out set
Test C: Naive majority-class baseline comparison

Run: python sanity_tests.py
"""

import os
import sys
import numpy as np

BASE  = os.path.dirname(os.path.abspath(__file__))
PROC  = os.path.join(BASE, "data", "processed_v2")
# Fallback to old processed if v2 not yet run
if not os.path.exists(os.path.join(PROC, "X_train.npy")):
    PROC = os.path.join(BASE, "data", "processed")

sys.path.insert(0, os.path.join(BASE, "..", "backend"))

print("[SANITY] Loading data...")
X_train = np.load(os.path.join(PROC, "X_train.npy"))
X_val   = np.load(os.path.join(PROC, "X_val.npy"))
X_test  = np.load(os.path.join(PROC, "X_test.npy"))
y_train = np.load(os.path.join(PROC, "y_train.npy"))
y_val   = np.load(os.path.join(PROC, "y_val.npy"))
y_test  = np.load(os.path.join(PROC, "y_test.npy"))
print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# ============================================================
# SANITY A: Naive baselines  (regression task)
# ============================================================
print("\n" + "=" * 60)
print("  SANITY A: Naive Baselines")
print("=" * 60)

mean_pred = y_train.mean()
median_pred = np.median(y_train)

mean_mae_val   = np.abs(y_val - mean_pred).mean()
median_mae_val = np.abs(y_val - median_pred).mean()
mean_mae_test  = np.abs(y_test - mean_pred).mean()

print(f"  Train mean target: {mean_pred:.4f}")
print(f"  Mean-predictor  MAE on val : {mean_mae_val:.4f}")
print(f"  Median-predictor MAE on val: {median_mae_val:.4f}")
print(f"  Mean-predictor  MAE on test: {mean_mae_test:.4f}")
print()
print("  A good model MUST beat these baselines substantially.")
print("  If model MAE < baseline MAE -> possible data leakage!")

# ============================================================
# SANITY B: Shuffled labels test  (requires sklearn/tensorflow)
# ============================================================
print("\n" + "=" * 60)
print("  SANITY B: Shuffled Labels Test")
print("=" * 60)

try:
    from sklearn.linear_model import Ridge
    import warnings
    warnings.filterwarnings("ignore")

    # Flatten sequences for sklearn
    N_tr, T, F = X_train.shape
    Xf_tr = X_train.reshape(N_tr, T * F)
    Xf_va = X_val.reshape(len(X_val), T * F)

    # Normal Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(Xf_tr, y_train)
    normal_mae = np.abs(ridge.predict(Xf_va) - y_val).mean()
    print(f"  Ridge normal   MAE on val: {normal_mae:.4f}")

    # Shuffled labels Ridge
    rng = np.random.RandomState(999)
    y_shuffled = y_train[rng.permutation(len(y_train))]
    ridge_shuf = Ridge(alpha=1.0)
    ridge_shuf.fit(Xf_tr, y_shuffled)
    shuf_mae = np.abs(ridge_shuf.predict(Xf_va) - y_val).mean()
    print(f"  Ridge shuffled MAE on val: {shuf_mae:.4f}")

    ratio = shuf_mae / (normal_mae + 1e-9)
    print(f"  Ratio (shuf/normal): {ratio:.2f}")
    if ratio < 1.1:
        print("  [LEAKAGE WARNING] Shuffled labels give similar MAE -> labels are predictable from features alone!")
    else:
        print("  [OK] Shuffled labels degrade performance as expected.")

except ImportError:
    print("  [SKIP] sklearn not available for linear baseline.")

# ============================================================
# SANITY C: Small subset train + held-out evaluation
# ============================================================
print("\n" + "=" * 60)
print("  SANITY C: Overfit Check (50 samples train, full test)")
print("=" * 60)

try:
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor

    N_tr, T, F = X_train.shape
    Xf_tr_full = X_train.reshape(N_tr, T * F)
    Xf_te = X_test.reshape(len(X_test), T * F)

    # Use only 50 training samples
    rng2 = np.random.RandomState(42)
    idx50 = rng2.choice(N_tr, size=50, replace=False)
    Xf_50 = Xf_tr_full[idx50]
    y_50  = y_train[idx50]

    ridge_50 = Ridge(alpha=1.0)
    ridge_50.fit(Xf_50, y_50)
    train_mae_50 = np.abs(ridge_50.predict(Xf_50) - y_50).mean()
    test_mae_50  = np.abs(ridge_50.predict(Xf_te) - y_test).mean()

    print(f"  50-sample Ridge train MAE: {train_mae_50:.4f}")
    print(f"  50-sample Ridge test  MAE: {test_mae_50:.4f}")
    gap = test_mae_50 - train_mae_50
    print(f"  Generalization gap (test-train): {gap:+.4f}")
    if gap > 0.05:
        print("  [OVERFIT] Large gap -- model memorizes tiny train set (expected behavior)")
    else:
        print("  [OK] Small gap -- model generalizes from 50 samples")

except ImportError:
    print("  [SKIP] sklearn not available.")

print("\n[SANITY TESTS COMPLETE]")
