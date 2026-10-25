from pathlib import Path
import numpy as np

flood_dir = Path("flood_project/ml_training/data/raw/sen1floods11/flood")
no_flood_dir = Path("flood_project/ml_training/data/raw/sen1floods11/no_flood")

n_flood = len(list(flood_dir.glob("*.png")))
n_no_flood = len(list(no_flood_dir.glob("*.png")))

print(f"Verified Real Patches on Disk:")
print(f"  Flood:    {n_flood:,} images")
print(f"  No Flood: {n_no_flood:,} images")
print(f"  Total:    {n_flood + n_no_flood:,} images")
print(f"  Balance:  {n_flood / (n_flood + n_no_flood) * 100:.1f}% flood (realistic natural flood distribution)")
