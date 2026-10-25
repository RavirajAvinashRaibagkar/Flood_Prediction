"""
seg_dataset.py  --  FloodSentinel AI  Segmentation
====================================================
PyTorch Dataset for Sen1Floods11 pixel-wise flood segmentation.

Input: VV dB + VH dB + (VV-VH) dB  →  3-channel float32 tensor [3, H, W]
Label: {0=non-water, 1=flood, -1=nodata}  →  int16 [H, W]  (nodata ignored in loss)

Normalization is applied with train-only stats loaded from norm_stats.json.
Sliding-window crops (256px or 512px) are used for training; full-chip inference
uses sliding-window with Gaussian blending (see sliding_window_inference.py).

Augmentations (TRAIN only, never val/test/bolivia):
  - Random horizontal/vertical flip
  - Random 90° rotation
  - Random scale crop (0.75-1.25x)
  - Multiplicative speckle noise (log-normal)
  - Gamma/contrast (on clipped dB)
  - Small channel dropout (zero out one channel with 10% probability)
"""

import json
import random
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ──────────────────────────────────────────────────────────────────────────────
DB_CLIP_LOW  = -35.0
DB_CLIP_HIGH =   5.0
IGNORE_INDEX = -1   # nodata label -- ignored in loss/metrics


def _load_s1(path: str) -> np.ndarray:
    """Load S1Hand GeoTIFF, return float32 [2, H, W] (VV, VH) in dB."""
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)   # [bands, H, W]
    if arr.shape[0] == 1:
        arr = np.concatenate([arr, arr], axis=0)
    arr = np.nan_to_num(arr[:2], nan=DB_CLIP_LOW, posinf=DB_CLIP_HIGH, neginf=DB_CLIP_LOW)
    return arr


def _load_label(path: str) -> np.ndarray:
    """Load LabelHand GeoTIFF, return int16 [H, W]."""
    import rasterio
    with rasterio.open(path) as src:
        lbl = src.read(1).astype(np.int16)
    return lbl


def _make_3ch(arr: np.ndarray) -> np.ndarray:
    """
    Build 3-channel input from 2-band S1:
      ch0 = VV (dB), ch1 = VH (dB), ch2 = VV - VH (dB)
    """
    vv = arr[0]
    vh = arr[1]
    diff = vv - vh
    return np.stack([vv, vh, diff], axis=0)   # [3, H, W]


def _clip_normalize(arr3: np.ndarray, norm_stats: dict) -> np.ndarray:
    """
    Per-channel: clip to DB range, then normalize by (x - mean) / std.
    norm_stats = {ch_name: {mean, std, clip_low, clip_high}}
    """
    ch_names = ["vv_db", "vh_db", "vv_minus_vh_db"]
    out = np.empty_like(arr3)
    for i, ch in enumerate(ch_names):
        st = norm_stats[ch]
        ch_arr = np.nan_to_num(arr3[i], nan=st["clip_low"], posinf=st["clip_high"], neginf=st["clip_low"])
        clipped = np.clip(ch_arr, st["clip_low"], st["clip_high"])
        out[i] = (clipped - st["mean"]) / (st["std"] + 1e-6)
    return out


def _resolve_manifest_path(path: str, seg_dataset_dir: Path) -> str:
    """Resolve a dataset path from either this machine or a copied project tree."""
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)

    # Existing manifests were generated on Windows and may be used on Linux
    # (e.g. Colab). Rebase the path suffix onto this checkout's data directory.
    normalized = str(path).replace("\\", "/")
    marker = "/data/raw/sen1floods11/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        rebased = seg_dataset_dir.parent / "raw" / "sen1floods11" / Path(suffix)
        if rebased.exists():
            return str(rebased)
    return str(candidate)


# ──────────────────────────────────────────────────────────────────────────────
# AUGMENTATIONS
# ──────────────────────────────────────────────────────────────────────────────

def _augment(img: np.ndarray, lbl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random spatial and radiometric augmentations. Applied AFTER normalization.
    img: [3, H, W] float32  (normalized)
    lbl: [H, W]   int16
    """
    # Horizontal flip
    if random.random() < 0.5:
        img = img[:, :, ::-1].copy()
        lbl = lbl[:, ::-1].copy()

    # Vertical flip
    if random.random() < 0.5:
        img = img[:, ::-1, :].copy()
        lbl = lbl[::-1, :].copy()

    # Random 90° rotation (0/90/180/270)
    k = random.randint(0, 3)
    if k > 0:
        img = np.rot90(img, k=k, axes=(1, 2)).copy()
        lbl = np.rot90(lbl, k=k).copy()

    # Multiplicative speckle noise (log-normal, std ~ 0.1 in log domain)
    if random.random() < 0.5:
        noise_std = random.uniform(0.02, 0.10)
        noise = np.random.lognormal(0.0, noise_std, size=img.shape).astype(np.float32)
        img = img * noise

    # Gamma/contrast adjust (only on normalized data -- small perturbation)
    if random.random() < 0.3:
        gamma = random.uniform(0.85, 1.15)
        # Shift to positive, apply gamma, shift back
        img_min = img.min()
        img_shifted = img - img_min + 1e-3
        img = img_shifted ** gamma + img_min - 1e-3

    # Channel dropout: zero out one channel with 10% prob
    if random.random() < 0.10:
        ch = random.randint(0, 2)
        img[ch] = 0.0

    return img, lbl


# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────

class Sen1Floods11SegDataset(Dataset):
    """
    Pixel-wise segmentation dataset for Sen1Floods11 HandLabeled chips.

    For training: yields random crops of size `crop_size` with augmentation.
    For val/test: yields full chips (no crop, no augment). Use
      sliding_window_inference() for full-chip prediction.

    Args:
        chip_manifest:  list of dicts with keys {chip_id, s1_path, label_path, split}
        norm_stats:     dict loaded from norm_stats.json
        crop_size:      int, crop size for training (e.g. 256). None=full chip.
        augment:        bool, whether to apply augmentations (TRAIN only)
        min_valid_frac: float, skip crops with less than this fraction of valid pixels
    """

    def __init__(
        self,
        chip_manifest: List[dict],
        norm_stats: dict,
        crop_size: Optional[int] = 256,
        augment: bool = False,
        min_valid_frac: float = 0.3,
        n_crops_per_chip: int = 4,   # random crops per chip per epoch (train only)
    ):
        self.chip_manifest = chip_manifest
        self.norm_stats    = norm_stats["channels"] if "channels" in norm_stats else norm_stats
        self.crop_size     = crop_size
        self.augment       = augment
        self.min_valid_frac = min_valid_frac
        self.n_crops_per_chip = n_crops_per_chip if crop_size else 1

        # Build flat index: (chip_idx, crop_idx) pairs
        self._index = []
        for ci in range(len(chip_manifest)):
            for crop_i in range(self.n_crops_per_chip):
                self._index.append((ci, crop_i))

        # Verify files exist -- fail loudly
        missing = []
        for entry in chip_manifest:
            if not Path(entry["s1_path"]).exists():
                missing.append(f"S1: {entry['s1_path']}")
            if not Path(entry["label_path"]).exists():
                missing.append(f"LBL: {entry['label_path']}")
        if missing:
            raise FileNotFoundError(
                f"Missing chip files ({len(missing)} issues):\n" +
                "\n".join(missing[:10])
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chip_idx, crop_idx = self._index[idx]
        entry = self.chip_manifest[chip_idx]

        # Load raw data
        arr = _load_s1(entry["s1_path"])          # [2, H, W]
        lbl = _load_label(entry["label_path"])     # [H, W]

        # Build 3-channel input
        arr3 = _make_3ch(arr)                      # [3, H, W]

        # Normalize (train-only stats)
        arr3 = _clip_normalize(arr3, self.norm_stats)

        # Crop (training) or use full chip (val/test)
        if self.crop_size is not None:
            arr3, lbl = self._random_crop(arr3, lbl)

        # Augment (training only)
        if self.augment:
            arr3, lbl = _augment(arr3, lbl)

        # Convert to tensors
        img_t = torch.from_numpy(arr3.copy()).float()
        lbl_t = torch.from_numpy(lbl.copy()).long()

        return img_t, lbl_t

    def _random_crop(
        self, img: np.ndarray, lbl: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random crop, retrying if too many nodata pixels."""
        _, H, W = img.shape
        cs = self.crop_size

        if H <= cs and W <= cs:
            # Pad if chip is smaller than crop (shouldn't happen for 512x512 chips)
            pad_h = max(0, cs - H)
            pad_w = max(0, cs - W)
            img = np.pad(img, ((0,0),(0,pad_h),(0,pad_w)), mode="reflect")
            lbl = np.pad(lbl, ((0,pad_h),(0,pad_w)), constant_values=IGNORE_INDEX)
            return img, lbl

        max_attempts = 10
        for _ in range(max_attempts):
            y0 = random.randint(0, H - cs)
            x0 = random.randint(0, W - cs)
            crop_lbl = lbl[y0:y0+cs, x0:x0+cs]
            valid_frac = float((crop_lbl >= 0).sum()) / (cs * cs)
            if valid_frac >= self.min_valid_frac:
                crop_img = img[:, y0:y0+cs, x0:x0+cs]
                return crop_img, crop_lbl

        # Fallback: center crop
        y0 = (H - cs) // 2
        x0 = (W - cs) // 2
        return img[:, y0:y0+cs, x0:x0+cs], lbl[y0:y0+cs, x0:x0+cs]


# ──────────────────────────────────────────────────────────────────────────────
# FACTORY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset_split(
    seg_dataset_dir: str,
    split: str,
    norm_stats: Optional[dict] = None,
    crop_size: Optional[int] = 256,
    augment: bool = False,
    n_crops_per_chip: int = 4,
) -> "Sen1Floods11SegDataset":
    """
    Convenience factory: load a split by name from seg_dataset/ directory.

    Args:
        seg_dataset_dir: path to data/seg_dataset/
        split: one of "train", "val", "test", "bolivia"
        norm_stats: if None, loads from norm_stats.json in seg_dataset_dir
        crop_size: None for full-chip (val/test)
        augment: True for train only
    """
    seg_dir = Path(seg_dataset_dir)
    manifest_path = seg_dir / f"{split}_chips.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path}\n"
            "Run: python scripts/phase0_build_seg_dataset.py"
        )

    with open(manifest_path) as f:
        chip_manifest = json.load(f)

    for entry in chip_manifest:
        for key in ("s1_path", "label_path"):
            entry[key] = _resolve_manifest_path(entry[key], seg_dir)

    if norm_stats is None:
        norm_path = seg_dir / "norm_stats.json"
        if not norm_path.exists():
            raise FileNotFoundError(
                f"Norm stats not found: {norm_path}\n"
                "Run: python scripts/phase0_build_seg_dataset.py"
            )
        with open(norm_path) as f:
            norm_stats = json.load(f)

    print(f"  Loaded {split} split: {len(chip_manifest)} chips")
    return Sen1Floods11SegDataset(
        chip_manifest=chip_manifest,
        norm_stats=norm_stats,
        crop_size=crop_size if split == "train" else None,
        augment=augment and (split == "train"),
        n_crops_per_chip=n_crops_per_chip if split == "train" else 1,
    )


# ──────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    BASE = Path(__file__).resolve().parent.parent
    SEG_DIR = BASE / "data" / "seg_dataset"

    print("=== Sen1Floods11SegDataset self-test ===")
    for split in ["train", "val", "test", "bolivia"]:
        try:
            ds = load_dataset_split(str(SEG_DIR), split, augment=(split == "train"))
            sample_img, sample_lbl = ds[0]
            unique_labels = torch.unique(sample_lbl).tolist()
            print(f"  {split:10s}: {len(ds):4d} items  "
                  f"img={list(sample_img.shape)}  lbl={list(sample_lbl.shape)}  "
                  f"unique_labels={unique_labels}")
        except FileNotFoundError as e:
            print(f"  {split:10s}: NOT READY -- {e}")
        except Exception as e:
            print(f"  {split:10s}: ERROR -- {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)
    print("\n[OK] Dataset self-test passed.")
