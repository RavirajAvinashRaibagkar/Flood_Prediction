"""
sliding_window_inference.py  --  FloodSentinel AI
==================================================
Sliding-window inference with Gaussian blending for full-chip prediction.
Used for full-chip evaluation (val/test/bolivia chips are 512x512).

Blending: Gaussian weights centered on each window to reduce border artifacts.
Overlap: 50% (stride = window_size // 2).
Final prediction: argmax of weighted logit sum.

Usage:
  from sliding_window_inference import sliding_window_predict

  # logits [1, H, W] for full chip
  logits = sliding_window_predict(model, img_tensor, window_size=256, stride=128, device=device)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


def gaussian_weight_map(size: int, sigma_factor: float = 0.25) -> np.ndarray:
    """
    2D Gaussian weight map of shape [size, size], peak=1 at center.
    sigma = size * sigma_factor
    """
    sigma = size * sigma_factor
    center = size / 2.0
    y, x = np.mgrid[0:size, 0:size]
    dist2 = (x - center)**2 + (y - center)**2
    w = np.exp(-dist2 / (2.0 * sigma**2))
    return w.astype(np.float32)


@torch.no_grad()
def sliding_window_predict(
    model: nn.Module,
    img: torch.Tensor,          # [C, H, W] or [1, C, H, W]
    window_size: int = 256,
    stride: Optional[int] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 4,
) -> torch.Tensor:
    """
    Sliding-window inference over a full SAR chip with Gaussian blending.

    Args:
        model:       PyTorch model (eval mode)
        img:         [C, H, W] normalized input tensor (no batch dim)
        window_size: inference window (must match training crop size)
        stride:      step between windows (default = window_size // 2)
        device:      inference device
        batch_size:  how many windows to batch together

    Returns:
        logits: [1, H, W] float32 tensor (raw logits before sigmoid)
    """
    if stride is None:
        stride = window_size // 2

    if img.dim() == 3:
        img = img.unsqueeze(0)   # [1, C, H, W]
    _, C, H, W = img.shape

    if device is None:
        device = next(model.parameters()).device

    img = img.to(device)
    model.eval()

    # Build Gaussian weight map
    gw = torch.from_numpy(gaussian_weight_map(window_size)).to(device)  # [ws, ws]

    # Output accumulators
    logit_sum  = torch.zeros(1, H, W, device=device)
    weight_sum = torch.zeros(1, H, W, device=device)

    # Collect all window coordinates
    windows = []
    y_starts = list(range(0, H - window_size + 1, stride))
    x_starts = list(range(0, W - window_size + 1, stride))
    # Ensure we cover the full image (add trailing window if needed)
    if y_starts and y_starts[-1] + window_size < H:
        y_starts.append(H - window_size)
    if x_starts and x_starts[-1] + window_size < W:
        x_starts.append(W - window_size)
    if H <= window_size:
        y_starts = [0]
    if W <= window_size:
        x_starts = [0]

    for y0 in y_starts:
        for x0 in x_starts:
            windows.append((y0, x0))

    # Batch inference
    for i in range(0, len(windows), batch_size):
        batch_windows = windows[i:i+batch_size]
        patches = []
        for y0, x0 in batch_windows:
            patch = img[:, :, y0:y0+window_size, x0:x0+window_size]
            patches.append(patch)
        batch_tensor = torch.cat(patches, dim=0)   # [B, C, ws, ws]

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            batch_logits = model(batch_tensor)   # [B, 1, ws, ws] or [B, ws, ws]

        if batch_logits.dim() == 4:
            batch_logits = batch_logits.squeeze(1)   # [B, ws, ws]

        for j, (y0, x0) in enumerate(batch_windows):
            logit_sum[0, y0:y0+window_size, x0:x0+window_size]  += batch_logits[j] * gw
            weight_sum[0, y0:y0+window_size, x0:x0+window_size] += gw

    # Normalize by weights
    weight_sum = weight_sum.clamp(min=1e-7)
    logits = logit_sum / weight_sum   # [1, H, W]

    return logits.cpu()


def predict_chip_full(
    model: nn.Module,
    chip_entry: dict,
    norm_stats: dict,
    window_size: int = 256,
    stride: int = 128,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    Full pipeline: load chip -> normalize -> sliding-window inference -> logits + label.

    Returns:
        logits: [1, H, W] float32
        label:  [H, W] int16 (0/1/-1)
    """
    import rasterio
    DB_CLIP_LOW  = -35.0
    DB_CLIP_HIGH =   5.0

    # Load S1 chip
    with rasterio.open(chip_entry["s1_path"]) as src:
        arr = src.read().astype(np.float32)
    if arr.shape[0] == 1:
        arr = np.concatenate([arr, arr], axis=0)
    arr = arr[:2]   # [2, H, W]

    # Build 3-channel input
    vv   = arr[0]
    vh   = arr[1]
    diff = vv - vh
    arr3 = np.stack([vv, vh, diff], axis=0)   # [3, H, W]

    # Normalize
    ch_names = ["vv_db", "vh_db", "vv_minus_vh_db"]
    stats = norm_stats.get("channels", norm_stats)
    arr_norm = np.empty_like(arr3)
    for i, ch in enumerate(ch_names):
        st = stats[ch]
        clipped = np.clip(arr3[i], st["clip_low"], st["clip_high"])
        arr_norm[i] = (clipped - st["mean"]) / (st["std"] + 1e-6)

    img_t = torch.from_numpy(arr_norm).float()   # [3, H, W]

    # Load label
    with rasterio.open(chip_entry["label_path"]) as src:
        label = src.read(1).astype(np.int16)

    # Sliding-window inference
    logits = sliding_window_predict(
        model, img_t, window_size=window_size, stride=stride, device=device
    )

    return logits, label
