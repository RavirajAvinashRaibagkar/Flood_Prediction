"""
UNet Flood Segmentation Model & Inference Adapter
=================================================
Loads the trained PyTorch U-Net (ResNet-34 backbone) checkpoint trained on
the official Sen1Floods11 Sentinel-1 SAR dataset.

Applies the exact same training pipeline preprocessing:
  1. 2-band Sentinel-1 (VV, VH) in dB
  2. Synthesizes 3rd band: (VV - VH) dB
  3. NaN / Inf sanitization (DB_CLIP_LOW)
  4. Train-only normalization (norm_stats.json)
  5. Calibrated threshold: 0.5517 (from validation calibration)
"""

import io
import json
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Defaults matching training pipeline
DB_CLIP_LOW = -35.0
DB_CLIP_HIGH = 5.0
DEFAULT_THRESHOLD = 0.5517

# Fallback normalization stats fitted strictly on train split
DEFAULT_NORM_STATS = {
    "vv_db": {"mean": -10.3807, "std": 3.6641, "clip_low": -35.0, "clip_high": 5.0},
    "vh_db": {"mean": -17.1824, "std": 4.2987, "clip_low": -35.0, "clip_high": 5.0},
    "vv_minus_vh_db": {"mean": 6.4715, "std": 2.7667, "clip_low": -35.0, "clip_high": 5.0},
}

HAS_TORCH = False
try:
    import torch
    import segmentation_models_pytorch as smp
    HAS_TORCH = True
except Exception as e:
    logger.warning(f"PyTorch or segmentation_models_pytorch not available: {e}")

HAS_RASTERIO = False
try:
    import rasterio  # noqa: F401  (availability probe)
    from rasterio.io import MemoryFile
    from rasterio.warp import transform_bounds
    HAS_RASTERIO = True
except Exception as e:
    logger.warning(f"rasterio not available: {e}")


def _find_checkpoint() -> Optional[Path]:
    """Locate the trained unet_resnet34_seed42_best.pt checkpoint."""
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / ".." / ".." / "ml_training" / "saved_models" / "unet_resnet34_seed42_best.pt",
        base_dir / ".." / "saved_models" / "unet_resnet34_seed42_best.pt",
        Path("flood_project/ml_training/saved_models/unet_resnet34_seed42_best.pt"),
        Path("saved_models/unet_resnet34_seed42_best.pt"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def _find_norm_stats() -> dict:
    """Load train-only normalization statistics."""
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / ".." / ".." / "ml_training" / "data" / "seg_dataset" / "norm_stats.json",
        base_dir / ".." / "data" / "seg_dataset" / "norm_stats.json",
        Path("flood_project/ml_training/data/seg_dataset/norm_stats.json"),
    ]
    for c in candidates:
        if c.exists():
            try:
                with open(c) as f:
                    return json.load(f)
            except Exception:
                pass
    return DEFAULT_NORM_STATS


def _find_threshold() -> float:
    """Load validation-calibrated threshold if results json exists."""
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / ".." / ".." / "ml_training" / "saved_models" / "unet_resnet34_seed42_results.json",
        base_dir / ".." / "saved_models" / "unet_resnet34_seed42_results.json",
        Path("flood_project/ml_training/saved_models/unet_resnet34_seed42_results.json"),
    ]
    for c in candidates:
        if c.exists():
            try:
                with open(c) as f:
                    data = json.load(f)
                    val_thresh = data.get("splits", {}).get("val", {}).get("threshold")
                    if val_thresh:
                        return float(val_thresh)
            except Exception:
                pass
    return DEFAULT_THRESHOLD


class UNetSegmentationAdapter:
    """
    Inference adapter for the U-Net flood segmentation model.
    """

    def __init__(self, device_str: str = "auto"):
        self.model = None
        self.device = torch.device("cpu")
        self.checkpoint_path = None
        self.norm_stats = _find_norm_stats()
        self.threshold = _find_threshold()
        self.loaded = False

        if not HAS_TORCH:
            logger.warning("UNetSegmentationAdapter initialized without PyTorch.")
            return

        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        self._init_model()

    def _init_model(self):
        try:
            self.checkpoint_path = _find_checkpoint()
            self.model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,  # We load trained checkpoint weights
                in_channels=3,
                classes=1,
                activation=None,
            )

            if self.checkpoint_path and self.checkpoint_path.exists():
                state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded U-Net checkpoint from {self.checkpoint_path} onto {self.device}")
                self.loaded = True
            else:
                logger.warning("U-Net checkpoint unet_resnet34_seed42_best.pt not found. Running with uninitialized weights.")

            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to initialize U-Net model: {e}")
            self.model = None
            self.loaded = False

    def is_ready(self) -> bool:
        return self.model is not None and self.loaded

    def preprocess_s1_bands(self, s1_bands: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Takes raw [2, H, W] or [bands, H, W] SAR array in dB (or linear power).
        Builds 3-channel [3, H, W] array (VV, VH, VV-VH) and normalizes using train stats.
        Returns (torch.Tensor [1, 3, H, W], sanitized 3-ch numpy array).
        """
        if s1_bands.ndim == 2:
            s1_bands = np.stack([s1_bands, s1_bands], axis=0)
        elif s1_bands.shape[0] == 1:
            s1_bands = np.concatenate([s1_bands, s1_bands], axis=0)

        # Keep first 2 bands (VV, VH)
        vv = s1_bands[0].astype(np.float32)
        vh = s1_bands[1].astype(np.float32)

        # If data is in linear power (all positive, wide range), convert to dB: 10 * log10(val + 1e-7)
        if np.nanpercentile(vv, 90) > 0.0 and np.nanpercentile(vv, 10) >= 0.0:
            vv = 10.0 * np.log10(np.maximum(vv, 1e-6))
            vh = 10.0 * np.log10(np.maximum(vh, 1e-6))

        # Sanitize NaNs and Infs to low dB
        vv = np.nan_to_num(vv, nan=DB_CLIP_LOW, posinf=DB_CLIP_HIGH, neginf=DB_CLIP_LOW)
        vh = np.nan_to_num(vh, nan=DB_CLIP_LOW, posinf=DB_CLIP_HIGH, neginf=DB_CLIP_LOW)
        diff = vv - vh

        arr3 = np.stack([vv, vh, diff], axis=0)  # [3, H, W]

        # Normalize with train-only stats
        ch_names = ["vv_db", "vh_db", "vv_minus_vh_db"]
        normed = np.empty_like(arr3)
        for i, ch in enumerate(ch_names):
            st = self.norm_stats.get(ch, DEFAULT_NORM_STATS[ch])
            clipped = np.clip(arr3[i], st["clip_low"], st["clip_high"])
            normed[i] = (clipped - st["mean"]) / (st["std"] + 1e-6)

        tensor = torch.from_numpy(normed).unsqueeze(0).to(self.device)  # [1, 3, H, W]
        return tensor, arr3

    def predict_mask(self, s1_tensor: torch.Tensor, original_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs U-Net forward pass. Handles arbitrary H, W via padding to multiple of 32.
        Returns:
            binary_mask: [H, W] int (0=non-water, 1=flood)
            prob_map: [H, W] float in [0, 1]
        """
        orig_h, orig_w = original_hw
        # Pad to multiple of 32 for U-Net encoder-decoder stages
        pad_h = (32 - (orig_h % 32)) % 32
        pad_w = (32 - (orig_w % 32)) % 32

        if pad_h > 0 or pad_w > 0:
            s1_tensor = torch.nn.functional.pad(s1_tensor, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.no_grad():
            logits = self.model(s1_tensor)
            if logits.dim() == 4:
                logits = logits.squeeze(1)  # [1, H_pad, W_pad]
            probs = torch.sigmoid(logits)[0]  # [H_pad, W_pad]

        # Crop back to original dimensions
        probs = probs[:orig_h, :orig_w].cpu().numpy()
        binary_mask = (probs >= self.threshold).astype(np.uint8)

        return binary_mask, probs

    def segment_geotiff(self, file_bytes: bytes) -> Dict[str, Any]:
        """
        End-to-end inference from uploaded GeoTIFF file bytes:
          - Reads SAR GeoTIFF with rasterio
          - Preprocesses (VV, VH, diff) + normalization
          - Runs U-Net inference
          - Returns flood stats and base64 visualization layers
        """
        if not HAS_RASTERIO:
            raise RuntimeError("rasterio is required for GeoTIFF segmentation")
        if not self.is_ready():
            raise RuntimeError("U-Net model checkpoint is not loaded")

        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as src:
                s1_arr = src.read()
                crs = src.crs
                bounds = src.bounds
                width = src.width
                height = src.height

        # Calculate WGS84 lat/lng bounds if projected
        latlng_bounds = None
        if crs and bounds:
            try:
                if crs.to_epsg() != 4326:
                    w, s, e, n = transform_bounds(crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top)
                    latlng_bounds = [[s, w], [n, e]]
                else:
                    latlng_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
            except Exception as e:
                logger.debug(f"Could not transform bounds to WGS84: {e}")

        # Preprocess
        tensor, raw_3ch = self.preprocess_s1_bands(s1_arr)
        mask, probs = self.predict_mask(tensor, (height, width))

        # Statistics
        flood_pixels = int(np.sum(mask == 1))
        total_pixels = int(mask.size)
        flood_frac = round(flood_pixels / max(total_pixels, 1), 4)

        # Generate Base64 RGBA Flood Overlay PNG (Cyan/Blue #00b4d8 for flood, transparent elsewhere)
        overlay_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        # Flood pixels: RGB (0, 180, 240) + Alpha 200
        overlay_rgba[mask == 1] = [0, 180, 240, 200]
        # Low-prob water pixels (0.35 to threshold): fainter blue for hazard indication
        hazard_mask = (probs >= 0.35) & (mask == 0)
        overlay_rgba[hazard_mask] = [30, 120, 200, 90]

        img_overlay = Image.fromarray(overlay_rgba, mode="RGBA")
        overlay_buf = io.BytesIO()
        img_overlay.save(overlay_buf, format="PNG")
        overlay_b64 = base64.b64encode(overlay_buf.getvalue()).decode("utf-8")

        # Generate Base64 SAR composite preview (VV dB visualized as grayscale 0-255)
        vv = raw_3ch[0]
        vv_norm = np.clip((vv - (-30.0)) / (5.0 - (-30.0)), 0.0, 1.0)
        sar_preview = (vv_norm * 255.0).astype(np.uint8)
        img_sar = Image.fromarray(sar_preview, mode="L")
        sar_buf = io.BytesIO()
        img_sar.save(sar_buf, format="PNG")
        sar_b64 = base64.b64encode(sar_buf.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "model": "U-Net (ResNet34 backbone, Sen1Floods11)",
            "device": str(self.device),
            "threshold": self.threshold,
            "threshold_used": self.threshold,
            "dimensions": {"width": width, "height": height, "0": height, "1": width},
            "flood_detected": bool(flood_pixels > 0),
            "flood_pixel_count": flood_pixels,
            "flood_pixels": flood_pixels,
            "total_pixels": total_pixels,
            "flood_area_fraction": flood_frac,
            "flood_percentage": round(flood_frac * 100, 2),
            "confidence": round(float(np.mean(probs[mask == 1])) if flood_pixels > 0 else 0.85, 4),
            "bounds": latlng_bounds,
            "mask_overlay_b64": f"data:image/png;base64,{overlay_b64}",
            "sar_preview_b64": f"data:image/png;base64,{sar_b64}",
            "summary": {
                "max_flood_probability": round(float(np.max(probs)), 4),
                "mean_flood_probability": round(float(np.mean(probs)), 4),
            },
        }

    def segment_image_stream(self, file_bytes: bytes, filename: str = "image.png") -> Dict[str, Any]:
        """
        Accepts either GeoTIFF or standard image file bytes.
        If GeoTIFF, uses rasterio full SAR pipeline.
        If standard image (PNG/JPG), converts grayscale to synthetic VV/VH dB proxy.
        """
        # If GeoTIFF or has TIF signature
        is_tif = filename.lower().endswith((".tif", ".tiff")) or file_bytes[:4] in (b"II*\x00", b"MM\x00*")

        if is_tif and HAS_RASTERIO:
            return self.segment_geotiff(file_bytes)

        # Standard image fallback: convert to synthetic SAR proxy
        img = Image.open(io.BytesIO(file_bytes)).convert("L")
        arr = np.array(img, dtype=np.float32)
        height, width = arr.shape

        # Scale 0-255 to typical SAR dB range [-30, 0]
        vv = (arr / 255.0) * 30.0 - 30.0
        vh = vv - 6.0
        s1_synthetic = np.stack([vv, vh], axis=0)

        tensor, raw_3ch = self.preprocess_s1_bands(s1_synthetic)
        mask, probs = self.predict_mask(tensor, (height, width))

        flood_pixels = int(np.sum(mask == 1))
        total_pixels = int(mask.size)
        flood_frac = round(flood_pixels / max(total_pixels, 1), 4)

        overlay_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        overlay_rgba[mask == 1] = [0, 180, 240, 200]
        img_overlay = Image.fromarray(overlay_rgba, mode="RGBA")
        overlay_buf = io.BytesIO()
        img_overlay.save(overlay_buf, format="PNG")
        overlay_b64 = base64.b64encode(overlay_buf.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "model": "U-Net (ResNet34 backbone, Sentinel-1 Proxy)",
            "device": str(self.device),
            "threshold": self.threshold,
            "threshold_used": self.threshold,
            "dimensions": {"width": width, "height": height, "0": height, "1": width},
            "flood_detected": bool(flood_pixels > 0),
            "flood_pixel_count": flood_pixels,
            "flood_pixels": flood_pixels,
            "total_pixels": total_pixels,
            "flood_area_fraction": flood_frac,
            "flood_percentage": round(flood_frac * 100, 2),
            "confidence": round(float(np.mean(probs[mask == 1])) if flood_pixels > 0 else 0.85, 4),
            "bounds": None,
            "mask_overlay_b64": f"data:image/png;base64,{overlay_b64}",
            "summary": {
                "max_flood_probability": round(float(np.max(probs)), 4),
                "mean_flood_probability": round(float(np.mean(probs)), 4),
            },
        }


# Singleton adapter instance
_adapter: Optional[UNetSegmentationAdapter] = None

def get_unet_adapter() -> UNetSegmentationAdapter:
    global _adapter
    if _adapter is None:
        _adapter = UNetSegmentationAdapter()
    return _adapter
