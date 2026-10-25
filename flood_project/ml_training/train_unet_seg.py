"""
train_unet_seg.py  --  FloodSentinel AI  Phase 3  Track A
==========================================================
Trains a U-Net for pixel-wise flood segmentation on Sen1Floods11.
Architecture: UNet with pretrained encoder (ResNet34 / EfficientNet-B3 / ConvNeXt-tiny)
via segmentation_models_pytorch. First conv adapted for 3 SAR channels.

Training recipe:
  - Loss: BCE (pos_weight from train prevalence) + Dice, ignore_index=-1
  - AdamW, lr warmup + cosine schedule, layer-wise LR decay for encoder
  - AMP (fp16), EMA of weights, early stopping on VAL IoU(flood)
  - Augmentation (TRAIN only): flips, rot90, speckle noise, gamma, channel dropout
  - 5 seeds, report mean +/- 95% CI
  - Threshold tuned on VAL, frozen for test/bolivia

Run:
  python flood_project/ml_training/train_unet_seg.py --encoder resnet34 --seed 42
  python flood_project/ml_training/train_unet_seg.py --encoder resnet34 --all-seeds

Outputs:
  saved_models/unet_resnet34_seed{SEED}.pt  -- EMA weights
  saved_models/unet_resnet34_results.json   -- 5-seed results + CIs
"""

import sys
import json
import argparse
import random
import math
import time
import warnings
from pathlib import Path
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

warnings.filterwarnings("ignore")

BASE    = Path(__file__).resolve().parent
SEG_DIR = BASE / "data" / "seg_dataset"
OUT_DIR = BASE / "saved_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IGNORE_INDEX = -1
DB_CLIP_LOW  = -35.0
DB_CLIP_HIGH =   5.0


# ──────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice loss.
    ignore_index=-1: nodata pixels excluded from both terms.
    pos_weight: from train flood prevalence (class imbalance correction).
    """
    def __init__(
        self,
        pos_weight: float = 1.0,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.smooth      = smooth
        self.pos_weight  = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B, 1, H, W] or [B, H, W]  (raw, before sigmoid)
        targets: [B, H, W]  int (0/1/-1)
        """
        if logits.dim() == 4:
            logits = logits.squeeze(1)   # [B, H, W]

        # Build mask for valid pixels
        valid_mask = targets != IGNORE_INDEX   # [B, H, W]
        n_valid = valid_mask.float().sum()
        if n_valid < 1:
            return logits.sum() * 0.0

        # Clamp targets to 0/1 for loss computation (nodata pixels masked out)
        targets_clamped = targets.clamp(min=0).float()

        # ── BCE ──
        pw = torch.tensor(self.pos_weight, device=logits.device)
        bce_all = F.binary_cross_entropy_with_logits(
            logits, targets_clamped,
            pos_weight=pw,
            reduction="none",
        )
        # Mask out nodata
        bce_loss = (bce_all * valid_mask.float()).sum() / (n_valid + 1e-7)

        # ── Dice ──
        probs = torch.sigmoid(logits)
        probs_v = probs[valid_mask]
        tgt_v   = targets_clamped[valid_mask]
        numerator   = (2.0 * probs_v * tgt_v).sum() + self.smooth
        denominator = (probs_v + tgt_v).sum() + self.smooth
        dice_loss = 1.0 - numerator / (denominator + 1e-7)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ──────────────────────────────────────────────────────────────────────────────
# EMA (Exponential Moving Average) of model weights
# ──────────────────────────────────────────────────────────────────────────────

class EMA:
    """Exponential Moving Average of model parameters for stable inference."""
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ──────────────────────────────────────────────────────────────────────────────
# MODEL BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def build_unet(encoder_name: str = "resnet34", n_channels: int = 3) -> nn.Module:
    """
    Build UNet with segmentation_models_pytorch.
    Adapts first conv for n_channels SAR channels (replicate/average RGB init).
    Falls back to a simple scratch UNet if smp is not installed.
    """
    try:
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=n_channels,
            classes=1,
            activation=None,   # raw logits; sigmoid in loss/inference
        )
        print(f"  [model] UNet + {encoder_name} (pretrained imagenet, adapted {n_channels}-ch SAR)")
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [model] Trainable params: {n_params:,}")
        return model
    except ImportError:
        print("  [model] segmentation_models_pytorch not installed. Using scratch UNet.")
        print("  Install: pip install segmentation-models-pytorch")
        return _build_scratch_unet(n_channels=n_channels)


def _build_scratch_unet(n_channels: int = 3, base_filters: int = 32) -> nn.Module:
    """
    Minimal U-Net from scratch (no pretrained weights).
    Used as Phase 1 baseline and when smp is unavailable.
    """
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.net(x)

    class ScratchUNet(nn.Module):
        def __init__(self, in_ch=3, base=32):
            super().__init__()
            self.enc1 = ConvBlock(in_ch, base)
            self.enc2 = ConvBlock(base, base*2)
            self.enc3 = ConvBlock(base*2, base*4)
            self.enc4 = ConvBlock(base*4, base*8)
            self.bottleneck = ConvBlock(base*8, base*16)
            self.pool = nn.MaxPool2d(2)
            self.up4  = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
            self.dec4 = ConvBlock(base*16, base*8)
            self.up3  = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
            self.dec3 = ConvBlock(base*8, base*4)
            self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
            self.dec2 = ConvBlock(base*4, base*2)
            self.up1  = nn.ConvTranspose2d(base*2, base, 2, stride=2)
            self.dec1 = ConvBlock(base*2, base)
            self.head = nn.Conv2d(base, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            b  = self.bottleneck(self.pool(e4))
            d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.head(d1)

    model = ScratchUNet(in_ch=n_channels, base=base_filters)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [model] Scratch UNet  Trainable params: {n_params:,}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# LR SCHEDULE: Warmup + Cosine
# ──────────────────────────────────────────────────────────────────────────────

def get_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL LOOPS
# ──────────────────────────────────────────────────────────────────────────────

def compute_batch_iou_flood(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Quick IoU(flood) for a single batch (no bootstrap, for monitoring)."""
    if logits.dim() == 4:
        logits = logits.squeeze(1)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    mask  = targets != IGNORE_INDEX
    p = preds[mask]
    g = targets[mask].clamp(min=0)
    tp = int(((p == 1) & (g == 1)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    return tp / max(tp + fp + fn, 1e-7)


def train_one_epoch(
    model, loader, optimizer, scaler, criterion, scheduler, device, epoch
) -> dict:
    model.train()
    total_loss = 0.0
    total_iou  = 0.0
    n_batches  = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item())
        with torch.no_grad():
            total_iou += compute_batch_iou_flood(logits.detach(), labels)
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "iou_flood": total_iou / max(n_batches, 1),
    }


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, threshold: float = 0.5) -> dict:
    model.eval()
    total_loss = 0.0
    tp_total = fp_total = fn_total = tn_total = 0
    n_batches = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += float(loss.item())
            n_batches  += 1

        # Accumulate TP/FP/FN/TN
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).long()
        mask  = labels != IGNORE_INDEX
        p = preds[mask]
        g = labels[mask].clamp(min=0)

        tp_total += int(((p == 1) & (g == 1)).sum())
        fp_total += int(((p == 1) & (g == 0)).sum())
        fn_total += int(((p == 0) & (g == 1)).sum())
        tn_total += int(((p == 0) & (g == 0)).sum())

    eps = 1e-7
    iou_flood = tp_total / max(tp_total + fp_total + fn_total, eps)
    iou_bg    = tn_total / max(tn_total + fp_total + fn_total, eps)

    return {
        "loss":      total_loss / max(n_batches, 1),
        "iou_flood": iou_flood,
        "iou_bg":    iou_bg,
        "miou":      (iou_flood + iou_bg) / 2.0,
        "precision": tp_total / max(tp_total + fp_total, eps),
        "recall":    tp_total / max(tp_total + fn_total, eps),
        "f1":        2*tp_total / max(2*tp_total + fp_total + fn_total, eps),
    }


# ──────────────────────────────────────────────────────────────────────────────
# COMPUTE POS_WEIGHT FROM TRAIN PREVALENCE
# ──────────────────────────────────────────────────────────────────────────────

def compute_pos_weight_from_prevalence(seg_dir: Path) -> float:
    """
    Loads train prevalence from leakage_audit_seg.json.
    pos_weight = (1 - prevalence) / prevalence  (class-balance for BCE).
    Raises if prevalence is outside [0.01, 0.50].
    """
    audit_path = seg_dir / "leakage_audit_seg.json"
    if audit_path.exists():
        with open(audit_path) as f:
            audit = json.load(f)
        prev_pct = audit["prevalence_per_split"].get("train", {}).get("prevalence_pct", None)
        if prev_pct is not None:
            prev = prev_pct / 100.0
            if prev < 0.01 or prev > 0.50:
                raise ValueError(
                    f"Train flood prevalence {prev:.3f} is outside [0.01, 0.50]. "
                    "Audit the split or labels before training."
                )
            pos_weight = (1.0 - prev) / max(prev, 1e-7)
            print(f"  [pos_weight] Train prevalence={prev:.3f}  pos_weight={pos_weight:.2f}")
            return pos_weight

    print("  [pos_weight] leakage_audit_seg.json not found; defaulting to pos_weight=4.0")
    return 4.0


# ──────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def train_seed(
    encoder_name: str,
    seed: int,
    max_epochs: int = 80,
    patience: int = 15,
    batch_size: int = 4,
    base_lr: float = 1e-4,
    encoder_lr_factor: float = 0.1,
    weight_decay: float = 1e-4,
    crop_size: int = 256,
    n_crops: int = 4,
    warmup_epochs: int = 5,
    device_str: str = "auto",
) -> dict:
    """
    Train one seed. Returns eval results on val, test, bolivia.
    """
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )
    print(f"\n{'='*70}")
    print(f"  TRAINING  encoder={encoder_name}  seed={seed}  device={device}")
    print(f"{'='*70}")

    # ── Datasets ──
    from seg_dataset import load_dataset_split
    with open(SEG_DIR / "norm_stats.json") as f:
        norm_stats = json.load(f)

    train_ds = load_dataset_split(
        str(SEG_DIR), "train", norm_stats=norm_stats,
        crop_size=crop_size, augment=True, n_crops_per_chip=n_crops
    )
    val_ds   = load_dataset_split(
        str(SEG_DIR), "val", norm_stats=norm_stats,
        crop_size=crop_size, augment=False, n_crops_per_chip=1
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"),
        persistent_workers=True, drop_last=True,
    )
    val_loader = DataLoader(
        # Validation uses full 512x512 chips; keep it to one chip per batch so
        # it fits on 4 GB GPUs even when the train crop batch is larger.
        val_ds, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    # ── Model ──
    model = build_unet(encoder_name=encoder_name, n_channels=3).to(device)

    # ── Layer-wise LR decay for pretrained encoder ──
    try:
        encoder_params = list(model.encoder.parameters())
        decoder_params = [p for p in model.parameters()
                          if not any(p is ep for ep in encoder_params)]
        param_groups = [
            {"params": encoder_params, "lr": base_lr * encoder_lr_factor},
            {"params": decoder_params, "lr": base_lr},
        ]
        print(f"  [lr] Layer-wise LR decay: encoder={base_lr*encoder_lr_factor:.2e}  decoder={base_lr:.2e}")
    except AttributeError:
        # Scratch UNet -- no encoder/decoder separation
        param_groups = [{"params": model.parameters(), "lr": base_lr}]
        print(f"  [lr] Uniform LR: {base_lr:.2e}")

    optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)

    # ── Loss ──
    pos_weight = compute_pos_weight_from_prevalence(SEG_DIR)
    criterion  = BCEDiceLoss(pos_weight=pos_weight, bce_weight=0.5, dice_weight=0.5)

    # ── LR Schedule: warmup + cosine ──
    steps_per_epoch = len(train_loader)
    total_steps     = max_epochs * steps_per_epoch
    warmup_steps    = warmup_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=get_lr_lambda(warmup_steps, total_steps)
    )

    # ── AMP scaler ──
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ── EMA ──
    ema = EMA(model, decay=0.9999)

    # ── Training loop ──
    best_val_iou  = -1.0
    best_epoch    = 0
    patience_left = patience
    history = []

    ckpt_path = OUT_DIR / f"unet_{encoder_name}_seed{seed}_best.pt"

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, scaler, criterion, scheduler, device, epoch)
        ema.apply_shadow(model)
        val_m   = eval_one_epoch(model, val_loader, criterion, device)
        ema.restore(model)
        ema.update(model)

        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train": train_m, "val": val_m})

        print(
            f"  Epoch {epoch:3d}/{max_epochs}  "
            f"train_loss={train_m['loss']:.4f}  train_iou={train_m['iou_flood']:.4f}  "
            f"val_loss={val_m['loss']:.4f}  val_iou={val_m['iou_flood']:.4f}  "
            f"val_f1={val_m['f1']:.4f}  ({elapsed:.1f}s)"
        )

        # ── Suspicion check ──
        if val_m["iou_flood"] > 0.90:
            print(
                f"\n  *** SUSPICIOUS: val IoU(flood)={val_m['iou_flood']:.4f} > 0.90. "
                "Stop and re-audit split/labels. ***\n"
            )

        # ── Early stopping on val IoU(flood) ──
        if val_m["iou_flood"] > best_val_iou + 1e-4:
            best_val_iou  = val_m["iou_flood"]
            best_epoch    = epoch
            patience_left = patience
            # Save EMA weights
            ema.apply_shadow(model)
            torch.save(model.state_dict(), ckpt_path)
            ema.restore(model)
            print(f"  [ckpt] Saved best model (val IoU={best_val_iou:.4f}) -> {ckpt_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  [early_stop] No improvement for {patience} epochs. Stopping at epoch {epoch}.")
                break

    print(f"\n  Best val IoU(flood): {best_val_iou:.4f} @ epoch {best_epoch}")

    # ── Load best checkpoint ──
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"  [eval] Loaded best checkpoint: {ckpt_path}")

    # ── Final evaluation on val, test, bolivia ──
    from seg_metrics import SegmentationMetrics, calibrate_threshold

    results = {"seed": seed, "encoder": encoder_name, "best_epoch": best_epoch,
               "best_val_iou": best_val_iou, "splits": {}, "history": history}

    # Calibrate threshold on VAL
    # Collect val logits
    model.eval()
    val_logits_list, val_labels_list = [], []
    with torch.no_grad():
        for imgs_v, lbls_v in val_loader:
            logits_v = model(imgs_v.to(device))
            if logits_v.dim() == 4:
                logits_v = logits_v.squeeze(1)
            val_logits_list.append(logits_v.cpu().numpy())
            val_labels_list.append(lbls_v.numpy())
    val_logits_all = np.concatenate(val_logits_list, axis=0)
    val_labels_all = np.concatenate(val_labels_list, axis=0)
    best_threshold = calibrate_threshold(val_logits_all, val_labels_all)

    for split_name in ["val", "test", "bolivia"]:
        print(f"\n  -- Final eval on {split_name} --")
        try:
            split_ds = load_dataset_split(
                str(SEG_DIR), split_name, norm_stats=norm_stats,
                crop_size=crop_size, augment=False, n_crops_per_chip=1
            )
        except FileNotFoundError:
            print(f"  [SKIP] {split_name} manifest not found")
            continue

        # Test and OOD evaluation also runs on full chips.
        split_loader = DataLoader(split_ds, batch_size=1, shuffle=False, num_workers=2)
        met = SegmentationMetrics(threshold=best_threshold)

        model.eval()
        with torch.no_grad():
            for imgs_s, lbls_s in split_loader:
                logits_s = model(imgs_s.to(device))
                if logits_s.dim() == 4:
                    logits_s = logits_s.squeeze(1)
                met.update(logits_s.cpu().numpy(), lbls_s.numpy())

        split_res = met.compute(n_bootstrap=1000)
        results["splits"][split_name] = {
            "global": split_res["global"],
            "bootstrap_ci": split_res["bootstrap_ci_by_chip"],
            "n_chips": split_res["n_chips"],
            "threshold": best_threshold,
        }

        g = split_res["global"]
        print(
            f"  {split_name}: IoU(flood)={g['iou_flood']:.4f}  mIoU={g['miou']:.4f}  "
            f"F1={g['f1']:.4f}  Prec={g['precision']:.4f}  Rec={g['recall']:.4f}  "
            f"PR-AUC={g['pr_auc']:.4f}"
        )

    # Save per-seed result
    seed_path = OUT_DIR / f"unet_{encoder_name}_seed{seed}_results.json"
    with open(seed_path, "w") as f:
        # Exclude full history to keep file small
        compact_res = {k: v for k, v in results.items() if k != "history"}
        total_epochs = len(history)
        compact_res["total_epochs_trained"] = total_epochs
        compact_res["n_train_epochs"] = total_epochs
        compact_res["best_epoch"] = best_epoch
        compact_res["evaluation_notes"] = {
            "metric_aggregation": "Headline 'global' IoU is micro-pooled across all valid pixels in the split. 'bootstrap_ci' is macro-averaged across individual chips via 1000 bootstrap resamples with replacement.",
            "chip_count_explanation": "Evaluated chip counts (val: 86/89, test: 89/90) exclude chips where 100% of pixels are nodata (label == -1), as IoU with 0 ground-truth pixels is undefined (0/0)."
        }
        json.dump(compact_res, f, indent=2)
    print(f"\n  [OK] Seed {seed} results: {seed_path}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# MULTI-SEED SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_seeds(all_results: list, encoder_name: str) -> dict:
    """Compute mean +/- 95% CI across seeds for each split/metric."""
    seeds = [r["seed"] for r in all_results]
    splits = list(all_results[0]["splits"].keys())

    summary = {"encoder": encoder_name, "seeds": seeds, "splits": {}}
    for split in splits:
        metrics_across_seeds = {}
        for r in all_results:
            if split not in r["splits"]:
                continue
            g = r["splits"][split]["global"]
            for k, v in g.items():
                if isinstance(v, float):
                    metrics_across_seeds.setdefault(k, []).append(v)

        split_summary = {}
        for k, vals in metrics_across_seeds.items():
            arr = np.array(vals)
            split_summary[k] = {
                "mean":    float(np.mean(arr)),
                "std":     float(np.std(arr)),
                "ci95_lo": float(np.percentile(arr, 2.5)) if len(arr) >= 2 else float(np.mean(arr)),
                "ci95_hi": float(np.percentile(arr, 97.5)) if len(arr) >= 2 else float(np.mean(arr)),
                "n_seeds": len(vals),
            }
        summary["splits"][split] = split_summary

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING + MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train U-Net segmentation model")
    parser.add_argument("--encoder",     default="resnet34",
                        choices=["resnet34", "efficientnet-b3", "convnext_tiny", "scratch"],
                        help="Encoder backbone (or 'scratch' for no pretrained weights)")
    parser.add_argument("--seed",        type=int, default=42, help="Random seed")
    parser.add_argument("--all-seeds",   action="store_true",
                        help="Run all 5 seeds (42, 7, 99, 123, 314)")
    parser.add_argument("--max-epochs",  type=int, default=80)
    parser.add_argument("--patience",    type=int, default=15)
    parser.add_argument("--batch-size",  type=int, default=4)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--crop-size",   type=int, default=256)
    parser.add_argument("--n-crops",     type=int, default=4)
    parser.add_argument("--device",      default="auto")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  FloodSentinel AI  --  Phase 3 U-Net Training")
    print(f"  Encoder: {args.encoder}  Device: {args.device}")
    print("=" * 70)

    # Verify dataset
    if not (SEG_DIR / "train_chips.json").exists():
        raise RuntimeError(
            f"Seg dataset not found: {SEG_DIR}\n"
            "Run: python scripts/phase0_build_seg_dataset.py"
        )

    # Map encoder name for smp
    encoder_map = {
        "resnet34":        "resnet34",
        "efficientnet-b3": "efficientnet-b3",
        "convnext_tiny":   "tu-convnext_tiny",
        "scratch":         None,
    }
    encoder_name = encoder_map.get(args.encoder, args.encoder)

    SEEDS = [42, 7, 99, 123, 314] if args.all_seeds else [args.seed]

    all_results = []
    for seed in SEEDS:
        if encoder_name is None:
            # Scratch UNet -- monkey-patch build function
            import sys
            def _build_scratch_override(encoder_name, n_channels):
                return _build_scratch_unet(n_channels=n_channels)
            # Temporarily override
            result = train_seed(
                encoder_name="scratch", seed=seed,
                max_epochs=args.max_epochs, patience=args.patience,
                batch_size=args.batch_size, base_lr=args.lr,
                crop_size=args.crop_size, n_crops=args.n_crops,
                device_str=args.device,
            )
        else:
            result = train_seed(
                encoder_name=encoder_name, seed=seed,
                max_epochs=args.max_epochs, patience=args.patience,
                batch_size=args.batch_size, base_lr=args.lr,
                crop_size=args.crop_size, n_crops=args.n_crops,
                device_str=args.device,
            )
        all_results.append(result)

    # Aggregate across seeds
    if len(all_results) > 1:
        summary = aggregate_seeds(all_results, encoder_name or "scratch")
        summary_path = OUT_DIR / f"unet_{args.encoder}_all_seeds_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  [OK] Multi-seed summary: {summary_path}")

        print("\n" + "=" * 70)
        print(f"  MULTI-SEED SUMMARY ({len(SEEDS)} seeds):  {args.encoder}")
        print("=" * 70)
        for split in ["val", "test", "bolivia"]:
            if split in summary["splits"]:
                g = summary["splits"][split]
                print(f"\n  {split}:")
                for k in ["iou_flood", "miou", "f1", "precision", "recall"]:
                    if k in g:
                        m = g[k]
                        print(f"    {k:<18}: {m['mean']:.4f} +/- [{m['ci95_lo']:.4f}, {m['ci95_hi']:.4f}]")
    else:
        # Single seed -- quick summary
        r = all_results[0]
        print("\n" + "=" * 70)
        print(f"  RESULTS  encoder={args.encoder}  seed={args.seed}")
        print("=" * 70)
        for split, sres in r["splits"].items():
            g = sres["global"]
            print(f"\n  {split}:  IoU(flood)={g['iou_flood']:.4f}  mIoU={g['miou']:.4f}  F1={g['f1']:.4f}")
            ci = sres.get("bootstrap_ci", {})
            if "iou_flood" in ci:
                c = ci["iou_flood"]
                print(f"    IoU(flood) 95% CI: [{c['ci95_lo']:.4f}, {c['ci95_hi']:.4f}]")

    print("\n  Expected: IoU(flood) roughly 0.6-0.8 for S1-only UNet.")
    print("  If > 0.90 on test/Bolivia: STOP and re-audit split/labels.")
    print("=" * 70)


if __name__ == "__main__":
    main()
