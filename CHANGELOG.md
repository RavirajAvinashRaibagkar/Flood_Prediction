# CHANGELOG.md — FloodSentinel AI
=====================================
All notable changes in chronological order.
Format: `[PHASE] YYYY-MM-DD: Description (one-line rationale)`

---

## Phase 0 — Fix the Task and the Split (2026-10-20)

### [PHASE 0] 2026-10-20: Discovered and downloaded official Sen1Floods11 GCS split CSVs
- **Script**: `scripts/phase0_list_gcs_splits.py`
- **Evidence**: Found 4 official split files at `gs://sen1floods11/v1.1/splits/flood_handlabeled/`:
  - `flood_train_data.csv` (train split, chip-level)
  - `flood_valid_data.csv` (val split)
  - `flood_test_data.csv` (test split)
  - `flood_bolivia_data.csv` (Bolivia holdout — never for tuning)
- **Rationale**: Must use official splits, never invent splits (per Phase 0 rule).

### [PHASE 0] 2026-10-20: Switched from classification to pixel-wise segmentation
- **Files changed**: Entire training pipeline redesigned
- **Old approach**: 128px patch → binary label (flood/no-flood for patch) — memorization of coarse labels
- **New approach**: Full 512×512 chip → pixel-wise binary mask, ignore_index=-1 for nodata
- **Scripts**: `seg_dataset.py`, `seg_metrics.py`, `train_unet_seg.py`
- **Rationale**: Segmentation is the scientifically correct task; patch classification loses spatial information and conflates boundary pixels.

### [PHASE 0] 2026-10-20: Built segmentation dataset with zero chip-ID overlap proof
- **Script**: `scripts/phase0_build_seg_dataset.py`
- **Assertion**: ZERO chip-ID overlap across train/val/test/bolivia (raises RuntimeError if violated)
- **Outputs**: `data/seg_dataset/{train,val,test,bolivia}_chips.json`, `leakage_audit_seg.json`
- **Rationale**: Chip-ID overlap = data leakage; must be proven and logged.

### [PHASE 0] 2026-10-20: Train-only normalization stats fitted correctly
- **Script**: `scripts/phase0_build_seg_dataset.py`, step 3
- **Channels**: VV dB, VH dB, VV-VH dB; clipped to [-35, 5] dB
- **Fit on**: TRAIN split ONLY; val/test/bolivia use these same stats
- **Saved**: `data/seg_dataset/norm_stats.json`
- **Rationale**: Fitting scaler on full dataset leaks val/test statistics into training.

### [PHASE 0] 2026-10-20: Flood-pixel prevalence computed per split (not 45%)
- **Output**: `data/seg_dataset/leakage_audit_seg.json → prevalence_per_split`
- **Expected**: ~9-17% flood pixels per split (NOT 45% which was patch-level artifact)
- **Rationale**: The old 45% was patch classification artifact; pixel-level prevalence is physically meaningful.

### [PHASE 0] 2026-10-20: pHash deduplicate check across all splits
- **Script**: `scripts/phase0_build_seg_dataset.py`, step 5
- **Method**: pHash on VV band, Hamming distance threshold = 10
- **Output**: cross-split dupe count in leakage_audit_seg.json
- **Rationale**: Perceptual duplicates across train/test inflate metrics.

### [PHASE 0] 2026-10-20: Loss and metrics ignore label=-1 (nodata)
- **Files**: `seg_metrics.py`, `train_unet_seg.py` (BCEDiceLoss, all metric computations)
- **Implementation**: `valid_mask = labels != -1` everywhere; nodata pixels excluded from BCE, Dice, TP/FP/FN/TN
- **Rationale**: Treating nodata as a class inflates easy-to-predict background metrics.

### [PHASE 0] 2026-10-20: Verified dataset license from local LICENSE.txt
- **Evidence**: `data/raw/sen1floods11/LICENSE.txt` contains "Creative Commons Attribution 4.0 International (CC BY 4.0)"
- **Not assumed**: License was physically verified from downloaded file, not from memory
- **Recorded**: `LICENSES.md` updated with verification note

---

## Phase 1 — Baselines (2026-10-20)

### [PHASE 1] 2026-10-20: Otsu / fixed-threshold baselines on segmentation task
- **Script**: `phase1_baselines.py`
- **Baselines**: majority class, Otsu(VV), Otsu(VH), fixed VV=-16dB, fixed VH=-22dB
- **Evaluated on**: val, test, bolivia SEPARATELY
- **CI method**: Bootstrap 95% CI resampled by CHIP (not pixel)
- **Rationale**: Classical baselines set realistic floor; any learned model must beat them.

---

## Phase 2 — More Real Data (2026-10-20, scripts only)

### [PHASE 2a] 2026-10-20: Download script for Sen1Floods11 S1Weak (~4,385 chips)
- **Script**: `scripts/download_sen1floods11_weak.py`
- **Exclusion**: Automatically excludes val/test/bolivia chip IDs before downloading
- **Use**: Pretraining only, never val/test/threshold calibration
- **Rationale**: Weak labels = approximate; must not contaminate evaluation.

### [PHASE 2b] 2026-10-20: Download script for MMFlood (Zenodo 6534637, 11.3 GB)
- **Script**: `scripts/download_mmflood.py`
- **MD5**: 2817eb94107f1c96ef68d843010bdf97
- **Split**: BY EVENT (95 events → ~76 train, 9 val, 10 test external)
- **Use**: Additional training data + EXTERNAL cross-dataset test
- **Rationale**: Cross-dataset evaluation proves generalization beyond a single dataset.

---

## Phase 3 — Models and Training (2026-10-20, scripts written)

### [PHASE 3] 2026-10-20: U-Net segmentation training with BCE+Dice loss
- **Script**: `train_unet_seg.py`
- **Architecture**: U-Net + pretrained encoder (resnet34 / efficientnet-b3 / convnext_tiny) via smp
- **Loss**: BCE (pos_weight from train prevalence) + Dice; both ignore_index=-1
- **Optimizer**: AdamW, layer-wise LR decay (encoder 10x slower), warmup + cosine
- **AMP**: fp16 gradient scaling enabled for CUDA
- **EMA**: Exponential moving average of weights (decay=0.9999)
- **Threshold**: Calibrated on VAL only, frozen for test/bolivia
- **5 seeds**: [42, 7, 99, 123, 314], report mean +/- 95% CI

### [PHASE 3] 2026-10-20: Segmentation metrics with bootstrap CI by chip
- **Script**: `seg_metrics.py`
- **Metrics**: IoU(flood), IoU(bg), mIoU, F1, precision, recall, PR-AUC
- **CI**: Bootstrap 1000 resamples by chip (not pixel)
- **Suspicion check**: Prints audit warning if IoU(flood) > 0.90

---

## Phase 0 & 1 Execution & Deliverables (2026-10-21)

### [PHASE 0] 2026-10-21: Successfully built segmentation dataset from official splits
- **Script**: `scripts/phase0_build_seg_dataset.py`
- **Output files**: `data/seg_dataset/{train,val,test,bolivia}_chips.json`, `split_manifest.json`, `norm_stats.json`, `leakage_audit_seg.json`
- **Verification**: Zero chip-ID overlap, zero cross-split pHash duplicates (446 chips), pixel prevalence confirmed: Bolivia 15.9%, Train 9.5%, Val 11.0%, Test 12.5%.
- **Norm stats**: VV mean=-10.38 dB, VH mean=-17.18 dB, VV-VH mean=6.47 dB fit strictly on train split.

### [PHASE 1] 2026-10-21: Evaluated classical segmentation baselines on val/test/bolivia
- **Script**: `phase1_baselines.py`
- **Output**: `saved_models/baseline_results.json`
- **Results (IoU flood)**:
  - Majority class: val 0.0000, test 0.0000, bolivia 0.0000
  - Otsu (VV): val 0.1834, test 0.2022, bolivia 0.2016
  - Otsu (VH): val 0.2114, test 0.2233, bolivia 0.2200
  - Fixed VV (-16 dB): val 0.4595, test 0.4544, bolivia 0.5045
  - Fixed VH (-22 dB): val 0.4785, test 0.5208, bolivia 0.6342

### [PHASE 2] 2026-10-21: Acquired official pre-trained paper checkpoints from GCS
- **Script**: `scripts/download_pretrained_checkpoints.py`
- **Target**: `data/raw/sen1floods11/checkpoints/` (4 checkpoints, ~503 MB, verified with MD5).

### [PHASE 6] 2026-10-21: Wired real flood mask injection into route.py
- **Script**: `route.py`
- **Change**: Updated `demo_route` and CLI `--mask` argument to accept real 2D flood masks from segmentation inference.
- **Test**: All 3 routing validation tests passing in `test_routing_validation.py`.

---

## Pending / In Progress

- [x] Phase 0: Run phase0_build_seg_dataset.py to completion (norm stats + prevalence + phash)
- [x] Phase 1: Run phase1_baselines.py on val/test/bolivia
- [x] Phase 2: Download pre-trained paper checkpoints (4 files, ~503 MB)
- [ ] Phase 2a: Download S1Weak (requires GCS access, ~4,385 chips)
- [ ] Phase 2b: Download MMFlood (requires Zenodo download, 11.3 GB)
- [x] Phase 2c: Check WorldFloods license (Apache 2.0 = commercial OK, verified in LICENSES.md)
- [x] Phase 2d: SpaceNet 8 license check (CC BY-SA 4.0 = share-alike, verified in LICENSES.md)
- [ ] Phase 3: Install segmentation_models_pytorch, run train_unet_seg.py
- [ ] Phase 3: DeepLabV3+, SegFormer-B2, Optuna 30 trials
- [ ] Phase 4: Per-event results, failure gallery, Grad-CAM
- [ ] Phase 5: Integration with backend /health endpoint
- [x] Phase 6: Routing validation with SpaceNet 8 + OSMnx & real mask ingestion
