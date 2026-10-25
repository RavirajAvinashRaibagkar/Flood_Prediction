# Train flood segmentation on the real Sen1Floods11 data

## What is already here

The project already has the real Sen1Floods11 hand-labeled Sentinel-1 SAR data: 446 paired chips and masks, plus official split files. `flood_project/ml_training/data/seg_dataset/leakage_audit_seg.json` records 252 train, 89 validation, 90 test, and 15 Bolivia holdout chips; it reports no cross-split duplicates and train-only normalization statistics. No dataset download or split rebuild is needed.

Train the U-Net segmentation model in `flood_project/ml_training/train_unet_seg.py`. Do not use `train_cnn_classifier.py` for this run: its current data generator trains on synthetic images. The U-Net consumes the real 2-band Sentinel-1 VV/VH data (with a derived third channel) and hand-labeled masks.

## Windows setup

Use Python 3.10–3.14. From PowerShell in the repository root:

```powershell
py -3.12 -m venv .venv-seg
.\.venv-seg\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the CUDA build of PyTorch and torchvision using the command selected for Windows + Pip + CUDA on the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). This computer has an NVIDIA RTX 2050 with 4 GB VRAM and a CUDA 13.1-capable driver; choose a supported CUDA wheel in the selector. The PyTorch wheel supplies its CUDA runtime, so a separate CUDA toolkit install is normally unnecessary.

Then install the project’s geospatial/model packages and check GPU availability:

```powershell
python -m pip install -r flood_project\ml_training\requirements-segmentation.txt
python -c "import torch, rasterio, segmentation_models_pytorch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), 'GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If that reports `CUDA False`, use the official selector to reinstall the CUDA-enabled PyTorch wheel before training.

## Train one reproducible checkpoint

The first run downloads the ImageNet ResNet-34 encoder weights. Run from the repository root:

```powershell
python flood_project\ml_training\train_unet_seg.py --encoder resnet34 --seed 42 --max-epochs 25 --patience 5 --batch-size 1 --crop-size 256 --n-crops 1 --device cuda
```

The batch size and one training crop per chip are conservative for this GPU’s 4 GB VRAM. The trainer uses random training crops and augmentation, early stopping on validation IoU, and evaluates the held-out validation, test, and Bolivia splits. Do not tune on test or Bolivia scores. If the CUDA process runs out of memory, retry with `--crop-size 128`; if CUDA is unavailable, use `--device cpu` (expect it to be much slower).

Expected outputs are under `flood_project/ml_training/saved_models/`:

- `unet_resnet34_seed42_best.pt` — best validation-IoU weights
- `unet_resnet34_seed42_results.json` — split metrics and selected threshold

Keep the split audit with the result, and report test/Bolivia IoU, F1, precision, recall, and the validation-selected threshold. A strong training score by itself is not evidence of generalization.

## Before using it in the app

This run trains a PyTorch U-Net for pixel masks. The current backend image endpoint serves a separate RGB MobileNet classifier; it does not load `.pt` U-Net checkpoints or accept the required Sentinel-1 VV/VH GeoTIFF input yet. The trained U-Net needs a matching GeoTIFF preprocessing/inference adapter and a segmentation API endpoint before the dashboard can use its masks. Do not replace the current classifier checkpoint with this `.pt` file.

## Google Colab option

Colab avoids installing the CUDA stack on Windows. GPU availability and runtime duration vary, so save the output checkpoint to Drive when training finishes. [Colab resource availability and runtime limits](https://research.google.com/colaboratory/faq.html)

1. In Google Drive, create `FloodSentinelTraining/ml_training/` and upload these folders/files while preserving their structure:
   - `data/raw/sen1floods11/HandLabeled/` (the `S1Hand`, `LabelHand`, and `JRCWaterHand` subfolders)
   - `data/seg_dataset/`
   - `train_unet_seg.py`, `seg_dataset.py`, and `seg_metrics.py`
2. Open a Colab notebook, choose **Runtime → Change runtime type → GPU**, and run:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Copy the bundle to the Colab VM, install the two project packages, and confirm CUDA is active:

```bash
!cp -r /content/drive/MyDrive/FloodSentinelTraining /content/FloodSentinelTraining
%cd /content/FloodSentinelTraining/ml_training
!pip install rasterio segmentation-models-pytorch timm
!python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

4. Train one seed using the same command as the Windows recipe, replacing its relative path with the current folder:

```bash
!python train_unet_seg.py --encoder resnet34 --seed 42 --max-epochs 25 --patience 5 --batch-size 1 --crop-size 256 --n-crops 1 --device cuda
```

5. Copy the resulting model and metrics back to Drive:

```bash
!mkdir -p /content/drive/MyDrive/FloodSentinelTraining/saved_models
!cp saved_models/unet_resnet34_seed42_best.pt saved_models/unet_resnet34_seed42_results.json /content/drive/MyDrive/FloodSentinelTraining/saved_models/
```

The dataset loader now rebases the existing Windows manifest paths onto the copied Colab project, so you don’t need to regenerate the splits. Keep the notebook connected until the final copy-back finishes.
