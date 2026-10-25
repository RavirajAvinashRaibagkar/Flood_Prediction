from google.cloud import storage
from pathlib import Path
from PIL import Image
import numpy as np

outdir = Path("flood_project/ml_training/data/raw/sen1floods11/HandLabeled")
(outdir / "S1Hand").mkdir(parents=True, exist_ok=True)
(outdir / "LabelHand").mkdir(parents=True, exist_ok=True)

client = storage.Client.create_anonymous_client()
bucket = client.bucket("sen1floods11")

s1_blob = bucket.blob("v1.1/data/flood_events/HandLabeled/S1Hand/Bolivia_103757_S1Hand.tif")
label_blob = bucket.blob("v1.1/data/flood_events/HandLabeled/LabelHand/Bolivia_103757_LabelHand.tif")

s1_path = outdir / "S1Hand" / "Bolivia_103757_S1Hand.tif"
label_path = outdir / "LabelHand" / "Bolivia_103757_LabelHand.tif"

print("Downloading Bolivia_103757_S1Hand.tif (size:", s1_blob.size, "bytes)...")
s1_blob.download_to_filename(str(s1_path))
print("Downloading Bolivia_103757_LabelHand.tif (size:", label_blob.size, "bytes)...")
label_blob.download_to_filename(str(label_path))

im_s1 = Image.open(s1_path)
print("S1 format:", im_s1.format, "size:", im_s1.size, "mode:", im_s1.mode)
# read bands
bands = []
try:
    for i in range(getattr(im_s1, 'n_frames', 1)):
        im_s1.seek(i)
        bands.append(np.array(im_s1))
    arr_s1 = np.stack(bands, axis=-1) if len(bands) > 1 else np.array(im_s1)
except Exception as e:
    arr_s1 = np.array(im_s1)
print("arr_s1 shape:", arr_s1.shape, "dtype:", arr_s1.dtype, "min/max:", np.nanmin(arr_s1), np.nanmax(arr_s1))

im_lbl = Image.open(label_path)
arr_lbl = np.array(im_lbl)
print("arr_lbl shape:", arr_lbl.shape, "unique values:", np.unique(arr_lbl))
flood_pixels = np.sum(arr_lbl == 1)
total_valid = np.sum(arr_lbl >= 0)
print(f"Flood fraction: {flood_pixels}/{total_valid} = {flood_pixels/max(1, total_valid):.4f}")
