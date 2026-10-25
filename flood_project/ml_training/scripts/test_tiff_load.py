from google.cloud import storage
from pathlib import Path
from PIL import Image
import numpy as np

outdir = Path("flood_project/ml_training/data/raw/sen1floods11/HandLabeled")
outdir.mkdir(parents=True, exist_ok=True)

client = storage.Client.create_anonymous_client()
bucket = client.bucket("sen1floods11")

# download one S1 and one Label
s1_blob = bucket.blob("v1.1/data/flood_events/HandLabeled/Bolivia_103757_S1Hand.tif")
label_blob = bucket.blob("v1.1/data/flood_events/HandLabeled/Bolivia_103757_LabelHand.tif")

s1_path = outdir / "Bolivia_103757_S1Hand.tif"
label_path = outdir / "Bolivia_103757_LabelHand.tif"

print("Downloading Bolivia_103757_S1Hand.tif...")
s1_blob.download_to_filename(str(s1_path))
print("Downloading Bolivia_103757_LabelHand.tif...")
label_blob.download_to_filename(str(label_path))

# Inspect with PIL
im_s1 = Image.open(s1_path)
print("S1 mode:", im_s1.mode, "size:", im_s1.size, "n_frames:", getattr(im_s1, 'n_frames', 1))
arr_s1 = np.array(im_s1)
print("S1 array shape:", arr_s1.shape, "dtype:", arr_s1.dtype, "min/max:", arr_s1.min(), arr_s1.max())

im_lbl = Image.open(label_path)
print("Label mode:", im_lbl.mode, "size:", im_lbl.size)
arr_lbl = np.array(im_lbl)
print("Label shape:", arr_lbl.shape, "unique:", np.unique(arr_lbl))
