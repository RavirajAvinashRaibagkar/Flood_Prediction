from google.cloud import storage
from collections import Counter

client = storage.Client.create_anonymous_client()
bucket = client.bucket("sen1floods11")
blobs = list(bucket.list_blobs(prefix="v1.1/data/flood_events/HandLabeled/"))

print(f"Total HandLabeled blobs: {len(blobs)}")
types = Counter()
total_bytes = 0
for b in blobs:
    name = b.name.split("/")[-1]
    total_bytes += b.size
    for suffix in ["_S1Hand.tif", "_LabelHand.tif", "_S2Hand.tif", "_JRCWaterHand.tif", "_QC.tif"]:
        if name.endswith(suffix):
            types[suffix] += 1
            break
print("Breakdown by type:")
for t, count in types.items():
    print(f"  {t}: {count}")
print(f"Total size: {total_bytes / (1024*1024):.1f} MB")
