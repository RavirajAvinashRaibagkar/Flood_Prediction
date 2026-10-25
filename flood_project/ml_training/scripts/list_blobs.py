from google.cloud import storage

client = storage.Client.create_anonymous_client()
bucket = client.bucket("sen1floods11")
blobs = list(bucket.list_blobs(prefix="v1.1/data/flood_events/HandLabeled/", max_results=20))

for b in blobs:
    print(b.name, b.size)
