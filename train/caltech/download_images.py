import json
import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

BASE = "/home/ANT.AMAZON.COM/jmigdal/data/bb9k/caltech-camera-traps"
JSON_PATH = os.path.join(BASE, "caltech_bboxes_20200316_filtered_20k.json")
IMAGES_DIR = os.path.join(BASE, "images")

BUCKET = "us-west-2.opendata.source.coop"
PREFIX = "agentmorris/lila-wildlife/caltech-unzipped/cct_images/"

os.makedirs(IMAGES_DIR, exist_ok=True)

with open(JSON_PATH) as f:
    data = json.load(f)

images = data["images"]
print(f"Images to download: {len(images)}")

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

lock = threading.Lock()
counters = {"done": 0, "skipped": 0, "failed": 0}
failed_list = []

def download_one(im):
    file_name = im["file_name"]
    key = PREFIX + file_name
    dest = os.path.join(IMAGES_DIR, file_name)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        with lock:
            counters["skipped"] += 1
        return
    try:
        s3.download_file(BUCKET, key, dest)
        with lock:
            counters["done"] += 1
    except Exception as e:
        with lock:
            counters["failed"] += 1
            failed_list.append((file_name, str(e)))

total = len(images)
with ThreadPoolExecutor(max_workers=32) as ex:
    futures = [ex.submit(download_one, im) for im in images]
    processed = 0
    for fut in as_completed(futures):
        fut.result()
        processed += 1
        if processed % 500 == 0 or processed == total:
            print(f"Progress: {processed}/{total} "
                  f"(downloaded={counters['done']}, skipped={counters['skipped']}, failed={counters['failed']})",
                  flush=True)

print("=== Download complete ===")
print(counters)
if failed_list:
    with open(os.path.join(BASE, "download_failures.json"), "w") as f:
        json.dump(failed_list, f, indent=2)
    print(f"Wrote {len(failed_list)} failures to download_failures.json")
