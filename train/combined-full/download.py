#!/usr/bin/env python3
"""Download any images referenced by train/combined-full/select.py's manifests
that aren't already on disk, into each SOURCE dataset's own images/ dir (not
combined-animal-full/ -- that only ever holds symlinks, built by
prepare_data.py). Reuses whatever's already been downloaded by each
dataset's own pipeline; only fetches what select.py's larger full-pool
selection actually added.

Same public/unsigned S3 bucket both datasets already use (see
~/data/bb9k/readme.txt): us-west-2.opendata.source.coop, under
agentmorris/lila-wildlife/{caltech-unzipped/cct_images,oregon-critters}/.
Caltech's S3 key is the bare file_name; Oregon's is prefix + source_path
(the original nested path, flattened to a basename for local storage by
select.py -- see train/oregon-critters/select_and_download.py for why).
"""
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

DATA_ROOT = Path.home() / "data/bb9k"
COMBINED_DIR = DATA_ROOT / "combined-animal-full"

BUCKET = "us-west-2.opendata.source.coop"

SOURCES = [
    {
        "name": "caltech",
        "manifest": COMBINED_DIR / "caltech_selected.json",
        "images_dir": DATA_ROOT / "caltech-camera-traps" / "images",
        "prefix": "agentmorris/lila-wildlife/caltech-unzipped/cct_images/",
        "key_field": "file_name",
    },
    {
        "name": "oregon",
        "manifest": COMBINED_DIR / "oregon_selected.json",
        "images_dir": DATA_ROOT / "oregon-critters" / "images",
        "prefix": "agentmorris/lila-wildlife/oregon-critters/",
        "key_field": "source_path",
    },
]


def main():
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    lock = threading.Lock()

    for source in SOURCES:
        with open(source["manifest"]) as f:
            images = json.load(f)["images"]
        source["images_dir"].mkdir(parents=True, exist_ok=True)

        todo = [im for im in images if not (source["images_dir"] / im["file_name"]).exists()]
        print(f"{source['name']}: {len(images)} selected, {len(images) - len(todo)} already on disk, {len(todo)} to download")

        counters = {"done": 0, "failed": 0}
        failed_list = []

        def download_one(im, source=source):
            key = source["prefix"] + im[source["key_field"]]
            dest = source["images_dir"] / im["file_name"]
            try:
                s3.download_file(BUCKET, key, str(dest))
                with lock:
                    counters["done"] += 1
            except Exception as e:
                with lock:
                    counters["failed"] += 1
                    failed_list.append((im["file_name"], str(e)))

        total = len(todo)
        with ThreadPoolExecutor(max_workers=32) as ex:
            futures = [ex.submit(download_one, im) for im in todo]
            processed = 0
            for fut in as_completed(futures):
                fut.result()
                processed += 1
                if processed % 1000 == 0 or processed == total:
                    print(
                        f"  {source['name']}: {processed}/{total} "
                        f"(downloaded={counters['done']}, failed={counters['failed']})",
                        flush=True,
                    )

        print(f"{source['name']}: done. {counters}")
        if failed_list:
            fail_path = COMBINED_DIR / f"{source['name']}_download_failures.json"
            with open(fail_path, "w") as f:
                json.dump(failed_list, f, indent=2)
            print(f"  wrote {len(failed_list)} failures to {fail_path}")


if __name__ == "__main__":
    main()
