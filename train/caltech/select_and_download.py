import json
import random
import os
import sys
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed

random.seed(42)

BASE = "/home/ANT.AMAZON.COM/jmigdal/data/bb9k/caltech-camera-traps"
JSON_PATH = os.path.join(BASE, "caltech_bboxes_20200316.json")
IMAGES_DIR = os.path.join(BASE, "images")
OUT_JSON = os.path.join(BASE, "caltech_bboxes_20200316_filtered_20k.json")

BUCKET = "us-west-2.opendata.source.coop"
PREFIX = "agentmorris/lila-wildlife/caltech-unzipped/cct_images/"

SMALL_RODENT_CAP = 10000
ADDITIONAL_CAP = 10000

print("Loading json...")
with open(JSON_PATH) as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]
categories = data["categories"]

cat_name_by_id = {c["id"]: c["name"] for c in categories}
cat_id_by_name = {c["name"]: c["id"] for c in categories}

img_by_id = {im["id"]: im for im in images}

# frame_num == 1 images only
frame1_ids = set(im["id"] for im in images if im.get("frame_num") == 1)
print(f"Total images: {len(images)}, frame_num==1 images: {len(frame1_ids)}")

# annotations restricted to frame1 images
anns_by_image = {}
for a in annotations:
    iid = a["image_id"]
    if iid in frame1_ids:
        anns_by_image.setdefault(iid, []).append(a)

def images_with_cat(cat_name):
    cid = cat_id_by_name.get(cat_name)
    if cid is None:
        return set()
    result = set()
    for iid, anns in anns_by_image.items():
        if any(a["category_id"] == cid for a in anns):
            result.add(iid)
    return result

rabbit_ids = images_with_cat("rabbit")
squirrel_ids = images_with_cat("squirrel")
rodent_ids = images_with_cat("rodent")

print(f"rabbit frame1 images: {len(rabbit_ids)}")
print(f"squirrel frame1 images: {len(squirrel_ids)}")
print(f"rodent frame1 images: {len(rodent_ids)}")

# Step 1: rabbit images, up to 10,000
rabbit_list = sorted(rabbit_ids)
if len(rabbit_list) > SMALL_RODENT_CAP:
    random.shuffle(rabbit_list)
    selected_rabbit = set(rabbit_list[:SMALL_RODENT_CAP])
else:
    selected_rabbit = set(rabbit_list)

small_rodent_selected = set(selected_rabbit)
print(f"Selected rabbit images: {len(selected_rabbit)}")

# Step 2: if under cap, fill from other small rodent categories (squirrel, rodent)
remaining_budget = SMALL_RODENT_CAP - len(small_rodent_selected)
other_pool = (squirrel_ids | rodent_ids) - small_rodent_selected
other_pool_list = sorted(other_pool)
random.shuffle(other_pool_list)

if remaining_budget > 0:
    take = other_pool_list[:remaining_budget]
    small_rodent_selected.update(take)
    print(f"Added {len(take)} other small-rodent images (squirrel/rodent). "
          f"Pool available was {len(other_pool_list)}, budget was {remaining_budget}.")
else:
    print("Rabbit alone met/exceeded the 10k small-rodent cap; no other small rodents added.")

print(f"Total small rodent (incl. rabbit) selected: {len(small_rodent_selected)} "
      f"(cap {SMALL_RODENT_CAP})")
assert len(small_rodent_selected) <= SMALL_RODENT_CAP

# Step 3: random sample of an additional 10k images from remaining frame1 pool
remaining_pool = list(frame1_ids - small_rodent_selected)
random.shuffle(remaining_pool)
additional_selected = set(remaining_pool[:ADDITIONAL_CAP])
print(f"Additional random sample selected: {len(additional_selected)} "
      f"(pool available was {len(remaining_pool)}, cap {ADDITIONAL_CAP})")

final_selected = small_rodent_selected | additional_selected
print(f"FINAL total selected images: {len(final_selected)} (max 20000)")
assert len(final_selected) <= 20000

# Build filtered coco json
filtered_images = [img_by_id[iid] for iid in final_selected]
filtered_annotations = [a for a in annotations if a["image_id"] in final_selected]

filtered_data = {
    "info": data.get("info", {}),
    "categories": categories,
    "images": filtered_images,
    "annotations": filtered_annotations,
}

with open(OUT_JSON, "w") as f:
    json.dump(filtered_data, f)

print(f"Wrote filtered json to {OUT_JSON} "
      f"({len(filtered_images)} images, {len(filtered_annotations)} annotations)")

# Save the final selected id list for the download step
with open(os.path.join(BASE, "selected_ids.json"), "w") as f:
    json.dump(sorted(final_selected), f)

print("Done with selection step.")
