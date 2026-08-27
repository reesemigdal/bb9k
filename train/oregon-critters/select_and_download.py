import json
import os
import random
import re
from collections import defaultdict

random.seed(42)

# Some (not all) sources encode a burst/frame index as "(N)" in the filename, e.g.
# ".../10831-1__2019-08-07__01-08-48(7)_ho.JPG" is frame 7 of a burst. Only present in
# COA_2019/COA_2021/HJA_GRID/ORSNAP (100% of those); absent everywhere else (COA_2020,
# DUNES, ESF_*, HJA_MARIE) with no equivalent marker there. Where it's present, keep only
# frame 1 (mirrors Caltech's frame_num==1 filter, avoids near-duplicate burst frames).
# Where it's absent, keep the image — no burst grouping is knowable from the filename.
FRAME_NUM_RE = re.compile(r"\((\d+)\)")


def is_frame_one(file_name):
    matches = FRAME_NUM_RE.findall(file_name)
    return not matches or int(matches[-1]) == 1

BASE = "/home/ANT.AMAZON.COM/jmigdal/data/bb9k/oregon-critters"
JSON_PATH = os.path.join(BASE, "oregon_critters.json")
OUT_JSON = os.path.join(BASE, "oregon_critters_filtered_20k.json")

# Broad "small woodland mammal" bucket: rabbits, squirrels, rodents, plus mustelids/skunks.
SMALL_ANIMAL_CATS = [
    "leporidae family",
    "california ground squirrel",
    "western gray squirrel",
    "douglas squirrel",
    "humboldt's flying squirrel",
    "townsend's chipmunk",
    "neotoma species",
    "mountain beaver",
    "small mammal",
    "marten",
    "mink",
    "weasel family",
    "western spotted skunk",
    "striped skunk",
]
HUMAN_CAT = "human"
EMPTY_CAT = "empty"

SMALL_ANIMAL_CAP = 10000
HUMAN_CAP = 5000
TOTAL_CAP = 20000

print("Loading json...")
with open(JSON_PATH) as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]
categories = data["categories"]

cat_id_by_name = {c["name"]: c["id"] for c in categories}
small_animal_cat_ids = {cat_id_by_name[n] for n in SMALL_ANIMAL_CATS}
human_cat_id = cat_id_by_name[HUMAN_CAT]
empty_cat_id = cat_id_by_name[EMPTY_CAT]

img_by_id = {im["id"]: im for im in images}

anns_by_image = defaultdict(list)
for a in annotations:
    anns_by_image[a["image_id"]].append(a)

# Images with at least one real (non-"empty") annotation. "empty" images are dropped
# entirely here, at selection time (unlike Caltech, which dropped car/empty/insect later).
non_empty_ids = {
    iid for iid, anns in anns_by_image.items() if any(a["category_id"] != empty_cat_id for a in anns)
}
frame1_ids = {im["id"] for im in images if is_frame_one(im["file_name"])}
usable_ids = non_empty_ids & frame1_ids
print(
    f"Total images: {len(images)}, non-empty: {len(non_empty_ids)}, "
    f"frame-1-or-unmarked: {len(frame1_ids)}, usable (both): {len(usable_ids)}"
)


def images_with_cats(cat_ids, pool):
    return {iid for iid in pool if any(a["category_id"] in cat_ids for a in anns_by_image[iid])}


# Bucket 1: broad small-animal categories, up to 10,000
small_animal_pool = sorted(images_with_cats(small_animal_cat_ids, usable_ids))
random.shuffle(small_animal_pool)
small_animal_selected = set(small_animal_pool[:SMALL_ANIMAL_CAP])
print(f"Small-animal pool: {len(small_animal_pool)}, selected: {len(small_animal_selected)} (cap {SMALL_ANIMAL_CAP})")

# Bucket 2: human images, up to 5,000 (only ~314 exist in the whole dataset)
human_pool = sorted(images_with_cats({human_cat_id}, usable_ids) - small_animal_selected)
random.shuffle(human_pool)
human_selected = set(human_pool[:HUMAN_CAP])
print(f"Human pool: {len(human_pool)}, selected: {len(human_selected)} (cap {HUMAN_CAP})")

# Bucket 3: everything else, filling up to the total cap
rest_pool = sorted(usable_ids - small_animal_selected - human_selected)
random.shuffle(rest_pool)
rest_cap = TOTAL_CAP - len(small_animal_selected) - len(human_selected)
rest_selected = set(rest_pool[:rest_cap])
print(f"Rest pool: {len(rest_pool)}, selected: {len(rest_selected)} (cap {rest_cap})")

final_selected = small_animal_selected | human_selected | rest_selected
print(f"FINAL total selected images: {len(final_selected)} (target {TOTAL_CAP})")

# Flatten file_name to just the basename for local storage (all basenames are unique
# dataset-wide) so images/ stays a single flat directory, like Caltech's. The original
# nested path is kept as source_path — download_images.py needs it to build the S3 key.
filtered_images = []
for iid in final_selected:
    im = dict(img_by_id[iid])
    im["source_path"] = im["file_name"]
    im["file_name"] = os.path.basename(im["file_name"])
    filtered_images.append(im)

filtered_annotations = [a for a in annotations if a["image_id"] in final_selected]

filtered_data = {
    "info": data.get("info", {}),
    "categories": categories,
    "images": filtered_images,
    "annotations": filtered_annotations,
}

with open(OUT_JSON, "w") as f:
    json.dump(filtered_data, f)

print(
    f"Wrote filtered json to {OUT_JSON} "
    f"({len(filtered_images)} images, {len(filtered_annotations)} annotations)"
)

with open(os.path.join(BASE, "selected_ids.json"), "w") as f:
    json.dump(sorted(final_selected), f)

print("Done with selection step.")
