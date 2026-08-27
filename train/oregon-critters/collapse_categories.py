import json
import os

BASE = "/home/ANT.AMAZON.COM/jmigdal/data/bb9k/oregon-critters"
IN_JSON = os.path.join(BASE, "oregon_critters_filtered_20k.json")
OUT_JSON = os.path.join(BASE, "oregon_critters_filtered_20k_animal.json")

# "empty" is defensive here (selection already excluded empty-only images); "human" is
# dropped by design: we train a single "animal" detector, but keep human images in the
# dataset as background/negative examples rather than deleting them, since removing the
# human box (not the image) is what was asked for.
REMOVE_CATS = {"empty", "human"}

with open(IN_JSON) as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]
categories = data["categories"]

cat_name_by_id = {c["id"]: c["name"] for c in categories}
remove_cat_ids = {cid for cid, name in cat_name_by_id.items() if name in REMOVE_CATS}

print(f"Removing category ids/names: {[(cid, cat_name_by_id[cid]) for cid in remove_cat_ids]}")

kept_annotations = [a for a in annotations if a["category_id"] not in remove_cat_ids]
removed_ann_count = len(annotations) - len(kept_annotations)
print(f"Annotations: {len(annotations)} -> {len(kept_annotations)} (removed {removed_ann_count})")

# Relabel all remaining annotations to a single "animal" category, id=1
for a in kept_annotations:
    a["category_id"] = 1

new_categories = [{"id": 1, "name": "animal", "supercategory": "animal"}]

# Unlike Caltech's collapse step, images are NOT deleted here even if they end up with
# zero annotations: human-only images are meant to survive as background/negative
# examples for training.
annotated_image_ids = {a["image_id"] for a in kept_annotations}
background_count = len(images) - len(annotated_image_ids)
print(f"Images: {len(images)} total, {len(annotated_image_ids)} with an animal box, {background_count} background")

filtered_data = {
    "info": data.get("info", {}),
    "categories": new_categories,
    "images": images,
    "annotations": kept_annotations,
}

with open(OUT_JSON, "w") as f:
    json.dump(filtered_data, f)

print(f"Wrote {OUT_JSON} ({len(images)} images, {len(kept_annotations)} annotations, 1 category)")
