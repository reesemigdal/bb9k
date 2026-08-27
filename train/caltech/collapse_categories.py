import json
import os

BASE = "/home/ANT.AMAZON.COM/jmigdal/data/bb9k/caltech-camera-traps"
IN_JSON = os.path.join(BASE, "caltech_bboxes_20200316_filtered_20k.json")
OUT_JSON = os.path.join(BASE, "caltech_bboxes_20200316_filtered_20k_animal.json")
IMAGES_DIR = os.path.join(BASE, "images")

REMOVE_CATS = {"car", "empty", "insect"}

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

kept_image_ids = {a["image_id"] for a in kept_annotations}
img_by_id = {im["id"]: im for im in images}
kept_images = [img_by_id[iid] for iid in kept_image_ids if iid in img_by_id]
removed_image_ids = set(img_by_id.keys()) - kept_image_ids

print(f"Images: {len(images)} -> {len(kept_images)} (removed {len(removed_image_ids)})")

# Relabel all remaining annotations to a single "animal" category, id=1
for a in kept_annotations:
    a["category_id"] = 1

new_categories = [{"id": 1, "name": "animal", "supercategory": "animal"}]

filtered_data = {
    "info": data.get("info", {}),
    "categories": new_categories,
    "images": kept_images,
    "annotations": kept_annotations,
}

with open(OUT_JSON, "w") as f:
    json.dump(filtered_data, f)

print(f"Wrote {OUT_JSON} ({len(kept_images)} images, {len(kept_annotations)} annotations, 1 category)")

# Remove now-unreferenced image files from disk
removed_files = 0
missing_files = 0
for iid in removed_image_ids:
    im = img_by_id[iid]
    path = os.path.join(IMAGES_DIR, im["file_name"])
    if os.path.exists(path):
        os.remove(path)
        removed_files += 1
    else:
        missing_files += 1

print(f"Deleted {removed_files} image files from disk (missing already: {missing_files})")
