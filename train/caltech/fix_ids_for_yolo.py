import json
import os

BASE = "/home/ANT.AMAZON.COM/jmigdal/data/bb9k/caltech-camera-traps"
IN_JSON = os.path.join(BASE, "caltech_bboxes_20200316_filtered_20k_animal.json")
OUT_JSON = os.path.join(BASE, "caltech_bboxes_20200316_filtered_20k_animal_yolo.json")

with open(IN_JSON) as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]
categories = data["categories"]

# Remap string/UUID image ids -> sequential ints, preserve original id for traceability
old_to_new_img_id = {}
for i, im in enumerate(images, start=1):
    old_to_new_img_id[im["id"]] = i

for im in images:
    im["original_id"] = im["id"]
    im["id"] = old_to_new_img_id[im["original_id"]]

for j, a in enumerate(annotations, start=1):
    a["original_image_id"] = a["image_id"]
    a["image_id"] = old_to_new_img_id[a["original_image_id"]]
    a["original_id"] = a["id"]
    a["id"] = j
    # required by ultralytics converter / COCO spec, safe defaults
    if "iscrowd" not in a:
        a["iscrowd"] = 0
    if "area" not in a:
        w, h = a["bbox"][2], a["bbox"][3]
        a["area"] = w * h

out = {
    "info": data.get("info", {}),
    "categories": categories,
    "images": images,
    "annotations": annotations,
}

with open(OUT_JSON, "w") as f:
    json.dump(out, f)

print(f"Wrote {OUT_JSON}: {len(images)} images, {len(annotations)} annotations")

# sanity checks
img_ids = set(im["id"] for im in images)
assert len(img_ids) == len(images), "duplicate image ids after remap"
assert all(isinstance(im["id"], int) for im in images)
assert all(isinstance(a["image_id"], int) and isinstance(a["id"], int) for a in annotations)
assert all(a["image_id"] in img_ids for a in annotations)
cat_ids = set(c["id"] for c in categories)
assert all(a["category_id"] in cat_ids for a in annotations)
print("Sanity checks passed.")
