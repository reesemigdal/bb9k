#!/usr/bin/env python3
"""Extend combined-animal-full/ with the Oxford-IIIT Pet dataset (cats +
dogs), producing a new combined dataset at
~/data/bb9k/combined-animal-full-domestic/: a merged COCO json
(combined_animal_yolo.json, single "animal" category) plus the usual
YOLO-ready images/labels/train.txt/val.txt/data.yaml.

Three sources:
- caltech, oregon: reused as-is from train/combined-full/select.py +
  download.py's output (combined-animal-full/{caltech,oregon}_selected.json)
  -- already single-"animal"-category, already border/degenerate-box
  filtered, already tagged `small_mammal`. Not touched here.
- cambridge_iiit: from cambridge_iiit_coco.json (see train/cambridge_iiit/),
  using only each image's `<breed>_whole` annotation (matched by category
  *name* suffix, not the `annotation_type` field -- see that dataset's
  README for why nothing should rely on that field being present) --
  `_face` annotations are dropped entirely, since mixing tight face boxes
  and looser whole-body boxes under one "animal" class would teach the
  detector inconsistent box semantics. Every image with a `_whole`
  annotation is used (7,367 of 7,390 -- the other 23 have no whole-animal
  mask to begin with, see that dataset's README) and every one is tagged
  `small_mammal = True`: cats and dogs are small mammals by any reasonable
  reading of that label, and there's no "other" pool to sample from within
  this dataset the way there was for Caltech/Oregon's wildlife categories,
  so there's no ratio to preserve here -- this dataset's images are simply
  ALL used. No border-margin filtering is applied to cambridge_iiit
  (unlike caltech/oregon's own selection): tightly-framed pet portraits
  legitimately touch the frame edge far more often than camera-trap
  wildlife shots, and the ask was explicitly to use all of them.

Because cambridge_iiit's images skew the small-mammal fraction well past
the 2/3 target train/combined-full/select.py aimed for (deliberately --
see above), this script does not try to hit any particular ratio; the
final train/val split is just a plain random split of everything, same
mechanism as train/combined-full/prepare_data.py but without that
script's oversampling equivalent (there isn't one here -- no oversampling
was used there either, this dataset just doesn't attempt a target ratio
at all).
"""
import argparse
import json
import os
import random
import shutil
from pathlib import Path

from ultralytics.data.converter import convert_coco

DATA_ROOT = Path.home() / "data/bb9k"
OLD_FULL_DIR = DATA_ROOT / "combined-animal-full"
COMBINED_DIR = DATA_ROOT / "combined-animal-full-domestic"
IMAGES_DIR = COMBINED_DIR / "images"
LABELS_DIR = COMBINED_DIR / "labels"
COMBINED_JSON = COMBINED_DIR / "combined_animal_yolo.json"

ID_OFFSET = 10_000_000

SOURCES = [
    {"prefix": "caltech", "images_dir": DATA_ROOT / "caltech-camera-traps" / "images", "manifest": OLD_FULL_DIR / "caltech_selected.json"},
    {"prefix": "oregon", "images_dir": DATA_ROOT / "oregon-critters" / "images", "manifest": OLD_FULL_DIR / "oregon_selected.json"},
    {"prefix": "cambridge_iiit", "images_dir": DATA_ROOT / "cambridge_iiit" / "images", "manifest": DATA_ROOT / "cambridge_iiit" / "cambridge_iiit_coco.json"},
]


def normalize_cambridge(data):
    """cambridge_iiit_coco.json -> (images, annotations) in the same shape as
    caltech/oregon's *_selected.json: single "animal" category (id 1),
    small_mammal tagged, only _whole annotations kept."""
    whole_cat_ids = {c["id"] for c in data["categories"] if c["name"].endswith("_whole")}
    anns_by_image = {}
    for a in data["annotations"]:
        if a["category_id"] in whole_cat_ids:
            anns_by_image.setdefault(a["image_id"], []).append(a)

    images = [im for im in data["images"] if im["id"] in anns_by_image]
    for im in images:
        im["small_mammal"] = True
        # cambridge_iiit_coco.json's own original_id (the breed_n.jpg stem) is more
        # meaningful than its sequential int id -- keep it, the merge loop below
        # will only fill original_id in if a source doesn't already have one.

    annotations = []
    for im in images:
        for a in anns_by_image[im["id"]]:
            annotations.append({
                "id": a["id"],
                "original_id": a.get("original_id", a["id"]),
                "image_id": a["image_id"],
                "category_id": 1,
                "bbox": a["bbox"],
                "area": a["area"],
                "iscrowd": a.get("iscrowd", 0),
                "segmentation": a.get("segmentation"),
            })
    return images, annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    merged_images = []
    merged_annotations = []
    categories = [{"id": 1, "name": "animal", "supercategory": "animal"}]
    symlinks = []

    for index, source in enumerate(SOURCES):
        images_dir = source["images_dir"]
        with open(source["manifest"]) as f:
            data = json.load(f)

        if source["prefix"] == "cambridge_iiit":
            images, annotations = normalize_cambridge(data)
        else:
            images, annotations = data["images"], data["annotations"]

        images = [im for im in images if (images_dir / im["file_name"]).exists()]
        missing = len(data["images"]) - len(images) if source["prefix"] != "cambridge_iiit" else None
        if missing:
            print(f"{source['prefix']}: warning: {missing} images missing on disk, skipping them")

        # old_to_new_id is keyed by each image's CURRENT id (whatever type/source it
        # already is -- string for caltech/oregon, int for cambridge_iiit), captured
        # before anything below mutates it.
        offset = index * ID_OFFSET
        old_to_new_id = {im["id"]: i + 1 + offset for i, im in enumerate(images)}
        kept_ids = set(old_to_new_id)

        for im in images:
            prefixed_name = f"{source['prefix']}__{im['file_name']}"
            symlinks.append((IMAGES_DIR / prefixed_name, images_dir / im["file_name"]))
            prior_id = im["id"]
            im.setdefault("original_id", prior_id)  # cambridge_iiit already has a better one (the stem); don't clobber it
            im["original_file_name"] = im["file_name"]
            im["file_name"] = prefixed_name
            im["id"] = old_to_new_id[prior_id]
        merged_images.extend(images)

        anns = [a for a in annotations if a["image_id"] in kept_ids]
        for i, a in enumerate(anns):
            a.setdefault("original_id", a["id"])
            a["image_id"] = old_to_new_id[a["image_id"]]
            a["id"] = i + 1 + offset
            a.setdefault("iscrowd", 0)
            if "area" not in a:
                w, h = a["bbox"][2], a["bbox"][3]
                a["area"] = w * h
        merged_annotations.extend(anns)

        n_sm = sum(im["small_mammal"] for im in images)
        print(f"{source['prefix']}: {len(images)} images ({n_sm} small_mammal), {len(anns)} annotations")

    if IMAGES_DIR.exists():
        shutil.rmtree(IMAGES_DIR)
    IMAGES_DIR.mkdir(parents=True)
    for link_path, target_path in symlinks:
        link_path.symlink_to(os.path.relpath(target_path, start=link_path.parent))
    print(f"symlinked {len(symlinks)} images into {IMAGES_DIR}")

    merged = {"info": {}, "categories": categories, "images": merged_images, "annotations": merged_annotations}
    with open(COMBINED_JSON, "w") as f:
        json.dump(merged, f)
    print(f"wrote {COMBINED_JSON}: {len(merged_images)} images, {len(merged_annotations)} annotations")

    convert_dir = COMBINED_DIR / "_convert_coco_tmp"
    if convert_dir.exists():
        shutil.rmtree(convert_dir)
    ann_dir = convert_dir / "annotations"
    ann_dir.mkdir(parents=True)
    shutil.copy(COMBINED_JSON, ann_dir / "instances_all.json")

    convert_coco(labels_dir=str(ann_dir), save_dir=str(convert_dir / "out"), use_segments=False, use_keypoints=False, cls91to80=False)

    if LABELS_DIR.exists():
        shutil.rmtree(LABELS_DIR)
    shutil.move(str(convert_dir / "out" / "labels" / "all"), str(LABELS_DIR))
    shutil.rmtree(convert_dir)
    print(f"wrote {len(list(LABELS_DIR.glob('*.txt')))} label files to {LABELS_DIR}")

    rng = random.Random(args.seed)
    items = [(im["file_name"], im["small_mammal"]) for im in merged_images]
    rng.shuffle(items)
    n_val = max(1, int(len(items) * args.val_frac))
    val_items, train_items = items[:n_val], items[n_val:]

    for split, split_items in [("train", train_items), ("val", val_items)]:
        names = [name for name, _ in split_items]
        n_sm = sum(sm for _, sm in split_items)
        with open(COMBINED_DIR / f"{split}.txt", "w") as f:
            f.write("\n".join(f"./images/{n}" for n in names) + "\n")
        print(f"{split}: {len(names)} images ({n_sm} small_mammal, {n_sm / len(names):.1%}) -> {COMBINED_DIR / f'{split}.txt'}")

    data_yaml = COMBINED_DIR / "data.yaml"
    data_yaml.write_text("train: train.txt\nval: val.txt\nnames:\n  0: animal\n")
    print(f"wrote {data_yaml}")


if __name__ == "__main__":
    main()
