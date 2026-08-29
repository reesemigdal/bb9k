#!/usr/bin/env python3
"""Merge the downloaded, already-selected images (select.py + download.py)
into the final YOLO-ready dataset at ~/data/bb9k/combined-animal-full/.

Unlike train/combined/prepare_data.py, the inputs here (`caltech_selected
.json` / `oregon_selected.json`) are already: filtered to a single `animal`
category, tagged `small_mammal`, and free of degenerate/border-cropped
annotations -- select.py did all of that against the *full* source pools
before any image was downloaded. This script just needs to merge the two,
symlink the (now-downloaded) images, run convert_coco, and split train/val.

No oversampling here (unlike train/combined/prepare_data.py's --sm-frac):
select.py already chose every small-mammal image plus a random half-as-many
"other" images, so the whole selected pool is 2:1 (small-mammal:other) by
construction, from distinct real images -- a plain random 90/10 split
naturally preserves that ratio in both train.txt and val.txt without any
duplicate lines.
"""
import argparse
import json
import os
import random
import shutil
from pathlib import Path

from ultralytics.data.converter import convert_coco

DATA_ROOT = Path.home() / "data/bb9k"
COMBINED_DIR = DATA_ROOT / "combined-animal-full"
IMAGES_DIR = COMBINED_DIR / "images"
LABELS_DIR = COMBINED_DIR / "labels"
COMBINED_JSON = COMBINED_DIR / "combined_animal_yolo.json"

SOURCES = [
    {"prefix": "caltech", "dir": DATA_ROOT / "caltech-camera-traps", "manifest": COMBINED_DIR / "caltech_selected.json"},
    {"prefix": "oregon", "dir": DATA_ROOT / "oregon-critters", "manifest": COMBINED_DIR / "oregon_selected.json"},
]

ID_OFFSET = 10_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    merged_images = []
    merged_annotations = []
    categories = None
    symlinks = []

    for index, source in enumerate(SOURCES):
        images_dir = source["dir"] / "images"
        with open(source["manifest"]) as f:
            data = json.load(f)

        images = [im for im in data["images"] if (images_dir / im["file_name"]).exists()]
        missing = len(data["images"]) - len(images)
        if missing:
            print(f"{source['prefix']}: warning: {missing} selected images still missing on disk, skipping them")

        # select.py's manifests kept each source's original (non-int) image ids --
        # caltech's are UUID strings, oregon's are nested S3-path strings -- since
        # they never went through that source's own fix_ids_for_yolo.py. Remap to
        # sequential ints (offset per source to stay unique across the merge),
        # same as fix_ids_for_yolo.py does, preserving the original as original_id.
        offset = index * ID_OFFSET
        old_to_new_id = {im["id"]: i + 1 + offset for i, im in enumerate(images)}
        kept_ids = set(old_to_new_id)

        for im in images:
            prefixed_name = f"{source['prefix']}__{im['file_name']}"
            symlinks.append((IMAGES_DIR / prefixed_name, images_dir / im["file_name"]))
            im["original_file_name"] = im["file_name"]
            im["original_id"] = im["id"]
            im["file_name"] = prefixed_name
            im["id"] = old_to_new_id[im["original_id"]]
        merged_images.extend(images)

        anns = [a for a in data["annotations"] if a["image_id"] in kept_ids]
        for i, a in enumerate(anns):
            a["image_id"] = old_to_new_id[a["image_id"]]
            a["original_id"] = a["id"]
            a["id"] = i + 1 + offset
            a.setdefault("iscrowd", 0)
            if "area" not in a:
                w, h = a["bbox"][2], a["bbox"][3]
                a["area"] = w * h
        merged_annotations.extend(anns)

        if categories is None:
            categories = data["categories"]
        else:
            assert categories == data["categories"]

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

    names_by_id = {c["id"] - 1: c["name"] for c in categories}
    data_yaml = COMBINED_DIR / "data.yaml"
    data_yaml.write_text(
        "train: train.txt\nval: val.txt\nnames:\n{}\n".format(
            "\n".join(f"  {i}: {name}" for i, name in sorted(names_by_id.items()))
        )
    )
    print(f"wrote {data_yaml}")


if __name__ == "__main__":
    main()
