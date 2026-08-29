#!/usr/bin/env python3
"""Select images for the "full pool" combined dataset — JSON-only, no image
downloads or file I/O against images/ at all, so this is cheap/fast to
re-run while tuning selection parameters.

Unlike train/combined/prepare_data.py (which merges each source's already
-built ~20k-image `*_animal_yolo.json`), this selects directly from each
source's *full* raw annotation json (63,025 images for Caltech, 99,909 for
Oregon) — the ~20k files are themselves just one particular random subsample
of these, built by each dataset's own select_and_download.py.

Per source, in order:
1. Burst/frame de-dup filter, replicating each dataset's own
   select_and_download.py logic (not encoded anywhere else, so it has to be
   redone here against the raw json): Caltech keeps only `frame_num == 1`;
   Oregon keeps images with no `(N)` burst marker in the filename, or where
   the marker is `(1)` (see train/oregon-critters/select_and_download.py's
   FRAME_NUM_RE for why this isn't a uniform rule across the whole dataset).
2. Drop annotations in junk categories (Caltech: car/empty/insect; Oregon:
   empty/human — same sets each dataset's own collapse_categories.py uses),
   then drop any image left with zero annotations. Unlike Oregon's own
   pipeline (which deliberately keeps human-only images as background
   negatives), this drops them — we're doing our own curated small-mammal-
   weighted selection here, not reproducing Oregon's full-dataset design.
3. Tag `small_mammal` per image against our established name sets (see
   train/combined/prepare_data.py's SOURCES for where these came from).
4. Drop degenerate (zero width/height) annotations, then drop any image
   left with zero annotations as a result (same as train/combined/
   prepare_data.py).
5. Drop border-cropped images: any image with an annotation within
   `--border-margin` (default 5%) of an edge is dropped entirely — computed
   straight from each image's `bbox`/`width`/`height` in the json, which is
   exactly why this step can run before any image is downloaded.

Selection: every surviving small-mammal image is kept. Non-small-mammal
("other") images are randomly sampled down to half that count, so the
final pool is small-mammal : other = 2 : 1 (2/3 : 1/3) by construction —
no oversampling/duplication needed downstream, these are all distinct
images.

Writes two manifests (one per source) into COMBINED_DIR, in the same
shape as each source's own `*_animal_yolo.json` (single `animal` category,
int ids, `original_id`/`file_name`/`source_path` preserved) so
train/combined-full/download.py and prepare_data.py can consume them
directly. Does not touch either source's own `*_filtered_20k*` files.
"""
import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

DATA_ROOT = Path.home() / "data/bb9k"
COMBINED_DIR = DATA_ROOT / "combined-animal-full"

CALTECH_JSON = DATA_ROOT / "caltech-camera-traps" / "caltech_bboxes_20200316.json"
CALTECH_REMOVE_CATS = {"car", "empty", "insect"}
CALTECH_SM = {"rabbit", "squirrel", "rodent"}

OREGON_JSON = DATA_ROOT / "oregon-critters" / "oregon_critters.json"
OREGON_REMOVE_CATS = {"empty", "human"}
OREGON_SM = {
    "townsend's chipmunk", "douglas squirrel", "leporidae family",
    "humboldt's flying squirrel", "mountain beaver", "neotoma species",
    "california ground squirrel", "western gray squirrel", "small mammal",
}

FRAME_NUM_RE = re.compile(r"\((\d+)\)")


def oregon_is_frame_one(file_name):
    matches = FRAME_NUM_RE.findall(file_name)
    return not matches or int(matches[-1]) == 1


def near_border(bbox, img_width, img_height, margin_frac):
    x, y, w, h = bbox
    margin_x = margin_frac * img_width
    margin_y = margin_frac * img_height
    return (
        x <= margin_x
        or y <= margin_y
        or (img_width - (x + w)) <= margin_x
        or (img_height - (y + h)) <= margin_y
    )


def load_and_filter(json_path, frame_filter, remove_cats, sm_names, border_margin, label):
    with open(json_path) as f:
        d = json.load(f)
    id_to_name = {c["id"]: c["name"] for c in d["categories"]}
    n_raw = len(d["images"])

    images = [im for im in d["images"] if frame_filter(im)]
    n_frame = len(images)
    kept_ids = {im["id"] for im in images}

    anns = [a for a in d["annotations"] if a["image_id"] in kept_ids]
    anns = [a for a in anns if id_to_name.get(a["category_id"]) not in remove_cats]
    anns = [a for a in anns if a["bbox"][2] > 0 and a["bbox"][3] > 0]
    valid_ids = {a["image_id"] for a in anns}
    images = [im for im in images if im["id"] in valid_ids]
    n_after_cats = len(images)

    anns_by_img = {}
    for a in anns:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    for im in images:
        cats = {id_to_name[a["category_id"]] for a in anns_by_img[im["id"]]}
        im["small_mammal"] = bool(cats & sm_names)

    border_ids = {
        im["id"]
        for im in images
        if any(near_border(a["bbox"], im["width"], im["height"], border_margin) for a in anns_by_img[im["id"]])
    }
    images = [im for im in images if im["id"] not in border_ids]
    kept_ids = {im["id"] for im in images}
    anns = [a for a in anns if a["image_id"] in kept_ids]
    n_final = len(images)

    # record species identity for reporting, then actually collapse to the single
    # "animal" class the output categories list claims -- annotations kept their
    # original species category_id up to this point, which would otherwise leave the
    # manifest inconsistent with its own categories list (and break convert_coco,
    # which trusts category_id to select the class index).
    for a in anns:
        a["species_name"] = id_to_name[a["category_id"]]
        a["category_id"] = 1

    n_sm = sum(1 for im in images if im["small_mammal"])
    print(
        f"{label}: raw={n_raw} -> frame-filtered={n_frame} -> junk/degenerate-filtered={n_after_cats} "
        f"-> border-filtered={n_final} ({n_after_cats - n_final} dropped for border) "
        f"| small_mammal={n_sm} ({n_sm / n_final:.1%}), other={n_final - n_sm}"
    )
    return images, anns, [{"id": 1, "name": "animal", "supercategory": "animal"}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--border-margin", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    COMBINED_DIR.mkdir(parents=True, exist_ok=True)

    caltech_images, caltech_anns, categories = load_and_filter(
        CALTECH_JSON,
        frame_filter=lambda im: im.get("frame_num") == 1,
        remove_cats=CALTECH_REMOVE_CATS,
        sm_names=CALTECH_SM,
        border_margin=args.border_margin,
        label="caltech",
    )
    oregon_images, oregon_anns, _ = load_and_filter(
        OREGON_JSON,
        frame_filter=lambda im: oregon_is_frame_one(im["file_name"]),
        remove_cats=OREGON_REMOVE_CATS,
        sm_names=OREGON_SM,
        border_margin=args.border_margin,
        label="oregon",
    )

    # Oregon's raw file_name is a full nested S3 key with its own "images/" segment
    # partway through (breaks Ultralytics' label-lookup convention if kept) -- flatten
    # to basename for local storage, same as train/oregon-critters/select_and_download.py,
    # keeping the original as source_path for the download step.
    for im in oregon_images:
        im["source_path"] = im["file_name"]
        im["file_name"] = os.path.basename(im["file_name"])

    rng = random.Random(args.seed)

    def pool(images, anns, source):
        anns_by_img = {}
        for a in anns:
            anns_by_img.setdefault(a["image_id"], []).append(a)
        return [(source, im, anns_by_img[im["id"]]) for im in images]

    all_items = pool(caltech_images, caltech_anns, "caltech") + pool(oregon_images, oregon_anns, "oregon")
    sm_items = [it for it in all_items if it[1]["small_mammal"]]
    other_items = [it for it in all_items if not it[1]["small_mammal"]]

    n_fill = round(len(sm_items) / 2)
    rng.shuffle(other_items)
    fill_items = other_items[:n_fill]
    if len(fill_items) < n_fill:
        print(f"warning: only {len(fill_items)} 'other' images available, wanted {n_fill}")

    selected = sm_items + fill_items
    print(
        f"\nselected: {len(sm_items)} small_mammal + {len(fill_items)} other "
        f"= {len(selected)} total ({len(sm_items) / len(selected):.1%} small_mammal)"
    )

    for source, out_name in [("caltech", "caltech_selected.json"), ("oregon", "oregon_selected.json")]:
        sel_images = [im for src, im, _ in selected if src == source]
        sel_ids = {im["id"] for im in sel_images}
        sel_anns = (caltech_anns if source == "caltech" else oregon_anns)
        sel_anns = [a for a in sel_anns if a["image_id"] in sel_ids]
        out = {"info": {}, "categories": categories, "images": sel_images, "annotations": sel_anns}
        with open(COMBINED_DIR / out_name, "w") as f:
            json.dump(out, f)
        n_sm = sum(1 for im in sel_images if im["small_mammal"])
        print(f"  {source}: {len(sel_images)} images ({n_sm} small_mammal), {len(sel_anns)} annotations -> {COMBINED_DIR / out_name}")

    print("\nspecies breakdown of the final selection (annotation count / distinct-image count):")
    all_sel_anns = [
        a
        for src, im, anns in selected
        for a in anns
    ]
    ann_counts = Counter(a["species_name"] for a in all_sel_anns)
    img_counts = Counter()
    seen = set()
    for a in all_sel_anns:
        key = (a["image_id"], a["species_name"])
        if key not in seen:
            seen.add(key)
            img_counts[a["species_name"]] += 1
    header = f"  {'species':30s} {'ann_count':>10s} {'img_count':>10s}"
    print(header)
    for name, cnt in ann_counts.most_common():
        print(f"  {name:30s} {cnt:10d} {img_counts[name]:10d}")


if __name__ == "__main__":
    main()
