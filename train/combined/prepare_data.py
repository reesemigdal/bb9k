#!/usr/bin/env python3
"""Merge the Caltech and Oregon Critters datasets into one YOLO-ready dataset.

Each source dataset already has its own sanitized COCO json
(`*_filtered_20k_animal_yolo.json`, produced by that dataset's own
prepare pipeline — int image/annotation ids, single `animal` category)
and its own flat `images/` directory. The two `images/` dirs are separate
folders on disk, so there's no single directory to point Ultralytics at
and no combined COCO json to convert yet. This script builds both:

- COMBINED_DIR/images/ — populated with symlinks (not copies; the source
  images total ~39 GB) into each source dataset's images/ dir, targeted by
  *relative* path (so the directory stays valid if data/bb9k/ is ever seen
  through a different absolute mount point, e.g. a container's home dir vs.
  the host's). Every filename is prefixed with its source name
  (`caltech__...`, `oregon__...`) so the merge is collision-proof even
  though the two sets happen not to overlap today, and so a file's origin
  stays obvious.
- COMBINED_DIR/combined_animal_yolo.json — a merged COCO json, analogous to
  each source's own `*_animal_yolo.json`. Each source's images/annotations
  are copied in with `file_name` rewritten to the prefixed symlink name,
  and image/annotation ids offset by a large per-source constant (source
  ids are already sequential ints starting at 1, well under the offset) so
  ids stay unique across the merge. Categories are identical (`{1: animal}`)
  across both sources and are asserted as such, then taken as-is. Each
  image also gets a `small_mammal: bool` field (see below). Annotations
  with zero width/height (a handful exist in Oregon's source data — a
  flagged-but-not-boxed animal) are dropped, and any image left with no
  annotations as a result is dropped entirely, same as the collapse step
  each source's own pipeline already applies to `car`/`empty`/`insect`.
  Separately, any image where *any* annotation has a side within
  `--border-margin` (default 5%) of that edge of the image is dropped
  outright (image + all its annotations, not just the offending box) —
  a camera-cropped animal's visible box understates its true extent, so
  the box (and the training signal from anything else that happened to
  share the frame) isn't trustworthy.

From there this follows the same convert_coco -> labels/ -> data.yaml
recipe as train/caltech/prepare_data.py (see that file's docstring for why
labels/ must be a sibling of images/ and why train.txt/val.txt entries
need the "./" prefix). Run this after both source datasets' own prepare
pipelines have produced their `_yolo.json` and `images/`.

Small-mammal oversampling
--------------------------
The `*_animal_yolo.json` files collapsed every species down to one
`animal` class, so species identity has to be recovered from each
source's pre-collapse, species-level json (`SOURCES[*]["species_json"]`)
via the `original_id` field `fix_ids_for_yolo.py` preserved on every
image. An image is tagged `small_mammal` if any of its species-level
annotations name a class in that source's `small_mammal_names` set
(rodents, lagomorphs, and Oregon's generic "small mammal" catch-all;
adjust the sets below if that definition should change).

val.txt is one line per unique image, at the sources' natural class
mix — oversampling only the training set keeps validation metrics
measuring performance on the true deployment distribution rather than
on an inflated one. train.txt is built from the *unique* post-split
image list: non-small-mammal images appear once each; small-mammal
images are repeated (duplicate lines pointing at the same symlinked
file and label — no pixels are duplicated on disk) enough times to
hit `--sm-frac` of the training set (default 2/3), spread as evenly as
possible across the small-mammal images via a base repeat count plus a
randomly-chosen subset getting one extra copy. If `--sm-frac` is ever
set at or below the natural fraction, small-mammal images are instead
randomly downsampled (without repeats) to hit it.
"""
import argparse
import json
import os
import random
import shutil
from pathlib import Path

from ultralytics.data.converter import convert_coco

DATA_ROOT = Path.home() / "data/bb9k"
COMBINED_DIR = DATA_ROOT / "combined-animal"
IMAGES_DIR = COMBINED_DIR / "images"
LABELS_DIR = COMBINED_DIR / "labels"
COMBINED_JSON = COMBINED_DIR / "combined_animal_yolo.json"

SOURCES = [
    {
        "prefix": "caltech",
        "dir": DATA_ROOT / "caltech-camera-traps",
        "json": "caltech_bboxes_20200316_filtered_20k_animal_yolo.json",
        "species_json": "caltech_bboxes_20200316_filtered_20k.json",
        "small_mammal_names": {"rabbit", "squirrel", "rodent"},
    },
    {
        "prefix": "oregon",
        "dir": DATA_ROOT / "oregon-critters",
        "json": "oregon_critters_filtered_20k_animal_yolo.json",
        "species_json": "oregon_critters_filtered_20k.json",
        "small_mammal_names": {
            "townsend's chipmunk", "douglas squirrel", "leporidae family",
            "humboldt's flying squirrel", "mountain beaver", "neotoma species",
            "california ground squirrel", "western gray squirrel", "small mammal",
        },
    },
]

ID_OFFSET = 10_000_000  # per source; each source's own ids are << this


def species_level_categories(source):
    """image_id -> set of species-level class names, from the pre-collapse json."""
    with open(source["dir"] / source["species_json"]) as f:
        species = json.load(f)
    id_to_name = {c["id"]: c["name"] for c in species["categories"]}
    img_to_cats = {}
    for a in species["annotations"]:
        img_to_cats.setdefault(a["image_id"], set()).add(id_to_name.get(a["category_id"]))
    return img_to_cats


def near_border(bbox, img_width, img_height, margin_frac):
    """True if any side of `bbox` (COCO [x, y, w, h], absolute pixels) sits within
    margin_frac of the corresponding image edge — a proxy for animals cropped by
    the camera frame, whose visible extent isn't their true extent."""
    x, y, w, h = bbox
    margin_x = margin_frac * img_width
    margin_y = margin_frac * img_height
    return (
        x <= margin_x
        or y <= margin_y
        or (img_width - (x + w)) <= margin_x
        or (img_height - (y + h)) <= margin_y
    )


def build_repeated_list(names, target_count, rng):
    """Return `names` repeated (or, if target_count is smaller, downsampled without
    repeats) as evenly as possible to reach exactly target_count entries."""
    n = len(names)
    if n == 0 or target_count <= 0:
        return []
    if target_count <= n:
        return rng.sample(names, target_count)
    base, remainder = divmod(target_count, n)
    shuffled = rng.sample(names, n)
    extra = set(shuffled[:remainder])
    result = []
    for name in names:
        result.extend([name] * (base + (1 if name in extra else 0)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--sm-frac", type=float, default=2 / 3, help="target fraction of TRAIN images that are small-mammal, via oversampling")
    parser.add_argument("--border-margin", type=float, default=0.05, help="drop images with any annotation within this fraction of the image border")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    merged_images = []
    merged_annotations = []
    categories = None
    symlinks = []  # (link_path, target_path)

    for index, source in enumerate(SOURCES):
        src_dir = source["dir"]
        images_dir = src_dir / "images"
        with open(src_dir / source["json"]) as f:
            data = json.load(f)

        images = [im for im in data["images"] if (images_dir / im["file_name"]).exists()]
        missing = len(data["images"]) - len(images)
        if missing:
            print(f"{source['prefix']}: warning: {missing} images listed in json are missing on disk, skipping them")

        img_to_species = species_level_categories(source)
        sm_names = source["small_mammal_names"]

        offset = index * ID_OFFSET
        old_to_new_id = {im["id"]: im["id"] + offset for im in images}
        kept_ids = set(old_to_new_id)

        for im in images:
            prefixed_name = f"{source['prefix']}__{im['file_name']}"
            symlinks.append((IMAGES_DIR / prefixed_name, images_dir / im["file_name"]))
            im["original_file_name"] = im["file_name"]
            im["small_mammal"] = bool(img_to_species.get(im["original_id"], set()) & sm_names)
            im["file_name"] = prefixed_name
            im["id"] = old_to_new_id[im["id"]]
        merged_images.extend(images)

        n_sm = sum(im["small_mammal"] for im in images)
        print(f"{source['prefix']}: {n_sm}/{len(images)} images tagged small_mammal ({n_sm / len(images):.1%})")

        anns = [a for a in data["annotations"] if a["image_id"] in kept_ids]
        n_degenerate = sum(1 for a in anns if a["bbox"][2] <= 0 or a["bbox"][3] <= 0)
        if n_degenerate:
            print(f"{source['prefix']}: warning: {n_degenerate} annotations have zero width/height, dropping them")
            anns = [a for a in anns if a["bbox"][2] > 0 and a["bbox"][3] > 0]
        for a in anns:
            a["original_image_id"] = a.get("original_image_id", a["image_id"])
            a["image_id"] = old_to_new_id[a["image_id"]]
            a["id"] = a["id"] + offset
        merged_annotations.extend(anns)

        if categories is None:
            categories = data["categories"]
        else:
            assert categories == data["categories"], (
                f"{source['prefix']} categories {data['categories']} differ from {categories}"
            )

        print(f"{source['prefix']}: {len(images)} images, {len(anns)} annotations")

    valid_ids = {a["image_id"] for a in merged_annotations}
    n_before = len(merged_images)
    dropped = [im for im in merged_images if im["id"] not in valid_ids]
    if dropped:
        print(
            f"dropping {len(dropped)} images left with zero valid annotations "
            f"(all their boxes were the zero-width/zero-height ones above)"
        )
        merged_images = [im for im in merged_images if im["id"] in valid_ids]
        dropped_names = {im["file_name"] for im in dropped}
        symlinks = [(link, target) for link, target in symlinks if link.name not in dropped_names]
    assert len(merged_images) == n_before - len(dropped)

    img_dims = {im["id"]: (im["width"], im["height"]) for im in merged_images}
    border_image_ids = {
        a["image_id"]
        for a in merged_annotations
        if near_border(a["bbox"], *img_dims[a["image_id"]], args.border_margin)
    }
    if border_image_ids:
        print(
            f"dropping {len(border_image_ids)} images with an annotation within "
            f"{args.border_margin:.0%} of the image border"
        )
        border_names = {im["file_name"] for im in merged_images if im["id"] in border_image_ids}
        merged_images = [im for im in merged_images if im["id"] not in border_image_ids]
        merged_annotations = [a for a in merged_annotations if a["image_id"] not in border_image_ids]
        symlinks = [(link, target) for link, target in symlinks if link.name not in border_names]

    if IMAGES_DIR.exists():
        shutil.rmtree(IMAGES_DIR)
    IMAGES_DIR.mkdir(parents=True)
    for link_path, target_path in symlinks:
        # relative, not absolute: an absolute target bakes in DATA_ROOT's current mount
        # path, which breaks the moment this directory is viewed from an environment
        # that mounts the same files under a different absolute prefix (e.g. a
        # container's /home/jmigdal vs. the host's /jmigdal-data). Relative targets
        # only depend on the data/bb9k/* directory layout staying put relative to
        # each other, which is guaranteed since this script controls both sides.
        link_path.symlink_to(os.path.relpath(target_path, start=link_path.parent))
    print(f"symlinked {len(symlinks)} images into {IMAGES_DIR}")

    merged = {
        "info": {},
        "categories": categories,
        "images": merged_images,
        "annotations": merged_annotations,
    }

    with open(COMBINED_JSON, "w") as f:
        json.dump(merged, f)
    print(f"wrote {COMBINED_JSON}: {len(merged_images)} images, {len(merged_annotations)} annotations")

    convert_dir = COMBINED_DIR / "_convert_coco_tmp"
    if convert_dir.exists():
        shutil.rmtree(convert_dir)
    ann_dir = convert_dir / "annotations"
    ann_dir.mkdir(parents=True)
    shutil.copy(COMBINED_JSON, ann_dir / "instances_all.json")

    convert_coco(
        labels_dir=str(ann_dir),
        save_dir=str(convert_dir / "out"),
        use_segments=False,
        use_keypoints=False,
        cls91to80=False,
    )

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

    val_names = [name for name, _ in val_items]
    val_sm = sum(sm for _, sm in val_items)
    with open(COMBINED_DIR / "val.txt", "w") as f:
        f.write("\n".join(f"./images/{n}" for n in val_names) + "\n")
    print(f"val: {len(val_names)} images ({val_sm} small_mammal, {val_sm / len(val_names):.1%}, natural mix) -> {COMBINED_DIR / 'val.txt'}")

    train_sm_names = [name for name, sm in train_items if sm]
    train_nonsm_names = [name for name, sm in train_items if not sm]
    target_sm = int(round(args.sm_frac / (1 - args.sm_frac) * len(train_nonsm_names)))
    train_sm_repeated = build_repeated_list(train_sm_names, target_sm, rng)
    train_names = train_nonsm_names + train_sm_repeated
    rng.shuffle(train_names)
    with open(COMBINED_DIR / "train.txt", "w") as f:
        f.write("\n".join(f"./images/{n}" for n in train_names) + "\n")
    print(
        f"train: {len(train_names)} images ({len(train_sm_names)} unique small_mammal repeated to "
        f"{len(train_sm_repeated)}, {len(train_sm_repeated) / len(train_names):.1%}; "
        f"{len(train_nonsm_names)} non-small_mammal) -> {COMBINED_DIR / 'train.txt'}"
    )

    names_by_id = {c["id"] - 1: c["name"] for c in categories}
    data_yaml = COMBINED_DIR / "data.yaml"
    data_yaml.write_text(
        "train: train.txt\n"
        "val: val.txt\n"
        "names:\n{}\n".format(
            "\n".join(f"  {i}: {name}" for i, name in sorted(names_by_id.items())),
        )
    )
    print(f"wrote {data_yaml}")


if __name__ == "__main__":
    main()
