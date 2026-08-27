#!/usr/bin/env python3
"""Convert the Oregon Critters COCO json into a YOLO-ready dataset.

Reads oregon_critters_filtered_20k_animal_yolo.json (see
train/oregon-critters/README.md for how that json was built) and writes, all
directly under DATA_DIR (as siblings of images/), using only relative paths
so the whole directory stays portable if it's ever moved or synced elsewhere:
  - labels/*.txt          YOLO-format labels, one per image, via convert_coco()
  - train.txt / val.txt   lists of "./images/..."-relative image paths per split
  - data.yaml             Ultralytics dataset config pointing at the above

labels/ must live next to images/ because Ultralytics finds each label by
string-replacing "/images/" with "/labels/" in the image path (see
ultralytics.data.utils.img2label_paths) — it is not looked up via any path
in data.yaml. train.txt/val.txt entries must start with "./" — Ultralytics'
loader only rewrites relative paths with that exact prefix, replacing it
with the txt file's own directory (ultralytics.data.base.get_img_files); a
bare "images/..." line would instead be resolved against the process's cwd
at train time. data.yaml omits `path:`, which defaults to its own directory
for the same reason. Images are already flat in DATA_DIR/images/ and are
left untouched; the train/val split is expressed as file lists rather than
by moving images around.

Unlike Caltech, this dataset intentionally includes background images (human
sightings with their box stripped by collapse_categories.py, so zero
"animal" annotations remain). convert_coco() only ever writes a label file
for images that have at least one annotation, so after it runs we explicitly
write an empty *.txt for every such image — otherwise Ultralytics still
trains on them correctly (a missing label file is treated the same as an
empty one), but reports them as "missing" instead of "background" in its
scan summary, which looks like a bug when it isn't one.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

from ultralytics.data.converter import convert_coco

DATA_DIR = Path.home() / "data/bb9k/oregon-critters"
SRC_JSON = DATA_DIR / "oregon_critters_filtered_20k_animal_yolo.json"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(SRC_JSON) as f:
        data = json.load(f)

    images = [im for im in data["images"] if (IMAGES_DIR / im["file_name"]).exists()]
    missing = len(data["images"]) - len(images)
    if missing:
        print(f"warning: {missing} images listed in json are missing on disk, skipping them")
    image_ids = {im["id"] for im in images}
    annotations = [a for a in data["annotations"] if a["image_id"] in image_ids]

    convert_dir = DATA_DIR / "_convert_coco_tmp"
    if convert_dir.exists():
        shutil.rmtree(convert_dir)
    ann_dir = convert_dir / "annotations"
    ann_dir.mkdir(parents=True)
    with open(ann_dir / "instances_all.json", "w") as f:
        json.dump({**data, "images": images, "annotations": annotations}, f)

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

    background_count = 0
    for im in images:
        label_path = (LABELS_DIR / im["file_name"]).with_suffix(".txt")
        if not label_path.exists():
            label_path.touch()
            background_count += 1
    print(f"wrote {len(list(LABELS_DIR.glob('*.txt')))} label files to {LABELS_DIR} ({background_count} background)")

    rng = random.Random(args.seed)
    file_names = [im["file_name"] for im in images]
    rng.shuffle(file_names)
    n_val = max(1, int(len(file_names) * args.val_frac))
    val_names, train_names = file_names[:n_val], file_names[n_val:]

    for split, names in [("train", train_names), ("val", val_names)]:
        list_path = DATA_DIR / f"{split}.txt"
        with open(list_path, "w") as f:
            f.write("\n".join(f"./images/{n}" for n in names) + "\n")
        print(f"{split}: {len(names)} images -> {list_path}")

    names_by_id = {c["id"] - 1: c["name"] for c in data["categories"]}
    data_yaml = DATA_DIR / "data.yaml"
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
