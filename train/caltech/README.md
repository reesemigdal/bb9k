# Caltech Camera Traps → YOLOv8 dataset

Builds a 20k-image-max subset of the LILA Caltech Camera Traps dataset, biased
toward rabbits/small rodents, and converts it into a COCO json that Ultralytics'
YOLOv8 tooling can actually train on.

Source data:
- Images: `s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/caltech-unzipped/cct_images/`
  (244,584 objects, ~112 GB total — only a subset is downloaded, see below)
- Annotations: `caltech_bboxes_20200316.json` (COCO-format, 63,025 images / 65,112
  annotations / 22 categories), downloaded from LILA ahead of time into
  `~/data/bb9k/caltech-camera-traps/`.

All scripts here have `BASE` (or `DATA_DIR` in `prepare_data.py`) hardcoded
to `~/data/bb9k/caltech-camera-traps` — that's where the raw json lives and
where all outputs (json files, `images/`, `labels/`, `train.txt`/`val.txt`/
`data.yaml`) are written, all as siblings. Update the path in each script if
the data directory moves.

## Steps (run in this order)

### 1. `select_and_download.py` — pick which images to use

Reads `caltech_bboxes_20200316.json` and selects images under these rules:
- Only consider images where `frame_num == 1` (first frame of each camera-trap burst).
- Take all `rabbit` images, up to a cap of 10,000.
- If that's under 10,000, fill the remainder (up to the same 10,000 cap) with a
  random sample of other small-rodent categories (`squirrel`, `rodent`).
- Then take a random sample of an additional 10,000 images from everything else
  (frame_num==1, not already selected).
- Max 20,000 images total. Random sampling uses a fixed seed (42) for reproducibility.

In our run: rabbit (2,195) + squirrel/rodent (1,688, all of them — pool was smaller
than the 10k cap) = 3,883 small-rodent images, + 10,000 random additional =
**13,883 images total**.

Outputs:
- `caltech_bboxes_20200316_filtered_20k.json` — filtered COCO json (images +
  matching annotations + original categories list) for the selected set.
- `selected_ids.json` — the list of selected image ids. Written for reference
  but not consumed by any later step.

### 2. `download_images.py` — download the selected images

Reads `caltech_bboxes_20200316_filtered_20k.json`, downloads each listed image
from the S3 bucket above (unsigned/public request) into
`caltech-camera-traps/images/`, multi-threaded (32 workers), skipping files
already on disk. Logs any failures to `download_failures.json`.

Result: 13,883 files downloaded, 0 failures, ~5.2 GB.

### 3. `collapse_categories.py` — drop unwanted categories, collapse the rest to one class

Starting from `caltech_bboxes_20200316_filtered_20k.json`:
- Drops `car`, `empty`, and `insect` annotations entirely.
- Drops any image left with zero remaining annotations after that (and deletes
  its file from `images/` on disk).
- Relabels every remaining annotation to a single category: `animal` (id `1`).

Result: 12,458 images, 13,017 annotations, 1 category. Output:
`caltech_bboxes_20200316_filtered_20k_animal.json`. Also deletes the 1,425
now-unreferenced image files from `images/`.

### 4. `fix_ids_for_yolo.py` — make the json actually loadable by Ultralytics

The source LILA json uses UUID-string `id`s for images (e.g.
`"5998cfa4-23d2-11e8-..."`). Ultralytics' `convert_coco()` (the standard
COCO→YOLO conversion used by YOLOv8 training) requires integer `image["id"]`
and `annotation["image_id"]`, and fails outright on strings
(`ValueError: Unknown format code 'd' for object of type 'str'`).

This script remaps image ids and annotation ids to sequential integers
(preserving the originals as `original_id` / `original_image_id` for
traceability back to the source filenames), and fills in `iscrowd`/`area`
defaults expected by the converter.

Output (final dataset json): **`caltech_bboxes_20200316_filtered_20k_animal_yolo.json`**

### 5. `prepare_data.py` — convert to a YOLO-ready dataset

Reads `caltech_bboxes_20200316_filtered_20k_animal_yolo.json` and runs
`ultralytics.data.converter.convert_coco()` on it, then does a train/val
split. Images themselves stay put in `images/` — the split is expressed as
file lists rather than by moving images around. Everything is written
directly under `~/data/bb9k/caltech-camera-traps/`, as siblings of
`images/`, using only relative paths so the whole directory stays portable
if it's ever moved or synced elsewhere:
- `labels/*.txt` — one YOLO-format label file per image, class `0` (single
  `animal` class), bbox coords normalized to `[0, 1]`. **Must be a sibling
  of `images/`**: Ultralytics finds each image's label by string-replacing
  `/images/` with `/labels/` in its path
  (`ultralytics.data.utils.img2label_paths`) — it's not looked up via
  `data.yaml` at all, so this can't live anywhere else.
- `train.txt` / `val.txt` — lists of image paths for each split (90/10,
  fixed seed 42; override with `--val-frac` / `--seed`), one per line as
  `./images/<file>.jpg`. The `./` prefix matters: Ultralytics' loader only
  rewrites relative entries with that exact prefix, replacing it with the
  txt file's own directory (`ultralytics.data.base.get_img_files`) — a bare
  `images/<file>.jpg` line would instead be resolved against the process's
  cwd at train time.
- `data.yaml` — Ultralytics dataset config pointing at the above (`train:
  train.txt`, `val: val.txt`, no `path:` key — it defaults to the yaml's own
  directory), ready to pass to `YOLO().train(data=...)`.

Result: 11,213 train / 1,245 val images, 12,458 label files. Verified by
actually loading `data.yaml` through `ultralytics.data.utils.check_det_dataset`
+ `ultralytics.data.dataset.YOLODataset` (run from an unrelated cwd, to rule
out cwd-dependence) — all images in both splits resolve a label (0
backgrounds, 0 corrupt).

Requires `ultralytics` + `opencv-python-headless` (not `opencv-python`, which
needs `libGL` that isn't available here) installed in the venv used to train.

This was also checked against a real training run: a small subset (40 train /
10 val) was converted the same way and a real 1-epoch YOLOv8n training +
validation run was executed end-to-end (losses computed, checkpoint saved,
val metrics produced) with no dataloader/label errors.

## Final artifacts

- `~/data/bb9k/caltech-camera-traps/images/` — 12,458 `.jpg` files (~4.8 GB).
- `~/data/bb9k/caltech-camera-traps/labels/` — matching YOLO label files.
- `~/data/bb9k/caltech-camera-traps/data.yaml` — use this for training, e.g.
  via `train/train.py` (see repo root `train/`).
