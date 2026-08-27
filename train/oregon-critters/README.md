# Oregon Critters → YOLOv8 dataset

Builds a 20k-image subset of the Oregon Critters camera-trap dataset, biased
toward rabbits/rodents/squirrels/small woodland mammals plus (intended, see
step 2) all available human sightings, and converts it into a COCO json
that Ultralytics' YOLOv8 tooling can actually train on. Same overall shape
as `train/caltech/` (see that README for the general pattern), adapted for
a few real differences between the two source datasets — called out below.

Source data:
- Images: `s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/oregon-critters/`
  (a subset is downloaded, see below)
- Annotations: `oregon_critters.json` (COCO-format, 99,909 images / 102,014
  annotations / 46 categories), downloaded ahead of time into
  `~/data/bb9k/oregon-critters/`.

All scripts here have `BASE` hardcoded to `~/data/bb9k/oregon-critters` —
that's where the raw json lives and where all outputs (json files,
`images/`, `labels/`, `train.txt`/`val.txt`/`data.yaml`) are written, all as
siblings. Update the path in each script if the data directory moves.

Unlike Caltech, this dataset has no explicit `frame_num` field, and
`sequence_level_annotation` is `False` for all 99,909 images — but that just
means annotations are per-image, not that there's no burst grouping to worry
about. Some sources instead encode a frame index in the filename itself as
`(N)` (e.g. `...(7)_ho.JPG` is frame 7 of a burst) — see step 1.

## Steps (run in this order)

### 1. `select_and_download.py` — pick which images to use

Reads `oregon_critters.json` and selects images under these rules:
- Drops any image whose only annotation is `empty` (i.e. no animal/human
  present) — dropped at selection time here, rather than after download like
  Caltech did with `car`/`empty`/`insect`.
- Frame filtering: some sources encode a burst/frame index as `(N)` in the
  filename (e.g. `...(7)_ho.JPG` = frame 7). Checked this against the whole
  dataset first — it's **not universal**: present in 100% of COA_2019/
  COA_2021/HJA_GRID/ORSNAP, absent from 100%(ish) of COA_2020/DUNES/ESF_*/
  HJA_MARIE. Where present, keep only `(1)` (mirrors Caltech's
  `frame_num==1` filter, avoids near-duplicate burst frames); where absent,
  keep the image as-is — no burst grouping is knowable from the filename for
  those sources, and dropping them would throw out ~62% of the dataset
  including its single largest source (COA_2020, ~53k images).
- Bucket 1 — broad "small woodland mammal" categories, up to a cap of
  10,000: `leporidae family` (rabbits), `california ground squirrel`,
  `western gray squirrel`, `douglas squirrel`, `humboldt's flying squirrel`,
  `townsend's chipmunk`, `neotoma species`, `mountain beaver`, `small
  mammal`, `marten`, `mink`, `weasel family`, `western spotted skunk`,
  `striped skunk`. (Deliberately broader than Caltech's rabbit/squirrel/
  rodent-only bucket — this dataset's "small woodland mammal" categories
  include mustelids and skunks.)
- Bucket 2 — `human` images, up to a cap of 5,000. Only ~314 exist in the
  whole dataset, so this takes all of them.
- Bucket 3 — random sample of everything else (any other non-empty
  category), filling up to a total cap of 20,000.
- Random sampling uses a fixed seed (42) for reproducibility. Buckets are
  mutually exclusive (each image counted in at most one).

`file_name` in the source json is a full nested S3 key (e.g.
`COA_2019/COA2019_Bat/images/....JPG`) — notably with its own `images/`
path segment partway through. Keeping that nested structure under our own
`images/` directory would produce a path containing `/images/` twice, which
breaks Ultralytics' label-lookup convention (it splits on the *last*
`/images/` occurrence). All basenames are unique dataset-wide (checked:
99,909 images, 99,909 unique basenames), so instead this script flattens
`file_name` down to just the basename for local storage, keeping the
original nested path as a new `source_path` field for `download_images.py`
to use when building the S3 key.

Outputs:
- `oregon_critters_filtered_20k.json` — filtered COCO json (images +
  matching annotations + original categories list) for the selected set.
- `selected_ids.json` — the list of selected image ids. Written for
  reference but not consumed by any later step.

### 2. `download_images.py` — download the selected images

Reads `oregon_critters_filtered_20k.json`, downloads each listed image from
the S3 bucket above (unsigned/public request, key = prefix + `source_path`)
into `oregon-critters/images/` (flat, named by `file_name`), multi-threaded
(32 workers), skipping files already on disk. Logs any failures to
`download_failures.json`.

Result: 19,750/20,000 downloaded, 250 failed (all genuine 404s, not
transient). 248 of those 250 are **every single selected `human` image** —
turns out the `ESF_TRAIL_Human` folder doesn't exist in the public S3
bucket at all, even though it's in the annotation json (the other ESF_TRAIL
species folders are all there). Presumably withheld from the public release
for privacy, since unlike the wildlife shots these are identifiable photos
of people. Net effect: the human bucket contributes 0 images to the final
dataset — accepted as-is rather than backfilled with something else. The
other 2 failures are unrelated stray 404s among regular animal images.

### 3. `collapse_categories.py` — drop unwanted categories, collapse the rest to one class

Starting from `oregon_critters_filtered_20k.json`:
- Drops `empty` annotations (defensive — selection already excluded
  empty-only images) and `human` annotations.
- Relabels every remaining annotation to a single category: `animal` (id
  `1`).
- **Does not delete images that end up with zero annotations**, unlike
  Caltech's collapse step. An image whose only annotation was `human` ends
  up with none after the drop above, and that's intentional: those stay in
  the dataset as background/negative examples for a single-class `animal`
  detector, rather than being removed. (An image with both a human and a
  real animal keeps the animal box, human box just dropped.)

Output: `oregon_critters_filtered_20k_animal.json`.

### 4. `fix_ids_for_yolo.py` — make the json actually loadable by Ultralytics

Same problem and fix as Caltech's version: image `id`s here are the
original nested-path strings (not UUIDs, but still non-integer), and
Ultralytics' `convert_coco()` requires integer `image["id"]` /
`annotation["image_id"]`. Remaps both to sequential integers (preserving
originals as `original_id`/`original_image_id`), and fills in
`iscrowd`/`area` defaults expected by the converter.

Output (final dataset json): **`oregon_critters_filtered_20k_animal_yolo.json`**

### 5. `prepare_data.py` — convert to a YOLO-ready dataset

Same approach as `train/caltech/prepare_data.py` — runs
`ultralytics.data.converter.convert_coco()`, then does a train/val split,
writing `labels/`, `train.txt`/`val.txt`, and `data.yaml` as relative-path
siblings of `images/` under `~/data/bb9k/oregon-critters/` (see that
script's docstring for why the paths have to be relative and structured
that way).

Two additions versus Caltech:
- `convert_coco()` only ever writes a label file for images that have at
  least one annotation, so any zero-annotation image (background) gets no
  label file at all. Ultralytics still trains on a missing-label image
  correctly (it's treated identically to an empty-label one — zero boxes),
  but reports it as "missing" rather than "background" in its dataset-scan
  summary, which reads like a bug. This script explicitly writes an empty
  `*.txt` for every such image after conversion so the scan reports them
  cleanly as backgrounds. In practice, since every `human` image failed to
  download (see step 2), the ~19,750 surviving images have no
  intentional backgrounds left — but 74 real annotations in the raw source
  json have a zero-size bbox (width=height=0, an artifact already present
  in the upstream data, not introduced by this pipeline), which
  `convert_coco()` silently drops. The images those belonged to end up as
  incidental backgrounds too (70 of them, once you exclude any that also
  had a second, valid annotation) — this script's empty-`*.txt` logic
  handles both cases identically, it doesn't need to know why a label
  ended up empty.
- Also fixes a bug this surfaced (and present, latent, in Caltech's copy
  too — fixed there as well): the script filters `images` down to ones that
  actually downloaded, but was leaving `annotations` unfiltered, so any
  annotation belonging to a missing image (like the 248 failed human ones)
  still referenced a now-excluded `image_id`, which crashed `convert_coco()`
  with a `KeyError`. Fixed by filtering `annotations` to the same surviving
  `image_id`s before handing off to `convert_coco()`.

Requires `ultralytics` + `opencv-python-headless` (not `opencv-python`,
which needs `libGL` that isn't available here) installed in the venv used
to train.

## Final artifacts

- `~/data/bb9k/oregon-critters/images/` — 19,750 `.jpg` files (20,000
  selected, 250 failed to download — see step 2).
- `~/data/bb9k/oregon-critters/labels/` — matching YOLO label files, all
  single-class `animal`, 70 incidentally empty (background) due to
  zero-size boxes in the source annotations.
- `~/data/bb9k/oregon-critters/data.yaml` — use this for training: 17,775
  train / 1,975 val (90/10, seed 42). Verified by loading it through
  `ultralytics.data.utils.check_det_dataset` + `ultralytics.data.dataset.
  YOLODataset` (run from an unrelated cwd) — 0 corrupt, 0 missing.
