# Combined (Caltech + Oregon Critters) → YOLOv8 dataset

Merges the two per-source animal-detection datasets — Caltech Camera Traps
and Oregon Critters, each already prepared independently (see
`train/caltech/README.md`) — into one YOLO-ready dataset at
`~/data/bb9k/combined-animal/`, with small-mammal images oversampled in the
training split and low-quality annotations filtered out.

## Prerequisites

Both source datasets must already have their own pipeline outputs in place
under `~/data/bb9k/`:

- `caltech-camera-traps/images/`, `caltech_bboxes_20200316_filtered_20k.json`
  (species-level), `caltech_bboxes_20200316_filtered_20k_animal_yolo.json`
  (collapsed to one `animal` class, int ids — see `train/caltech/README.md`).
- `oregon-critters/images/`, `oregon_critters_filtered_20k.json`
  (species-level), `oregon_critters_filtered_20k_animal_yolo.json`
  (collapsed to one `animal` class, int ids — built the same way as
  Caltech's, just without a committed script for that dataset).

## Command

```
python train/combined/prepare_data.py
```

Flags (all optional, shown with their defaults):

```
--val-frac 0.1          # fraction of unique images held out for val
--sm-frac 0.6667         # target fraction of TRAIN images that are small-mammal
--border-margin 0.05     # drop images with any bbox side within this fraction of the edge
--seed 42
```

Re-running is idempotent/deterministic (same seed → same split and
oversampling) and safe to repeat after either source dataset changes —
it fully regenerates `combined-animal/images/`, `labels/`,
`combined_animal_yolo.json`, `train.txt`, `val.txt`, and `data.yaml` each time.

## What it does, in order

1. **Load** each source's `*_animal_yolo.json`, keeping only images that
   still exist on disk (Oregon currently has 250 listed images missing).
2. **Tag `small_mammal`** on every image by cross-referencing the
   *species-level* json (the animal-collapsed json lost species identity) via
   the `original_id` field `fix_ids_for_yolo.py` preserved. Small-mammal
   class sets (rodents + lagomorphs, plus Oregon's generic "small mammal"
   catch-all — see `train/combined/prepare_data.py`'s `SOURCES` for the
   exact lists and rationale):
   - Caltech: `rabbit`, `squirrel`, `rodent`
   - Oregon: `townsend's chipmunk`, `douglas squirrel`, `leporidae family`,
     `humboldt's flying squirrel`, `mountain beaver`, `neotoma species`,
     `california ground squirrel`, `western gray squirrel`, `small mammal`
3. **Drop degenerate annotations** (zero width/height boxes — 74 exist in
   Oregon's source data, an animal flagged but never actually boxed), then
   drop any image left with zero valid annotations as a result (70 images).
4. **Drop border-cropped images**: any image with an annotation whose box
   touches within `--border-margin` (default 5%) of any edge is dropped
   entirely (image + all its boxes, not just the offending one) — a
   camera-cropped animal's visible box understates its true extent, so
   nothing in that frame is trustworthy ground truth. This is the single
   biggest cut: ~10.7k of ~32.1k images.
5. **Symlink** surviving images into `combined-animal/images/`, prefixed
   `caltech__`/`oregon__` (collision-proof, traceable), targeted by
   *relative* path so the directory stays valid across environments that
   mount `~/data` under a different absolute prefix (a container's
   `/home/jmigdal` vs. a host's `/jmigdal-data`, etc. — this was a real
   bug, fixed after hitting it).
6. **Write `combined_animal_yolo.json`** — merged COCO json, ids offset
   per-source to stay unique, each image carrying `small_mammal`,
   `original_id`/`original_file_name` (traceability back to the source).
7. **Run `ultralytics.data.converter.convert_coco()`** on that json to
   produce `labels/*.txt`.
8. **Split + oversample**: shuffle unique images, hold out `--val-frac` for
   `val.txt` at the *natural* class mix (no oversampling — validation
   should measure performance on the true distribution). For `train.txt`,
   non-small-mammal images appear once each; small-mammal images are
   repeated (duplicate lines, same symlinked file + label, no pixels
   duplicated on disk) to hit `--sm-frac` of the training set, spread as
   evenly as possible (a base repeat count + a random subset getting one
   extra copy).
9. **Write `data.yaml`** (`train: train.txt`, `val: val.txt`,
   `names: {0: animal}`).

## Known data-quality fix outside this pipeline

One Oregon source image, `oregon-critters/images/Drive_05_06050044_ik.JPG`,
had a libjpeg-detectable defect (`Corrupt JPEG data: 39 extraneous bytes
before marker 0xd9` — harmless, it decoded fine, but spammed the training
console every time it was read). Fixed by re-saving it in place via
`PIL.Image.open(...).save(..., "JPEG", quality=100, subsample=0)`. This
lives on the *source* file, not in this script — if Oregon's images are
ever re-downloaded from scratch, that one file's warning would need to be
fixed again the same way (it's harmless either way; see commit history /
conversation for the one-off fix).

## Result (last run)

- **Caltech**: 12,458 images, 3,883 (31.2%) tagged small-mammal.
- **Oregon**: 19,750 images (250 missing on disk skipped), 11,847 (60.0%)
  tagged small-mammal; 74 degenerate annotations dropped, 70 images dropped
  as a result.
- **Border-crop filter**: 10,679 images dropped (5% margin).
- **Combined**: 21,459 images (7,893 caltech / 13,566 oregon), 21,843
  annotations, 11,091 (51.7%) small-mammal overall.
- **val.txt**: 2,145 images, natural 51.5% small-mammal mix.
- **train.txt**: 27,984 lines / 19,314 distinct images — 9,986 unique
  small-mammal images repeated to 18,656 lines (66.7%), 9,328
  non-small-mammal images appearing once each.

Verified via `ultralytics.data.utils.check_det_dataset` +
`ultralytics.data.dataset.YOLODataset`: both splits load with 0
corrupt/missing, all symlinks resolve, zero near-border annotations and
zero zero-area annotations remain.

## Training

```
python train/train.py --dataset combined --epochs 100 --batch 64
```

(`train/train.py --dataset {caltech,oregon,combined}` picks the matching
`data.yaml`; `combined` is the default.)
