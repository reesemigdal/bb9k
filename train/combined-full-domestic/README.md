# Combined-full-domestic (Caltech + Oregon + Oxford-IIIT Pet) → YOLOv8 dataset

Extends `train/combined-full/`'s wildlife dataset with the Oxford-IIIT Pet
dataset (cats + dogs), at `~/data/bb9k/combined-animal-full-domestic/`.
Same merged-COCO + YOLO-ready-dataset shape as `combined-animal-full/`,
now from three sources instead of two.

## Command

```
python train/combined-full-domestic/prepare_data.py
```

Flags: `--val-frac 0.1`, `--seed 42` (defaults shown).

Requires `combined-animal-full/{caltech,oregon}_selected.json` (from
`train/combined-full/select.py` + `download.py`) and
`cambridge_iiit/cambridge_iiit_coco.json` (from
`train/cambridge_iiit/convert_to_coco.py`) to already exist.

## Sources

- **caltech, oregon** — reused as-is from `combined-animal-full/{caltech,
  oregon}_selected.json`: already single-`animal`-category, already
  border/degenerate-box filtered, already tagged `small_mammal`. Not
  touched here.
- **cambridge_iiit** — from `cambridge_iiit_coco.json`, using only each
  image's `<breed>_whole` annotation (matched by category *name* suffix,
  not the `annotation_type` field — see that dataset's README for why
  nothing should rely on that field being present). `_face` annotations
  are dropped entirely: mixing tight face boxes and looser whole-body
  boxes under one `animal` class would teach the detector inconsistent
  box semantics, and this combined dataset is whole-animal detection
  (matching Caltech/Oregon's own boxes) throughout.

  Every image with a `_whole` annotation is used — 7,367 of 7,390 (the
  other 23 have no whole-animal mask to begin with; see that dataset's
  README) — **all of them**, every one tagged `small_mammal = True`: cats
  and dogs are small mammals by any reasonable reading of that label, and
  unlike Caltech/Oregon's wildlife categories there's no "other" pool
  within this dataset to sample from, so there's no ratio to preserve —
  this dataset simply isn't downsampled at all.

  **No border-margin filtering** is applied to cambridge_iiit (unlike
  caltech/oregon's own selection, which drops any image with a box within
  5% of an edge): tightly-framed pet portraits legitimately touch the
  frame edge far more often than camera-trap wildlife shots, and the ask
  was explicitly to use all of them regardless.

## No target ratio this time

`train/combined-full/select.py` aimed for 2:1 (small-mammal:other) by
construction. Adding cambridge_iiit's 7,367 images (100% small-mammal, no
"other" counterpart) unavoidably pushes that ratio well past 2/3 — by
design, per the instruction that led to this dataset. So `prepare_data.py`
here doesn't attempt to hit any ratio: it's a plain random 90/10 split of
everything, same split mechanism as `combined-full` but without that
script's oversampling equivalent (irrelevant here, since nothing is being
downsampled or repeated).

## Result (last run)

- **caltech**: 5,946 images (2,479 small_mammal), 6,132 annotations.
- **oregon**: 26,389 images (19,078 small_mammal), 26,595 annotations.
- **cambridge_iiit**: 7,367 images (7,367 small_mammal — all of them),
  7,367 annotations.
- **Combined: 39,702 images**, 40,094 annotations, single `animal`
  category.
- **train.txt**: 35,732 images, 72.8% small-mammal.
- **val.txt**: 3,970 images, 73.1% small-mammal.

Verified via `ultralytics.data.utils.check_det_dataset` +
`ultralytics.data.dataset.YOLODataset`: both splits load with matching
image/label counts, 0 broken symlinks, all image/annotation ids unique
and referentially valid. Ultralytics' own scanner also auto-repaired one
pre-existing corrupt JPEG among the cambridge_iiit images (same benign
self-healing behavior seen with Caltech/Oregon previously — a source-data
defect, not introduced by this pipeline).

## Training

```
python train/train.py --dataset combined-full-domestic --epochs 100 --batch 64
```
