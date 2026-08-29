# Combined-full (Caltech + Oregon Critters, from full pools) → YOLOv8 dataset

A second combined dataset, at `~/data/bb9k/combined-animal-full/`, built the
same small-mammal-first way as `train/combined/` but from each source's
*entire* raw annotation pool (63,025 Caltech images / 99,909 Oregon images)
instead of each dataset's own pre-existing ~20k-image subsample. The
difference that matters: with a much bigger pool to draw from, every
small-mammal image can be used once, with real distinct "other" images
filling out the rest — no oversampling/duplicate lines needed the way
`train/combined/` needed for its train split.

## Why a separate pipeline from `train/combined/`

`train/combined/prepare_data.py` merges each source's already-built
`*_animal_yolo.json` (itself a ~20k-image random subsample, selected by
each dataset's own `select_and_download.py` long before "small mammal"
was a consideration) and then oversamples small-mammal *images* by
repeating lines in `train.txt` to hit a target ratio. That works, but caps
the small-mammal signal at whatever fraction of the original ~20k
subsample happened to be small-mammal, and repeats real images rather than
finding new ones.

This pipeline instead selects directly from the full pool *before*
subsampling, so "use all the small-mammal images" means all ~21.5k of them
across both datasets combined, not just the ~11k that happened to survive
into the earlier 20k-image draws.

## Commands (run in this order)

```
python train/combined-full/select.py       # JSON-only, no downloads — fast, safe to re-run/tune
python train/combined-full/download.py     # fetches only what select.py picked and isn't already local
python train/combined-full/prepare_data.py # merges the two into combined-animal-full/
```

Flags (all optional):

```
select.py:       --border-margin 0.05  --seed 42
prepare_data.py: --val-frac 0.1  --seed 42
```

## select.py — full-pool selection (JSON only, no images touched)

Reads each source's *raw* annotation json directly:
`caltech-camera-traps/caltech_bboxes_20200316.json` and
`oregon-critters/oregon_critters.json` (not the `*_filtered_20k*` files —
those are themselves just one particular subsample of these).

Per source, in order:

1. **Frame/burst de-dup**, replicating each dataset's own
   `select_and_download.py` logic (not encoded anywhere else, so it had to
   be redone here against the raw json): Caltech keeps only
   `frame_num == 1`; Oregon keeps images with no `(N)` burst marker in the
   filename, or where it's `(1)` — present in COA_2019/COA_2021/HJA_GRID/
   ORSNAP, absent everywhere else, so absence isn't itself a reason to
   drop an image.
2. **Drop junk-category annotations** (Caltech: `car`/`empty`/`insect`;
   Oregon: `empty`/`human` — same sets each dataset's own
   `collapse_categories.py` uses), then drop any image left with zero
   annotations. Unlike Oregon's own pipeline (which deliberately keeps
   human-only images as background negatives), this drops them — this is
   a curated small-mammal-weighted selection, not a reproduction of
   Oregon's full-dataset design.
3. **Tag `small_mammal`** against our established name sets (same as
   `train/combined/prepare_data.py`'s `SOURCES`):
   - Caltech: `rabbit`, `squirrel`, `rodent`
   - Oregon: `townsend's chipmunk`, `douglas squirrel`, `leporidae family`,
     `humboldt's flying squirrel`, `mountain beaver`, `neotoma species`,
     `california ground squirrel`, `western gray squirrel`, `small mammal`
4. **Drop degenerate annotations** (zero width/height) and any image left
   empty as a result.
5. **Drop border-cropped images**: any image with an annotation within
   `--border-margin` (default 5%) of an edge is dropped entirely — computed
   straight from `bbox`/`width`/`height` already in the json, which is why
   this can run *before* downloading a single image.

**Selection**: every surviving small-mammal image is kept; "other"
(non-small-mammal) images are randomly sampled down to half that count, so
the final pool is 2:1 small-mammal:other (66.7%) by construction.

Writes `caltech_selected.json` / `oregon_selected.json` into
`combined-animal-full/` — same shape as each source's own
`*_animal_yolo.json` (single `animal` category; `species_name` kept per
annotation for traceability/reporting even though `category_id` is already
collapsed to `1`), but ids are **not yet** remapped to sequential ints —
that happens in `prepare_data.py` (these manifests never went through
each source's own `fix_ids_for_yolo.py`, so ids are still each raw json's
original strings).

### Result (last run)

```
caltech: raw=63025 -> frame-filtered=26927 -> junk/degenerate-filtered=23799 -> border-filtered=15108 (8691 dropped for border)
         | small_mammal=2479 (16.4%), other=12629
oregon:  raw=99909 -> frame-filtered=75327 -> junk/degenerate-filtered=68831 -> border-filtered=45109 (23722 dropped for border)
         | small_mammal=19078 (42.3%), other=26031

selected: 21557 small_mammal + 10778 other = 32335 total (66.7% small_mammal)
  caltech: 5946 images (2479 small_mammal), 6132 annotations
  oregon:  26389 images (19078 small_mammal), 26595 annotations
```

Species breakdown of the selection (top small-mammal classes plus what
landed in the random "other" fill) is in the conversation history / can be
regenerated by rerunning `select.py`, which prints it each time.

## download.py — fetch what's missing

Same public/unsigned S3 bucket both datasets already use (see
`~/data/bb9k/readme.txt`): `us-west-2.opendata.source.coop`, under
`agentmorris/lila-wildlife/{caltech-unzipped/cct_images,oregon-critters}/`.
Downloads straight into each **source** dataset's own `images/` dir (not
`combined-animal-full/` — that only ever holds symlinks), reusing whatever
either dataset's own earlier ~20k-image pipeline already fetched. Skips
anything already on disk.

Last run: 2,018 new Caltech images + 16,378 new Oregon images downloaded,
**0 failures** (Caltech now 14,476 images on disk total, Oregon 36,128).

## prepare_data.py — final merge

Same recipe as `train/combined/prepare_data.py` (relative symlinks into
`combined-animal-full/images/`, prefixed `caltech__`/`oregon__`; id offset
per source; `convert_coco()` for `labels/`) — but **no oversampling step**:
since `select.py` already chose the pool to be 2:1 small-mammal:other from
distinct real images, a plain random 90/10 split naturally preserves that
ratio in both `train.txt` and `val.txt` without any duplicate lines.

### Result (last run)

- **32,335 images total**, 32,727 annotations, all real/distinct (no
  duplicate lines anywhere).
- **train.txt**: 29,102 images, 66.8% small-mammal.
- **val.txt**: 3,233 images, 65.9% small-mammal.

Verified via `ultralytics.data.utils.check_det_dataset` +
`ultralytics.data.dataset.YOLODataset`: both splits load with matching
image/label counts, 0 broken symlinks. Ultralytics' own scanner also found
and auto-repaired 440 pre-existing corrupt-JPEG files among the Caltech
images (missing/malformed end-of-image marker — a source-data defect, not
introduced by this pipeline; re-saved in place via PIL, same self-healing
behavior Ultralytics always applies during dataset verification). 0 from
Oregon this time.

## Training

```
python train/train.py --dataset combined-full --epochs 100 --batch 64
```

`train/train.py`'s default `--dataset` is still `combined` (the
oversampled-from-20k version) — pass `--dataset combined-full` explicitly
to use this one.
