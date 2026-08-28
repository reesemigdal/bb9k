# Oxford-IIIT Pet (cambridge_iiit) → COCO

Converts the Oxford-IIIT Pet dataset's native annotations
(`~/data/bb9k/cambridge_iiit/`) into one COCO json covering the *entire*
dataset, as faithfully as the source data allows — no filtering, category
collapsing, or train/val split yet. Unlike Caltech/Oregon, this source
isn't COCO to begin with (PASCAL VOC XML + a plain-text classification
list + per-pixel trimap PNGs), so this script plays the role their own
already-COCO `caltech_bboxes_20200316.json`/`oregon_critters.json` played
for those datasets: the unfiltered starting point everything else builds
from.

Source: [Parkhi et al., "Cats and Dogs", CVPR 2012](https://www.robots.ox.ac.uk/~vgg/data/pets/)
— 7,390 images, 37 breeds (25 cat + 12 dog), ~200 images/breed.

## Command

```
python train/cambridge_iiit/convert_to_coco.py
```

No arguments. Reads from `~/data/bb9k/cambridge_iiit/{images,annotations}/`,
writes `~/data/bb9k/cambridge_iiit/cambridge_iiit_coco.json`.

## Source layout

- `images/*.jpg` — 7,390 images, named `<breed>_<n>.jpg`; capitalized first
  letter = cat, lowercase = dog (verified consistent with the official
  breed/class-id list for all 37 breeds).
- `annotations/list.txt` — `<stem> <class-id 1-37> <species 1|2> <breed-id>`
  for 7,349 of the 7,390 images. The other 41 exist as real image files but
  aren't listed — still fully classifiable from their filename, so they're
  not dropped (a few even still have an xml head box).
- `annotations/trainval.txt`, `annotations/test.txt` — same 3-column
  format, partition list.txt's 7,349 images into the paper's split
  (3,680 / 3,669).
- `annotations/xmls/*.xml` — PASCAL VOC bounding boxes, **face only**
  (confirmed by the README and by extent — visibly smaller than the
  trimap silhouette), for 3,686 of the 7,390 images.
- `annotations/trimaps/*.png` — per-pixel trimap (1=foreground,
  2=background, 3=unclassified/boundary) for every one of the 7,390
  images — this is the **whole-animal** annotation.

## Design decisions

**Two categories per breed** — `"<breed>_face"` and `"<breed>_whole"`
(74 categories total), rather than one breed category plus a
per-annotation type flag — so an annotation's kind is identifiable from
`category_id` alone, no cross-referencing needed. Each category also
carries an `annotation_type` (`"face"`/`"whole"`) field as a convenience
for filtering, but it's redundant with the name suffix by construction —
nothing here or downstream should need to rely on it existing.

- `"<breed>_face"` — straight from the xml, when one exists. Other xml
  fields (`pose`/`truncated`/`occluded`/`difficult`/the xml's own literal
  `species_name_xml`) kept alongside on the annotation, unmodified.
- `"<breed>_whole"` — derived from the trimap, for every image that has
  one (7,367 of 7,390; the other 23 have an all-background trimap with
  zero foreground/boundary pixels — a pre-existing source-data quirk, not
  something this script drops on its own initiative beyond "there's
  nothing there to encode"). Foreground(1) *and* unclassified/boundary(3)
  pixels are both treated as animal — in a matting-style trimap like this,
  "unclassified" is the fuzzy true edge of the subject (fur, blur), not a
  separate region.

  **Connected-component cleanup**: the raw foreground mask can have
  disjoint stray components from trimap noise — found by inspecting
  `boxer_149.jpg`, whose naive whole-mask bbox came out `[0, 0, 378, 375]`
  (essentially the entire 500×375 image) because a sparse, scattered noise
  band across the top of the frame and a small stray blob happened to sit
  near two different edges. Each connected component is kept only if
  **both** `area >= 20px` and `fill_ratio = area/bbox_area >= 0.20`.
  Checked across all 1,494 non-largest components in the 1,304 images
  that have more than one: fill_ratio is strongly bimodal (25th
  percentile 2.9%, 75th percentile 70.8%, almost nothing between), so
  this threshold cleanly separates scattered noise from real disjoint
  blobs (e.g. a body split by occlusion, or a genuinely separate part —
  `boxer_149`'s own small corner blob, 2,068px at 73.5% fill, survives
  the filter as a legitimate part of the dog). Area alone was checked and
  rejected as the sole criterion: it can't tell that corner blob apart
  from noise, since both are small — only shape regularity can. After
  fixing, `boxer_149`'s bbox becomes `[0, 104, 283, 271]` — its left edge
  legitimately still touches 0 (that corner blob is real and does reach
  the frame edge), but the noise-driven top/right overextension is gone.

  A second, distinct noise pattern slipped past that filter: degenerate
  **line-shaped** components (e.g. a 1px-wide column running the full
  height of the frame, found in `saint_bernard_199.jpg`, whose bbox came
  out `[0, 0, 432, 290]` before this fix). A 1px-wide bbox has no room to
  be "sparse" in, so it trivially scores 100% fill_ratio no matter how
  spurious it is. Checked across every kept minority component in the
  dataset: this is *exclusively* a border artifact (676/676 thin
  components — `min(w,h) <= 3px` — were within 5% of an image edge; 0
  were interior), and aspect ratio (`min(w,h)/max(w,h)`) cleanly separates
  it from real border-touching parts (border-adjacent noise components:
  75th percentile aspect ratio 2.6%; interior components: 1st percentile
  34.8% — `boxer_149`'s corner blob sits at 48.7%). So a component within
  `BORDER_MARGIN` (5%) of an edge is additionally required to have
  `aspect_ratio >= 0.20`; components away from the border skip this check
  entirely, since a real thin anatomical part (a tail, a leg) shouldn't be
  penalized just for being thin. After fixing, `saint_bernard_199`'s bbox
  becomes `[6, 109, 426, 181]`. Re-verified against the full dataset: 0
  thin border components remain in the final output.

  Surviving components are unioned back into one mask and encoded as a
  COCO RLE `segmentation` via pycocotools, with `bbox`/`area` derived
  *from that mask* (pycocotools' own mask→bbox/area conversion), not
  estimated any other way, so they're guaranteed consistent with the
  stored segmentation.

**Breed** (not just cat/dog species) is the category's base name, since
it's the more specific of the two classifications this dataset provides,
and a downstream user who only wants species can always collapse via each
category's `supercategory` (`"Cat"`/`"Dog"`). Breed is derived from each
filename directly (strips a trailing `_<n>`), not solely from `list.txt`,
because that also correctly classifies the 41 images `list.txt` omits —
verified to agree with `list.txt`'s own class-id grouping for all 37
breeds first.

**All ids (image/annotation/category) are sequential ints**, matching
what `convert_coco()`-based tooling elsewhere in this repo expects —
each source's original (non-int) identifier is kept as `original_id` on
every image and annotation for traceability.

**Every image file gets a record**, regardless of whether it has an xml
box, a `list.txt` classification, or a trainval/test split — absence is
recorded via `has_face_bbox`/`has_whole`/`in_official_list`/`split`
(`None` when unknown) rather than by omitting the image, so a later
filtering step can decide what it needs instead of this one guessing.
`trimap_file` is also kept on every image record that has one, as a
pointer back to the raw pixel data in case a future use wants it directly
rather than through the derived RLE.

Image `mode` (PIL) is recorded per image since it's not uniformly RGB:
7,378 RGB, 6 palette (`P`), 3 grayscale (`L`), 3 `RGBA` — worth knowing
before assuming `.convert("RGB")` is a no-op anywhere downstream.

## Result (last run)

- **7,390 images** total (7,349 in the official `list.txt`, 41 not).
- **11,054 annotations**: 3,687 `_face` (3,686 images — one, `Bengal_105`,
  has 2 face boxes) + 7,367 `_whole`.
- **74 categories** (37 breeds × face/whole), 2,400 cat images / 4,990 dog
  images.
- **Split**: 3,680 trainval / 3,669 test / 41 unsplit (the 41 not in
  `list.txt`).

Validated: all image/annotation/category ids are ints and unique, every
annotation's `image_id`/`category_id` resolves, no annotation carries an
`annotation_type` field, category names correctly come out as e.g.
`Abyssinian_face`/`Abyssinian_whole`, and 50 sampled `_whole` segmentations
round-trip through `pycocotools.mask.decode()` with their pixel count
exactly matching the stored `area`.

## Not done here (by design)

No image download/selection step is needed — all 7,390 images and their
annotations are already local. No category filtering/collapsing,
small-mammal-style tagging, train/val split, or YOLO conversion — this is
the base, complete COCO json; a later script can build a task-specific
dataset from it the same way `train/combined/prepare_data.py` and
`train/combined-full/select.py` do for Caltech/Oregon, once there's an
actual downstream use to design around (face detection? whole-animal
detection? breed classification? some combination merged into the
existing `combined-*` animal detectors?).
