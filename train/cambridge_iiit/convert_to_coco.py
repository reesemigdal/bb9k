#!/usr/bin/env python3
"""Convert the Oxford-IIIT Pet dataset's native annotations into one COCO
json covering the *entire* dataset (~/data/bb9k/cambridge_iiit/), as
faithfully as the source data allows. This is the analog, for this
dataset, of the COCO json Caltech/Oregon already came with from LILA --
those didn't need a conversion step; this one does, since the source
format here is PASCAL VOC XML (bboxes) + a plain-text classification list
+ per-pixel trimap PNGs, not COCO to begin with.

Deliberately does NOT filter, select a subset, or pick a train/val split --
this is the base "as complete as possible" conversion; those are
downstream-application decisions for a later script, same as
caltech_bboxes_20200316.json / oregon_critters.json are the unfiltered
starting points their own select_and_download.py scripts work from.

Source layout (see annotations/README):
- images/*.jpg               7,390 images, named "<breed>_<n>.jpg";
                              capitalized first letter = cat, lowercase = dog.
- annotations/list.txt        "<stem> <class-id 1-37> <species 1|2> <breed-id>"
                              for 7,349 of the 7,390 images (see below for the
                              other 41).
- annotations/trainval.txt,
  annotations/test.txt        same 3-column format, partition list.txt's
                              7,349 images into the paper's train/test split.
- annotations/xmls/*.xml      PASCAL VOC bounding-box annotations, for
                              3,686 of the 7,390 images only (not all).
                              Despite the generic PASCAL VOC field names,
                              the README is explicit these are *face* boxes
                              ("Head bounding box annotations"), not
                              whole-animal boxes -- confirmed by their
                              extent being visibly smaller than the trimap
                              silhouette below.
- annotations/trimaps/*.png   per-pixel trimap (1=foreground, 2=background,
                              3=unclassified/boundary) for every one of the
                              7,390 images -- this is the whole-*animal*
                              annotation (vs. the xmls' face-only boxes).

Every breed gets **two** categories -- `"<breed>_face"` and
`"<breed>_whole"` (74 categories total) -- rather than one breed category
plus a per-annotation type flag, so an annotation's kind is identifiable
from `category_id` alone without cross-referencing anything else. Each
category also carries an `annotation_type` ("face"/"whole") field as a
convenience for filtering, but it's redundant with the name suffix by
construction -- nothing here or downstream should need to rely on it
existing.

- `"<breed>_face"`: straight from an xml file, when one exists (3,686
  images). Other xml fields (pose/truncated/occluded/difficult/the xml's
  own literal `species_name_xml`) are kept alongside, unmodified.
- `"<breed>_whole"`: derived from that image's trimap, for every image
  that has one (7,367 of 7,390; see below). Foreground(1) and
  unclassified/boundary(3) pixels are both treated as animal -- in a
  matting-style trimap like this, the "unclassified" band is the fuzzy
  true edge of the subject (fur, blur), not a separate region.

  The raw foreground mask can have multiple disjoint connected components
  -- confirmed on real examples (e.g. boxer_149.jpg's mask had 4: the real
  dog split across two touching-but-not-connected regions, plus a sparse
  scattered noise band across the top of the frame and a small solid stray
  blob in a corner) -- and naively taking the bbox of the *whole* mask
  produces a wildly oversized box the moment a stray component sits near
  an image edge. Each connected component is kept only if BOTH:
  - `area >= AREA_FLOOR` (20px) -- drops genuine speckle noise (a handful
    of stray pixels), too few to judge shape from.
  - `fill_ratio = area / bbox_area >= FILL_RATIO_MIN` (0.20) -- drops
    components that are sparse/scattered relative to their own bounding
    box, which is what a real (if partial) part of the animal generally
    isn't. Checked across all 1,494 non-largest components in the
    1,304 images with >1 component: fill_ratio is strongly bimodal (25th
    percentile 2.9%, 75th percentile 70.8%, almost nothing between), so
    this cleanly separates noise from real disjoint blobs (e.g. a body
    split by occlusion) without an area cutoff doing that job -- area
    alone is a poor discriminator, since some legitimate small components
    (like boxer_149's corner blob, 2,068px at 73.5% fill) are just as
    solid as the main blob, only smaller.

  Surviving components (usually all of them -- only 1,304/7,390 images
  have more than one to begin with) are unioned back into a single mask
  and encoded as a COCO RLE `segmentation` via pycocotools, with
  `bbox`/`area` derived *from that mask* (pycocotools' own mask->bbox/area
  conversion), not estimated any other way. An image whose trimap has
  zero surviving pixels (23 have zero foreground+boundary to begin with;
  none were found where filtering removes every component) gets no
  `_whole` annotation.

Breed (not just cat/dog species) is used as the category's base name,
since it's the more specific of the two classifications this dataset
provides and a downstream user needing only cat/vs/dog can always
collapse via each category's `supercategory`. It's derived from each
filename directly (strips a trailing "_<n>"), not solely from list.txt --
verified to agree with list.txt's own class-id grouping for all 37 breeds
-- because that also correctly classifies the 41 images list.txt omits
(some of which still have an xml face box; dropping them would throw
that away).

Every image file gets an entry regardless of whether it has an xml box,
a list.txt classification, or a trainval/test split -- absence is recorded
via `has_face_bbox`/`has_whole`/`in_official_list`/`split` rather than by
omitting the image, so nothing downstream is silently assumed.

All ids (image, annotation, category) are sequential ints, as COCO/
convert_coco()-based tooling elsewhere in this repo expects -- each
source's original (non-int) identifier is preserved as `original_id` for
traceability.
"""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

DATA_DIR = Path.home() / "data/bb9k/cambridge_iiit"
IMAGES_DIR = DATA_DIR / "images"
ANN_DIR = DATA_DIR / "annotations"
OUT_JSON = DATA_DIR / "cambridge_iiit_coco.json"

STEM_RE = re.compile(r"^(.+)_(\d+)$")
SPECIES_NAME = {"1": "Cat", "2": "Dog"}

# whole_from_trimap's connected-component cleanup thresholds -- see module docstring
# for how these were picked (component-level fill_ratio/aspect_ratio distributions
# across all multi-component trimaps).
AREA_FLOOR = 20
FILL_RATIO_MIN = 0.20
BORDER_MARGIN = 0.05
ASPECT_RATIO_MIN = 0.20


def load_list_file(path):
    """stem -> (class_id, species_id, species_breed_id), all as-read strings."""
    out = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        stem, class_id, species_id, breed_id = line.split()
        out[stem] = (class_id, species_id, breed_id)
    return out


def whole_from_trimap(trimap_path):
    """(segmentation RLE, bbox, area) derived from a trimap's foreground+boundary
    pixels, or None if nothing survives. The raw mask can have disjoint stray
    components (trimap noise); each connected component is kept only if:
    - it's large enough (area >= AREA_FLOOR) and solid enough (fill_ratio =
      area/bbox_area >= FILL_RATIO_MIN) -- catches scattered/sparse noise
      (e.g. boxer_149.jpg's noise band across the top of the frame); AND
    - if it's within BORDER_MARGIN of an image edge, it's also not
      pathologically thin (aspect_ratio = min(w,h)/max(w,h) >= ASPECT_RATIO_MIN).
      A degenerate line-shaped component (e.g. a 1px-wide column running the
      full height of the frame, as in saint_bernard_199.jpg) trivially scores
      100% fill_ratio -- a 1px-wide bbox has no room to be "sparse" in --
      so fill_ratio alone can't catch it; checked against all kept minority
      components in this dataset and this is *exclusively* a border artifact
      (676/676 thin components were border-adjacent, 0 interior), hence
      restricting the extra check to border-adjacent components rather than
      applying it everywhere (a real anatomical part like boxer_149's own
      corner blob, aspect_ratio ~0.49, would otherwise also touch the border).
    See module docstring for how the thresholds were picked. Surviving
    components are unioned back into one mask; bbox/area come from
    pycocotools' own mask conversion on that union, not estimated separately,
    so they're guaranteed consistent with the segmentation actually stored."""
    trimap = np.array(Image.open(trimap_path))
    img_h, img_w = trimap.shape
    fg = np.isin(trimap, [1, 3]).astype(np.uint8)
    if not fg.any():
        return None

    margin_x, margin_y = BORDER_MARGIN * img_w, BORDER_MARGIN * img_h
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    clean = np.zeros_like(fg)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        fill_ratio = area / (w * h)
        if area < AREA_FLOOR or fill_ratio < FILL_RATIO_MIN:
            continue
        near_border = x <= margin_x or y <= margin_y or (img_w - (x + w)) <= margin_x or (img_h - (y + h)) <= margin_y
        if near_border and (min(w, h) / max(w, h)) < ASPECT_RATIO_MIN:
            continue
        clean[labels == i] = 1
    if not clean.any():
        return None

    rle = mask_utils.encode(np.asfortranarray(clean))
    rle["counts"] = rle["counts"].decode("ascii")
    bbox = mask_utils.toBbox(rle).tolist()
    area = float(mask_utils.area(rle))
    return rle, bbox, area


def main():
    list_entries = load_list_file(ANN_DIR / "list.txt")
    splits = {}
    for split_name, fname in [("trainval", "trainval.txt"), ("test", "test.txt")]:
        for stem in load_list_file(ANN_DIR / fname):
            splits[stem] = split_name

    # breed_name -> official class_id, built from list.txt; verified 1:1 (see conversation /
    # repo history for the check) with the name every filename itself implies.
    name_to_class_id = {}
    for stem, (class_id, species_id, _) in list_entries.items():
        name = STEM_RE.match(stem).group(1)
        name_to_class_id.setdefault(name, class_id)

    # two categories per breed (face, whole), not one breed category plus a per-annotation
    # type flag -- see module docstring.
    breeds = {}  # class_id -> (breed_name, species_id)
    for stem, (class_id, species_id, _) in list_entries.items():
        breeds.setdefault(class_id, (STEM_RE.match(stem).group(1), species_id))
    assert len(breeds) == 37, f"expected 37 breeds, got {len(breeds)}"

    categories = []
    face_cat_id = {}
    whole_cat_id = {}
    for class_id in sorted(breeds, key=int):
        breed_name, species_id = breeds[class_id]
        for suffix, type_map in [("face", face_cat_id), ("whole", whole_cat_id)]:
            cat_id = len(categories) + 1
            categories.append({
                "id": cat_id,
                "name": f"{breed_name}_{suffix}",
                "supercategory": SPECIES_NAME[species_id],
                "breed_name": breed_name,
                "breed_id": int(class_id),
                "annotation_type": suffix,
            })
            type_map[class_id] = cat_id
    assert len(categories) == 74

    images = []
    annotations = []
    unmatched_names = set()
    xml_species_mismatches = []
    next_ann_id = 1

    jpg_paths = sorted(IMAGES_DIR.glob("*.jpg"))
    for image_id, jpg_path in enumerate(jpg_paths, start=1):
        stem = jpg_path.stem
        m = STEM_RE.match(stem)
        if not m:
            print(f"warning: filename doesn't match '<breed>_<n>' pattern, skipping: {jpg_path.name}")
            continue
        breed_name = m.group(1)
        class_id = name_to_class_id.get(breed_name)
        if class_id is None:
            unmatched_names.add(breed_name)
            continue

        with Image.open(jpg_path) as im:
            width, height = im.size
            mode = im.mode

        in_official_list = stem in list_entries
        if in_official_list:
            list_class_id, species_id, species_breed_id = list_entries[stem]
            assert list_class_id == class_id, f"{stem}: list.txt class_id {list_class_id} != filename-derived {class_id}"
            species_breed_id = int(species_breed_id)
        else:
            # one of the 41 images list.txt omits -- still classifiable from its filename.
            species_id = "1" if breed_name[0].isupper() else "2"
            species_breed_id = None

        trimap_path = ANN_DIR / "trimaps" / f"{stem}.png"

        image_record = {
            "id": image_id,
            "original_id": stem,
            "file_name": jpg_path.name,
            "width": width,
            "height": height,
            "mode": mode,
            "breed_id": int(class_id),
            "breed_name": breed_name,
            "species": SPECIES_NAME[species_id],
            "species_breed_id": species_breed_id,
            "split": splits.get(stem),
            "in_official_list": in_official_list,
            "trimap_file": f"annotations/trimaps/{stem}.png" if trimap_path.exists() else None,
            "has_face_bbox": False,
            "has_whole": False,
        }
        images.append(image_record)

        if trimap_path.exists():
            result = whole_from_trimap(trimap_path)
            if result is None:
                print(f"warning: {stem}: trimap has zero foreground/boundary pixels, no _whole annotation")
            else:
                rle, bbox, area = result
                image_record["has_whole"] = True
                annotations.append({
                    "id": next_ann_id,
                    "original_id": f"{stem}_whole",
                    "image_id": image_id,
                    "category_id": whole_cat_id[class_id],
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": rle,
                })
                next_ann_id += 1

        xml_path = ANN_DIR / "xmls" / f"{stem}.xml"
        if not xml_path.exists():
            continue
        image_record["has_face_bbox"] = True

        root = ET.parse(xml_path).getroot()
        xml_size = root.find("size")
        xml_width, xml_height = int(xml_size.find("width").text), int(xml_size.find("height").text)
        if (xml_width, xml_height) != (width, height):
            print(f"warning: {stem}: xml size {xml_width}x{xml_height} != actual jpg size {width}x{height}")

        for i, obj in enumerate(root.findall("object")):
            species_name = obj.find("name").text
            if species_name.capitalize() != SPECIES_NAME[species_id]:
                xml_species_mismatches.append((stem, species_name, SPECIES_NAME[species_id]))
            box = obj.find("bndbox")
            xmin, ymin = float(box.find("xmin").text), float(box.find("ymin").text)
            xmax, ymax = float(box.find("xmax").text), float(box.find("ymax").text)
            w, h = xmax - xmin, ymax - ymin
            annotations.append({
                "id": next_ann_id,
                "original_id": f"{stem}_face_{i}",
                "image_id": image_id,
                "category_id": face_cat_id[class_id],
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "iscrowd": 0,
                "species_name_xml": species_name,
                "pose": obj.find("pose").text if obj.find("pose") is not None else None,
                "truncated": int(obj.find("truncated").text) if obj.find("truncated") is not None else None,
                "occluded": int(obj.find("occluded").text) if obj.find("occluded") is not None else None,
                "difficult": int(obj.find("difficult").text) if obj.find("difficult") is not None else None,
            })
            next_ann_id += 1

    if unmatched_names:
        raise SystemExit(f"filenames with a breed name not among the 37 known categories: {unmatched_names}")
    if xml_species_mismatches:
        print(f"warning: {len(xml_species_mismatches)} xml <object><name> values disagree with the image's species: {xml_species_mismatches[:10]}")

    out = {
        "info": {
            "description": "Oxford-IIIT Pet Dataset (Parkhi et al., CVPR 2012), converted to COCO",
            "contributors": "Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, C. V. Jawahar",
            "source": "annotations/README in this dataset's own directory",
        },
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f)

    n_face = sum(im["has_face_bbox"] for im in images)
    n_whole = sum(im["has_whole"] for im in images)
    n_official = sum(im["in_official_list"] for im in images)
    face_cat_ids = set(face_cat_id.values())
    n_face_anns = sum(1 for a in annotations if a["category_id"] in face_cat_ids)
    print(f"wrote {OUT_JSON}")
    print(f"  images: {len(images)} total, {n_official} in official list.txt, {n_face} with a face bbox, {n_whole} with a whole mask")
    print(f"  annotations: {len(annotations)} ({n_face_anns} face, {len(annotations) - n_face_anns} whole)")
    print(f"  categories: {len(categories)} (37 breeds x face/whole)")
    n_cat = sum(1 for im in images if im["species"] == "Cat")
    print(f"  species: {n_cat} cat images, {len(images) - n_cat} dog images")
    n_trainval = sum(1 for im in images if im["split"] == "trainval")
    n_test = sum(1 for im in images if im["split"] == "test")
    print(f"  split: {n_trainval} trainval, {n_test} test, {len(images) - n_trainval - n_test} unsplit")


if __name__ == "__main__":
    main()
