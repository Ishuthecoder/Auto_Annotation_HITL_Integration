"""
Auto-Annotation Pipeline — Refactored Step Functions

Each function is fully parameterised (no hardcoded paths) and returns
the path to its primary output so the next step can consume it.

Pipeline
--------
  step1_run_yolo_segmentation()    → <output_dir>/coco_all.json
  step2_add_categories()           → <output_dir>/coco_added_cats.json
  step3_generate_crops()           → <output_dir>/crops/
  step4_run_color_classification() → <output_dir>/coco_updated.json  ← HITL input
"""
from __future__ import annotations

import copy
import json
import logging
import os
from typing import Dict, List, Optional, Set

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

log = logging.getLogger("auto_annotation")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    """Shoelace formula."""
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _collect_images(image_dir: str, subset: Optional[List[str]] = None) -> List[str]:
    """Return sorted image filenames from image_dir, optionally filtered to subset."""
    exts = (".jpg", ".jpeg", ".png")
    all_imgs = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(exts))
    if subset is not None:
        keep = set(subset)
        return [f for f in all_imgs if f in keep]
    return all_imgs


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — YOLOv8 Segmentation → COCO JSON
# ─────────────────────────────────────────────────────────────────────────────

def step1_run_yolo_segmentation(
    image_dir: str,
    model_path: str,
    output_dir: str,
    conf: float = 0.20,
    min_area: float = 200.0,
    valid_class_ids: Optional[Set[int]] = None,  # ← intentionally ignored: ALL objects go to HITL
    image_files: Optional[List[str]] = None,
) -> str:
    """
    Run YOLOv8 segmentation on images and produce a COCO JSON.

    *** IMPORTANT — ALL detected objects are written to coco_all.json regardless
    of class ID so that every object appears in the HITL review UI with its
    correct label.  Class-based filtering happens in Step 3 (crop generation
    only), NOT here. ***

    Args:
        image_dir:       Directory containing source images.
        model_path:      Path to YOLOv8 segmentation .pt weights.
        output_dir:      Directory where coco_all.json is written.
        conf:            YOLO confidence threshold.
        min_area:        Minimum polygon area (px²) — smaller polygons are dropped.
        valid_class_ids: Accepted for API compatibility but NOT used here.
                         All detections are kept so HITL sees every object.
        image_files:     Explicit list of filenames to process (timestamp-filtered
                         subset). If None, all images in image_dir are processed.

    Returns:
        Absolute path to the generated coco_all.json.
    """
    log.info("=" * 60)
    log.info("STEP 1 — YOLOv8 Segmentation (ALL classes → HITL)")
    log.info("  image_dir  : %s", image_dir)
    log.info("  model      : %s", model_path)
    log.info("  output_dir : %s", output_dir)
    if valid_class_ids:
        log.info(
            "  ⚠️  valid_class_ids=%s is IGNORED in Step 1 — all objects are "
            "kept so every detection appears in HITL with its correct label.",
            valid_class_ids,
        )
    log.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "coco_all.json")

    model = YOLO(model_path)
    log.info("Model loaded — classes: %s", model.names)

    coco: Dict = {
        "images": [],
        "annotations": [],
        # Use ALL categories the model knows about
        "categories": [{"id": i, "name": n} for i, n in model.names.items()],
    }

    files = _collect_images(image_dir, image_files)
    log.info("Processing %d images...", len(files))

    ann_id = 1
    class_counts: Dict[int, int] = {}

    for img_id, fname in enumerate(tqdm(files, desc="Step 1")):
        img_path = os.path.join(image_dir, fname)
        result = model(img_path, conf=conf, verbose=False)[0]
        h, w = result.orig_shape

        coco["images"].append({"id": img_id, "file_name": fname, "width": w, "height": h})

        if result.masks is None:
            continue

        classes = result.boxes.cls.cpu().numpy().astype(int)
        for poly, cls_id in zip(result.masks.xy, classes):
            cls_id = int(cls_id)
            # ─── NO class filter here — ALL objects go into HITL ───────────

            poly = np.asarray(poly)
            if poly.shape[0] < 3:
                continue

            area = float(_polygon_area(poly[:, 0], poly[:, 1]))
            if area < min_area:
                continue

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "segmentation": [poly.flatten().tolist()],
                "area": area,
                "iscrowd": 0,
            })
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
            ann_id += 1

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    log.info(
        "✅ Step 1 done — %d images | %d annotations → %s",
        len(coco["images"]), len(coco["annotations"]), output_path,
    )
    log.info("   Per-class annotation counts:")
    for cid, cnt in sorted(class_counts.items()):
        log.info("     class %2d (%s): %d", cid, model.names.get(cid, "?"), cnt)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Add Fine-Grained Categories to COCO
# ─────────────────────────────────────────────────────────────────────────────

def step2_add_categories(
    coco_path: str,
    new_categories: List[str],
    output_dir: str,
) -> str:
    """
    Append new category names to an existing COCO JSON (no duplicates).

    Args:
        coco_path:       Path to coco_all.json (Step 1 output).
        new_categories:  Category name strings to add, e.g. ["Green_HD", "Blue_PP"].
        output_dir:      Directory where coco_added_cats.json is written.

    Returns:
        Absolute path to coco_added_cats.json.
    """
    log.info("=" * 60)
    log.info("STEP 2 — Adding Categories: %s", new_categories)
    log.info("=" * 60)

    output_path = os.path.join(output_dir, "coco_added_cats.json")

    with open(coco_path) as f:
        coco_data = json.load(f)

    categories = coco_data.get("categories", [])
    existing_names = {cat["name"] for cat in categories}
    next_id = max((cat["id"] for cat in categories), default=-1) + 1
    added = 0

    for name in new_categories:
        if name in existing_names:
            log.warning("  Skipping '%s' — already exists.", name)
            continue
        categories.append({"id": next_id, "name": name, "supercategory": "None"})
        log.info("  Added: id=%d  name=%s", next_id, name)
        next_id += 1
        added += 1

    coco_data["categories"] = categories
    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    log.info("✅ Step 2 done — %d new categories added → %s", added, output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Polygon-Tight Crop Generation
# ─────────────────────────────────────────────────────────────────────────────

def step3_generate_crops(
    image_dir: str,
    coco_path: str,
    output_dir: str,
    valid_class_ids: Set[int],
    class_id_to_folder: Optional[Dict[int, str]] = None,
) -> str:
    """
    Crop objects from source images using polygon bounding boxes.

    Only annotations whose category_id is in valid_class_ids are cropped
    (e.g. HD bottles → Color_HD folder, PP bottles → Color_PP folder).
    All other classes still appear in HITL (they were kept in Step 1),
    but they don't need a colour-classifier crop.

    Crop filenames are  <image_stem>__ann<ann_id>.jpg  which lets Step 4
    look up the exact annotation in the COCO JSON by its ID.

    Args:
        image_dir:          Directory containing source images.
        coco_path:          Path to coco_added_cats.json (Step 2 output).
        output_dir:         Directory where crops/ subfolder is created.
        valid_class_ids:    Only crop annotations whose category_id is in this set.
        class_id_to_folder: Maps class_id → subfolder name inside crops/.
                            e.g. {3: "Color_HD", 12: "Color_PP"}.
                            Defaults to "class_<id>" for unknown IDs.

    Returns:
        Absolute path to the crops root directory.
    """
    log.info("=" * 60)
    log.info("STEP 3 — Crop Generation  |  valid_class_ids=%s", valid_class_ids)
    log.info("  class_id_to_folder: %s", class_id_to_folder)
    log.info("=" * 60)

    crop_root = os.path.join(output_dir, "crops")
    os.makedirs(crop_root, exist_ok=True)

    with open(coco_path) as f:
        coco = json.load(f)

    # ── DIAGNOSTIC: show every category_id present in the COCO and whether
    #    it matches a crop folder — this exposes any ID mismatch for Color_PP
    all_cat_ids_in_anns: Dict[int, int] = {}
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        all_cat_ids_in_anns[cid] = all_cat_ids_in_anns.get(cid, 0) + 1

    coco_cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    log.info("  Category IDs found in COCO annotations:")
    for cid, cnt in sorted(all_cat_ids_in_anns.items()):
        matched = cid in valid_class_ids
        folder  = (class_id_to_folder or {}).get(cid, f"class_{cid}")
        log.info(
            "    id=%2d  name=%-20s  count=%5d  crop_match=%s  folder=%s",
            cid, coco_cat_id_to_name.get(cid, "?"), cnt,
            "✅ YES" if matched else "❌ NO ",
            folder if matched else "(skipped)",
        )

    unmatched_ids = set(valid_class_ids) - set(all_cat_ids_in_anns.keys())
    if unmatched_ids:
        log.warning(
            "  ⚠️  valid_class_ids %s have NO annotations in COCO — "
            "check that your --class_folder_map IDs match the actual YOLO class IDs "
            "printed in Step 1.",
            unmatched_ids,
        )

    image_id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    total_saved = 0
    per_folder: Dict[str, int] = {}

    for ann in tqdm(coco["annotations"], desc="Step 3"):
        cls_id = ann["category_id"]
        if cls_id not in valid_class_ids:
            continue

        folder_name = (class_id_to_folder or {}).get(cls_id, f"class_{cls_id}")
        image_name = image_id_to_name.get(ann["image_id"])
        if not image_name:
            continue

        img = cv2.imread(os.path.join(image_dir, image_name))
        if img is None:
            log.warning("  Cannot read: %s", image_name)
            continue

        H, W = img.shape[:2]
        xs, ys = [], []
        for poly in ann["segmentation"]:
            xs.extend(poly[0::2])
            ys.extend(poly[1::2])
        if not xs:
            continue

        x1 = max(0, int(min(xs)));  x2 = min(W, int(max(xs)))
        y1 = max(0, int(min(ys)));  y2 = min(H, int(max(ys)))
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        out_dir = os.path.join(crop_root, folder_name)
        os.makedirs(out_dir, exist_ok=True)

        stem = os.path.splitext(image_name)[0]
        cv2.imwrite(os.path.join(out_dir, f"{stem}__ann{ann['id']}.jpg"), crop)
        total_saved += 1
        per_folder[folder_name] = per_folder.get(folder_name, 0) + 1

    log.info("✅ Step 3 done — %d crops saved → %s", total_saved, crop_root)
    for folder, cnt in sorted(per_folder.items()):
        log.info("   %-20s : %d crops", folder, cnt)
    if not per_folder:
        log.warning(
            "  ⚠️  ZERO crops were saved!  "
            "Check the class IDs above — your --class_folder_map IDs must "
            "match the YOLO model's actual class IDs shown in Step 1."
        )
    return crop_root


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Color Classification → Update COCO (HITL input)
# ─────────────────────────────────────────────────────────────────────────────

def step4_run_color_classification(
    crop_dir: str,
    coco_path: str,
    model_path: str,
    output_dir: str,
    conf: float = 0.25,
    crop_folders: Optional[List[str]] = None,
) -> str:
    """
    Run a color classifier on crop images and update annotation category_ids.

    This produces  coco_updated.json  — the definitive COCO file that is fed
    directly into the HITL review pipeline.

    Args:
        crop_dir:     Root crops directory (Step 3 output).
        coco_path:    Path to coco_added_cats.json (Step 2 output).
        model_path:   Path to the color-classification YOLOv8 .pt weights.
        output_dir:   Directory where coco_updated.json is written.
        conf:         Classifier confidence threshold.
        crop_folders: Subfolders inside crop_dir to process.
                      If None, every subfolder in crop_dir is processed.

    Returns:
        Absolute path to coco_updated.json  ← pass this to the HITL --coco arg.
    """
    log.info("=" * 60)
    log.info("STEP 4 — Color Classification (→ HITL COCO)")
    log.info("  crop_dir   : %s", crop_dir)
    log.info("  model      : %s", model_path)
    log.info("=" * 60)

    output_path = os.path.join(output_dir, "coco_updated.json")

    model = YOLO(model_path)

    with open(coco_path) as f:
        coco_original = json.load(f)

    coco_updated = copy.deepcopy(coco_original)
    category_name_to_id = {cat["name"]: cat["id"] for cat in coco_updated["categories"]}
    image_name_to_id    = {img["file_name"]: img["id"] for img in coco_updated["images"]}
    stem_to_id          = {os.path.splitext(img["file_name"])[0]: img["id"] for img in coco_updated["images"]}
    ann_id_to_ann: Dict[int, Dict] = {ann["id"]: ann for ann in coco_updated["annotations"]}

    stats = {k: 0 for k in
             ("updated", "err_invalid_name", "err_missing_parent",
              "err_missing_ann", "err_no_prediction", "err_class_not_found")}

    folders = crop_folders or [
        d for d in os.listdir(crop_dir) if os.path.isdir(os.path.join(crop_dir, d))
    ]

    for folder in folders:
        folder_path = os.path.join(crop_dir, folder)
        if not os.path.exists(folder_path):
            log.warning("  Folder missing: %s", folder_path)
            continue
        log.info("  Processing folder: %s", folder)

        for crop_name in tqdm(sorted(os.listdir(folder_path)), desc=f"Step 4 [{folder}]"):
            name_no_ext, _ = os.path.splitext(crop_name)

            if "__ann" not in name_no_ext:
                stats["err_invalid_name"] += 1
                continue
            try:
                base, ann_str = name_no_ext.rsplit("__ann", 1)
                ann_id = int(ann_str)
            except ValueError:
                stats["err_invalid_name"] += 1
                continue

            if base not in stem_to_id:
                stats["err_missing_parent"] += 1
                continue

            ann = ann_id_to_ann.get(ann_id)
            if ann is None:
                stats["err_missing_ann"] += 1
                continue

            result = model(os.path.join(folder_path, crop_name), conf=conf, verbose=False)[0]
            if result.boxes is None or len(result.boxes.cls) == 0:
                stats["err_no_prediction"] += 1
                continue

            pred_name = result.names[int(result.boxes.cls[0].item())].strip()
            if pred_name in category_name_to_id:
                ann["category_id"] = category_name_to_id[pred_name]
                stats["updated"] += 1
            else:
                stats["err_class_not_found"] += 1

    with open(output_path, "w") as f:
        json.dump(coco_updated, f, indent=2)

    log.info("✅ Step 4 done")
    for k, v in stats.items():
        log.info("   %-22s: %d", k, v)
    log.info("   Output → %s", output_path)
    return output_path
