#!/usr/bin/env python3
"""
run_auto_annotation.py — Master Orchestrator for Auto-Annotation Pipeline
==========================================================================
Runs all 5 steps in sequence, then optionally launches HITL review.

Steps:
  1. YOLO Segmentation → coco.json          (ALL objects, REAL class IDs)
  2. Add Color Categories → coco_added_cats.json
  3. Crop HD & PP objects → crops/Color_HD/, crops/Color_PP/
  4. Color Classification → coco_updated.json
  5. COCO → YOLO labels   → yolo_labels/

Then: HITL gets coco_updated.json (every object, every segmentation, correct labels)

Usage:
  python run_auto_annotation.py \
    --image_dir   /path/to/images \
    --shape_model /path/to/segmentation_model.pt \
    --color_model /path/to/color_classification_model.pt \
    --output_dir  /path/to/output \
    --hd_class_id 3 \
    --pp_class_id 12 \
    --launch_hitl
"""

import argparse
import os
import sys
import json
import copy
import subprocess
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Extra categories to add in step 2 (all compound color+material names)
NEW_CATEGORIES = [
    "Blue_HD", "Red_HD", "White_HD", "Color_HD",
    "Green_HD", "Black_HD",
    "Clear_PP", "White_PP", "Black_PP", "Color_PP",
    "Red_pp", "Blue_PP", "Green_PP", "Glass_PP",
]

# Mapping: color model prediction → compound category name per folder
# Color model predicts generic colors; we combine with material type from folder
HD_COLOR_MAP = {
    "blue": "Blue_HD",
    "red": "Red_HD",
    "white": "White_HD",
    "green": "Green_HD",
    "black": "Black_HD",
    "coloured": "Color_HD",
    "clear transparent": "Color_HD",
    "brown": "Color_HD",
    "yellow": "Color_HD",
    "silver": "Color_HD",
    "grey": "Color_HD",
    "mixed": "Color_HD",
    "unapplicable": "Color_HD",
}
PP_COLOR_MAP = {
    "clear transparent": "Clear_PP",
    "white": "White_PP",
    "black": "Black_PP",
    "red": "Red_pp",
    "blue": "Blue_PP",
    "green": "Green_PP",
    "coloured": "Color_PP",
    "brown": "Color_PP",
    "yellow": "Color_PP",
    "silver": "Color_PP",
    "grey": "Color_PP",
    "mixed": "Color_PP",
    "unapplicable": "Color_PP",
}
FOLDER_COLOR_MAP = {
    "Color_HD": HD_COLOR_MAP,
    "Color_PP": PP_COLOR_MAP,
}

# YOLO label order for step 5
CLASS_NAMES = [
    "Blue_HD","Red_HD","White_HD","Color_HD","Unknown","Film","Oil_Pouch",
    "Milk_Packaging","HMLD","Clear_PP","White_PP","Black_PP","Color_PP",
    "Red_pp","HIPS","MLP","Newspaper","Cardboard","Paper","Greyboard",
    "Aluminium_Cans","Aluminium_Foil","Steel","PET","Glass_PP",
    "Green_HD","Black_HD","Blue_PP","Green_PP"
]


# =========================================================
# UTILS
# =========================================================
def polygon_area(x, y):
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def collect_images(image_dir):
    exts = (".jpg", ".jpeg", ".png")
    return sorted(f for f in os.listdir(image_dir) if f.lower().endswith(exts))


# =========================================================
# STEP 1: YOLO → COCO with REAL class IDs (ALL objects)
# =========================================================
def step1_yolo_to_coco(image_dir, model_path, output_dir, conf, min_area):
    print("\n" + "="*60)
    print("STEP 1 — YOLO Segmentation (ALL classes, REAL IDs)")
    print("="*60)

    coco_path = os.path.join(output_dir, "coco.json")
    vis_dir = os.path.join(output_dir, "yolo_vis")
    os.makedirs(vis_dir, exist_ok=True)

    model = YOLO(model_path)
    print(f"Model classes: {model.names}")

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": n} for i, n in model.names.items()]
    }

    images = collect_images(image_dir)
    print(f"Processing {len(images)} images...")

    ann_id = 1
    class_counts = {}

    for img_id, fname in enumerate(tqdm(images, desc="Step 1")):
        img_path = os.path.join(image_dir, fname)
        result = model(img_path, conf=conf, verbose=False)[0]
        h, w = result.orig_shape

        coco["images"].append({"id": img_id, "file_name": fname, "width": w, "height": h})
        cv2.imwrite(os.path.join(vis_dir, fname), result.plot())

        if result.masks is None:
            continue

        classes = result.boxes.cls.cpu().numpy().astype(int)
        for poly, cls_id in zip(result.masks.xy, classes):
            cls_id = int(cls_id)
            poly = np.asarray(poly)
            if poly.shape[0] < 3:
                continue

            area = polygon_area(poly[:, 0], poly[:, 1])
            if area < min_area:
                continue

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "segmentation": [poly.flatten().tolist()],
                "area": float(area),
                "iscrowd": 0
            })
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
            ann_id += 1

    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"✅ Step 1 done — {len(coco['images'])} images | {len(coco['annotations'])} annotations")
    print("   Per-class counts:")
    for cid, cnt in sorted(class_counts.items()):
        print(f"     class {cid:2d} ({model.names.get(cid, '?'):20s}): {cnt}")

    return coco_path


# =========================================================
# STEP 2: Add Color Categories
# =========================================================
def step2_add_categories(coco_path, output_dir):
    print("\n" + "="*60)
    print("STEP 2 — Adding Color Categories")
    print("="*60)

    out_path = os.path.join(output_dir, "coco_added_cats.json")

    with open(coco_path) as f:
        coco = json.load(f)

    categories = coco.get("categories", [])
    existing = {c["name"] for c in categories}
    next_id = max((c["id"] for c in categories), default=-1) + 1

    for name in NEW_CATEGORIES:
        if name in existing:
            print(f"  ⚠️ Skipping '{name}' — already exists")
            continue
        categories.append({"id": next_id, "name": name, "supercategory": "None"})
        print(f"  Added: id={next_id}  name={name}")
        next_id += 1

    coco["categories"] = categories
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"✅ Step 2 done → {out_path}")
    return out_path


# =========================================================
# STEP 3: Crop HD & PP objects (using annotation IDs)
# =========================================================
def step3_generate_crops(image_dir, coco_path, output_dir, hd_class_id, pp_class_id):
    print("\n" + "="*60)
    print(f"STEP 3 — Cropping HD (class {hd_class_id}) & PP (class {pp_class_id})")
    print("="*60)

    crop_root = os.path.join(output_dir, "cropped_by_label_polygon", "yolo_object")
    os.makedirs(crop_root, exist_ok=True)

    with open(coco_path) as f:
        coco = json.load(f)

    # Diagnostic: show what class IDs exist
    cat_counts = {}
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        cat_counts[cid] = cat_counts.get(cid, 0) + 1

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    print("  Category IDs in COCO:")
    for cid, cnt in sorted(cat_counts.items()):
        is_hd = "✅ HD" if cid == hd_class_id else ""
        is_pp = "✅ PP" if cid == pp_class_id else ""
        print(f"    id={cid:2d} ({cat_names.get(cid, '?'):20s}): {cnt:5d}  {is_hd}{is_pp}")

    if hd_class_id not in cat_counts:
        print(f"  ⚠️ WARNING: --hd_class_id={hd_class_id} not found! Check Step 1 class IDs above.")
    if pp_class_id not in cat_counts:
        print(f"  ⚠️ WARNING: --pp_class_id={pp_class_id} not found! Check Step 1 class IDs above.")

    image_id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    total = {"Color_HD": 0, "Color_PP": 0}

    for ann in tqdm(coco["annotations"], desc="Step 3"):
        cls_id = ann["category_id"]

        if cls_id == hd_class_id:
            folder = "Color_HD"
        elif cls_id == pp_class_id:
            folder = "Color_PP"
        else:
            continue

        image_name = image_id_to_name.get(ann["image_id"])
        if not image_name:
            continue

        img = cv2.imread(os.path.join(image_dir, image_name))
        if img is None:
            continue

        H, W = img.shape[:2]
        xs, ys = [], []
        for poly in ann["segmentation"]:
            xs.extend(poly[0::2])
            ys.extend(poly[1::2])
        if not xs:
            continue

        x1 = max(0, int(min(xs))); x2 = min(W, int(max(xs)))
        y1 = max(0, int(min(ys))); y2 = min(H, int(max(ys)))
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        out_dir = os.path.join(crop_root, folder)
        os.makedirs(out_dir, exist_ok=True)

        stem = os.path.splitext(image_name)[0]
        cv2.imwrite(os.path.join(out_dir, f"{stem}__ann{ann['id']}.jpg"), crop)
        total[folder] += 1

    print(f"✅ Step 3 done — Color_HD: {total['Color_HD']} | Color_PP: {total['Color_PP']}")
    return crop_root


# =========================================================
# STEP 4: Color Classification → coco_updated.json
# =========================================================
def step4_color_classification(crop_root, coco_path, model_path, output_dir, conf):
    print("\n" + "="*60)
    print("STEP 4 — Color Classification (folder + color → compound name)")
    print("="*60)

    out_path = os.path.join(output_dir, "coco_updated.json")

    model = YOLO(model_path)
    print(f"  Color model classes: {model.names}")

    with open(coco_path) as f:
        coco = copy.deepcopy(json.load(f))

    cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    ann_id_to_ann = {ann["id"]: ann for ann in coco["annotations"]}
    stem_to_id = {os.path.splitext(img["file_name"])[0]: img["id"] for img in coco["images"]}

    print("  Compound name mapping:")
    for folder, cmap in FOLDER_COLOR_MAP.items():
        print(f"    {folder}:")
        for color, compound in cmap.items():
            cid = cat_name_to_id.get(compound, '?')
            print(f"      {color:20s} → {compound:12s} (id={cid})")

    stats = {"updated": 0, "err_invalid": 0, "err_parent": 0, "err_ann": 0, "err_no_pred": 0, "err_unmapped": 0}

    BATCH_SIZE = 64  # GPU batch size for faster inference

    for folder in ["Color_HD", "Color_PP"]:
        folder_path = os.path.join(crop_root, folder)
        if not os.path.exists(folder_path):
            print(f"  ⚠️ Missing: {folder_path}")
            continue

        color_map = FOLDER_COLOR_MAP.get(folder, {})

        # Pre-filter valid crops
        print(f"  Scanning: {folder}...")
        valid_crops = []  # list of (crop_path, ann_id)
        for crop_name in sorted(os.listdir(folder_path)):
            name_no_ext = os.path.splitext(crop_name)[0]
            if "__ann" not in name_no_ext:
                stats["err_invalid"] += 1
                continue
            try:
                base, ann_str = name_no_ext.rsplit("__ann", 1)
                ann_id = int(ann_str)
            except ValueError:
                stats["err_invalid"] += 1
                continue

            if base not in stem_to_id:
                stats["err_parent"] += 1
                continue

            ann = ann_id_to_ann.get(ann_id)
            if ann is None:
                stats["err_ann"] += 1
                continue

            valid_crops.append((os.path.join(folder_path, crop_name), ann_id))

        print(f"  Processing: {folder} — {len(valid_crops)} valid crops (batch_size={BATCH_SIZE})")

        # Batch inference
        for i in tqdm(range(0, len(valid_crops), BATCH_SIZE), desc=f"Step 4 [{folder}]"):
            batch = valid_crops[i : i + BATCH_SIZE]
            paths = [p for p, _ in batch]
            ann_ids = [a for _, a in batch]

            results = model(paths, conf=conf, verbose=False)

            for result, aid in zip(results, ann_ids):
                if result.boxes is None or len(result.boxes.cls) == 0:
                    stats["err_no_pred"] += 1
                    continue

                pred_color = result.names[int(result.boxes.cls[0].item())].strip()

                # Map generic color → compound name (e.g. "blue" + Color_HD → "Blue_HD")
                compound_name = color_map.get(pred_color)
                if compound_name and compound_name in cat_name_to_id:
                    ann_id_to_ann[aid]["category_id"] = cat_name_to_id[compound_name]
                    stats["updated"] += 1
                else:
                    stats["err_unmapped"] += 1

    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"✅ Step 4 done")
    for k, v in stats.items():
        print(f"   {k:20s}: {v}")
    return out_path


# =========================================================
# STEP 5: COCO → YOLO labels
# =========================================================
def step5_coco_to_yolo(coco_path, output_dir):
    print("\n" + "="*60)
    print("STEP 5 — COCO → YOLO Segmentation Labels")
    print("="*60)

    label_dir = os.path.join(output_dir, "yolo_labels")
    os.makedirs(label_dir, exist_ok=True)

    name_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}

    with open(coco_path) as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_id_to_info = {img["id"]: img for img in coco["images"]}
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for img_id, anns in tqdm(anns_by_image.items(), desc="Step 5"):
        info = img_id_to_info[img_id]
        w, h = info["width"], info["height"]
        lines = []
        for ann in anns:
            cat_name = cat_id_to_name.get(ann["category_id"], "")
            if cat_name not in name_to_idx:
                continue
            cls = name_to_idx[cat_name]
            for seg in ann["segmentation"]:
                coords = np.array(seg).reshape(-1, 2)
                coords[:, 0] /= w
                coords[:, 1] /= h
                line = [str(cls)] + [f"{x:.6f}" for x in coords.flatten()]
                lines.append(" ".join(line))

        txt = os.path.join(label_dir, os.path.splitext(info["file_name"])[0] + ".txt")
        with open(txt, "w") as f:
            f.write("\n".join(lines))

    print(f"✅ Step 5 done → {label_dir}")
    return label_dir


# =========================================================
# LAUNCH HITL
# =========================================================
def launch_hitl(coco_path, image_dir, output_dir, port, batch_size):
    print("\n" + "="*60)
    print("LAUNCHING HITL REVIEW SERVER")
    print(f"  COCO: {coco_path}")
    print(f"  Images: {image_dir}")
    print(f"  URL: http://0.0.0.0:{port}")
    print("="*60)

    backend = os.path.join(SCRIPT_DIR, "backend", "main.py")
    hitl_out = os.path.join(output_dir, "hitl_corrections")
    os.makedirs(hitl_out, exist_ok=True)

    cmd = [
        sys.executable, backend,
        "--coco", coco_path,
        "--images", image_dir,
        "--output_dir", hitl_out,
        "--batch_size", str(batch_size),
        "--port", str(port),
    ]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("HITL server stopped.")


# =========================================================
# CLI
# =========================================================
def build_parser():
    p = argparse.ArgumentParser(
        description="Auto-Annotation Pipeline → HITL Review",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required paths
    p.add_argument("--image_dir",   required=True,
                   help="Directory containing source images.")
    p.add_argument("--shape_model", required=True,
                   help="YOLOv8 segmentation model (.pt) for Step 1.")
    p.add_argument("--color_model", required=True,
                   help="YOLOv8 color-classification model (.pt) for Step 4.")
    p.add_argument("--output_dir",  default=os.path.join(SCRIPT_DIR, "auto_annotation_pipeline", "output"),
                   help="Root output directory for all pipeline artifacts.")

    # Class IDs for HD and PP
    p.add_argument("--hd_class_id", type=int, default=3,
                   help="YOLO class ID for HD bottles (check Step 1 logs).")
    p.add_argument("--pp_class_id", type=int, default=12,
                   help="YOLO class ID for PP bottles (check Step 1 logs).")

    # Confidence thresholds
    p.add_argument("--shape_conf", type=float, default=0.20,
                   help="Confidence threshold for segmentation model (Step 1).")
    p.add_argument("--color_conf", type=float, default=0.25,
                   help="Confidence threshold for color model (Step 4).")
    p.add_argument("--min_area",   type=float, default=200.0,
                   help="Minimum polygon area (px²).")

    # Pipeline control
    p.add_argument("--start_step", type=int, default=1, choices=[1, 2, 3, 4, 5],
                   help="Resume from this step (skips earlier steps).")
    p.add_argument("--skip_hitl",  action="store_true",
                   help="Skip HITL launch after pipeline.")

    # HITL server
    p.add_argument("--port",       type=int, default=8000,
                   help="Port for HITL server.")
    p.add_argument("--batch_size", type=int, default=2000,
                   help="HITL batch size.")
    p.add_argument("--launch_hitl", action="store_true",
                   help="Auto-launch HITL server after pipeline completes.")

    return p


# =========================================================
# MAIN
# =========================================================
def main():
    args = build_parser().parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("AUTO-ANNOTATION PIPELINE → HITL")
    print(f"  Images:      {args.image_dir}")
    print(f"  Shape model: {args.shape_model}")
    print(f"  Color model: {args.color_model}")
    print(f"  Output:      {output_dir}")
    print(f"  HD class ID: {args.hd_class_id}")
    print(f"  PP class ID: {args.pp_class_id}")
    print(f"  Start step:  {args.start_step}")
    print("="*60)

    # File paths between steps
    coco_path        = os.path.join(output_dir, "coco.json")
    coco_cats_path   = os.path.join(output_dir, "coco_added_cats.json")
    crop_root        = os.path.join(output_dir, "cropped_by_label_polygon", "yolo_object")
    coco_updated     = os.path.join(output_dir, "coco_updated.json")

    # Step 1
    if args.start_step <= 1:
        coco_path = step1_yolo_to_coco(
            args.image_dir, args.shape_model, output_dir,
            args.shape_conf, args.min_area,
        )
    else:
        print(f"\n⏭️  Skipping Step 1 — using {coco_path}")

    # Step 2
    if args.start_step <= 2:
        coco_cats_path = step2_add_categories(coco_path, output_dir)
    else:
        print(f"\n⏭️  Skipping Step 2 — using {coco_cats_path}")

    # Step 3
    if args.start_step <= 3:
        crop_root = step3_generate_crops(
            args.image_dir, coco_cats_path, output_dir,
            args.hd_class_id, args.pp_class_id,
        )
    else:
        print(f"\n⏭️  Skipping Step 3 — using {crop_root}")

    # Step 4
    if args.start_step <= 4:
        coco_updated = step4_color_classification(
            crop_root, coco_cats_path, args.color_model,
            output_dir, args.color_conf,
        )
    else:
        print(f"\n⏭️  Skipping Step 4 — using {coco_updated}")

    # Step 5
    if args.start_step <= 5:
        step5_coco_to_yolo(coco_updated, output_dir)

    print("\n" + "="*60)
    print("✅ ALL STEPS COMPLETE")
    print(f"   Final COCO: {coco_updated}")
    print("="*60)

    # Launch HITL
    if args.launch_hitl and not args.skip_hitl:
        launch_hitl(coco_updated, args.image_dir, output_dir, args.port, args.batch_size)
    elif not args.skip_hitl:
        print(f"\nTo launch HITL:")
        print(f"  python run_auto_annotation.py --image_dir {args.image_dir} "
              f"--shape_model {args.shape_model} --color_model {args.color_model} "
              f"--start_step 5 --skip_hitl --launch_hitl")
        print(f"\n  OR directly:")
        print(f"  python backend/main.py --coco {coco_updated} --images {args.image_dir} "
              f"--output_dir {output_dir}/hitl_corrections --port {args.port}")


if __name__ == "__main__":
    main()
