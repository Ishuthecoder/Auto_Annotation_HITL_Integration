import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# =========================================================
# PATHS (UPDATED)
# =========================================================
IMAGE_DIR = "/media/wi/ssd_hub/Avinash_work/downloaded_images_v10"
MODEL_PATH = "/home/wi/Avinash_Works/waste-masknet/outputs/runs/yolov8m-seg_seg_20260217_123427/weights/best.pt"

OUTPUT_DIR = "/media/wi/ssd_hub/output_v8"
COCO_PATH = os.path.join(OUTPUT_DIR, "coco_all.json")
CROP_DIR = os.path.join(OUTPUT_DIR, "cropped_by_label_polygon/yolo_object")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

CONF = 0.2

# IMPORTANT: your known mapping
HD_CLASS_ID = 3
PP_CLASS_ID = 12

VALID_CLASSES = {HD_CLASS_ID, PP_CLASS_ID}

# =========================================================
# STEP 1: YOLO → COCO
# =========================================================
print("🚀 Running YOLO and creating COCO...")

model = YOLO(MODEL_PATH)

# 🔍 sanity check (VERY IMPORTANT)
print("Class mapping check:")
print(f"{HD_CLASS_ID} → {model.names[HD_CLASS_ID]}")
print(f"{PP_CLASS_ID} → {model.names[PP_CLASS_ID]}")

coco = {
    "images": [],
    "annotations": [],
    "categories": []
}

# categories from model
for i, name in model.names.items():
    coco["categories"].append({"id": i, "name": name})

ann_id = 1
img_id = 0

image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

for fname in tqdm(image_files):

    img_path = os.path.join(IMAGE_DIR, fname)
    result = model(img_path, conf=CONF, verbose=False)[0]

    h, w = result.orig_shape

    coco["images"].append({
        "id": img_id,
        "file_name": fname,
        "width": w,
        "height": h
    })

    if result.masks is not None:

        classes = result.boxes.cls.cpu().numpy().astype(int)

        for poly, cls in zip(result.masks.xy, classes):

            poly = np.asarray(poly)

            if poly.shape[0] < 3:
                continue

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(cls),
                "segmentation": [poly.flatten().tolist()],
                "iscrowd": 0
            })

            ann_id += 1

    img_id += 1

# save coco
with open(COCO_PATH, "w") as f:
    json.dump(coco, f, indent=2)

print(f"✅ COCO saved at {COCO_PATH}")

# =========================================================
# STEP 2: BUILD LOOKUP TABLE
# =========================================================
print("🧠 Building lookup table...")

image_id_to_name = {
    img["id"]: img["file_name"]
    for img in coco["images"]
}

lookup = {}

for ann in coco["annotations"]:
    image_name = image_id_to_name[ann["image_id"]]
    key = f"{image_name}__ann{ann['id']}"

    lookup[key] = {
        "category_id": ann["category_id"],
        "segmentation": ann["segmentation"]
    }

print(f"✅ Lookup table created with {len(lookup)} entries")

# =========================================================
# STEP 3: CROP WITH CORRECT FOLDER NAMES
# =========================================================
print("✂️ Cropping only HD & PP objects...")

total_saved = 0

for key, val in tqdm(lookup.items()):

    cls_id = val["category_id"]

    if cls_id not in VALID_CLASSES:
        continue

    # =====================================================
    # 🔥 CRITICAL FIX: MAP ID → FOLDER NAME
    # =====================================================
    if cls_id == HD_CLASS_ID:
        folder_name = "Color_HD"
    elif cls_id == PP_CLASS_ID:
        folder_name = "Color_PP"
    else:
        continue

    image_name, ann_part = key.split("__ann")
    ann_id_val = int(ann_part)

    img_path = os.path.join(IMAGE_DIR, image_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    H, W = img.shape[:2]

    xs, ys = [], []

    for poly in val["segmentation"]:
        xs.extend(poly[0::2])
        ys.extend(poly[1::2])

    if not xs:
        continue

    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        continue

    # correct folder
    out_dir = os.path.join(CROP_DIR, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    out_name = f"{image_name}__ann{ann_id_val}.jpg"

    cv2.imwrite(os.path.join(out_dir, out_name), crop)
    total_saved += 1

print(f"✅ Cropping complete | Saved: {total_saved} crops")