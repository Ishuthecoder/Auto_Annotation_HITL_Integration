# import os
# import json
# import time
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from pycocotools.coco import COCO
# from tqdm import tqdm

# # =========================================================
# # CONFIG
# # =========================================================
# BASE_DIR = "auto_annotation_pipeline"

# IMAGE_DIR = "/home/wi/Avinash_Works/Auto_Annotation/auto_annotation_pipeline/downloaded_images_v4"
# OUTPUT_DIR = f"{BASE_DIR}/output_v5"

# VIS_DIR = f"{OUTPUT_DIR}/yolo_vis"
# CROP_DIR = f"{OUTPUT_DIR}/cropped_by_label_polygon"
# COCO_JSON = f"{OUTPUT_DIR}/coco.json"
# MANIFEST_PATH = f"{OUTPUT_DIR}/manifest.jsonl"

# MODEL_PATH = "/home/wi/Avinash_Works/waste-masknet/outputs/runs/yolov8m-seg_seg_20260202_174102/weights/best.pt"

# CONF = 0.25
# MIN_AREA = 200
# PAD_RATIO = 0.05
# CLASS_NAME = "yolo_object"

# os.makedirs(VIS_DIR, exist_ok=True)
# os.makedirs(CROP_DIR, exist_ok=True)

# # =========================================================
# # UTILS
# # =========================================================
# def polygon_area(x, y):
#     return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# # =========================================================
# # YOLO → COCO POLYGON
# # =========================================================
# def run_yolo_and_create_coco():
#     print("🚀 Running YOLOv8 segmentation...")
#     model = YOLO(MODEL_PATH)

#     coco = {
#         "images": [],
#         "annotations": [],
#         "categories": [{"id": 1, "name": CLASS_NAME}]
#     }

#     ann_id = 1
#     img_id = 0
#     start = time.time()

#     images = sorted(
#         f for f in os.listdir(IMAGE_DIR)
#         if f.lower().endswith((".jpg", ".jpeg", ".png"))
#     )

#     for fname in images:
#         img_path = os.path.join(IMAGE_DIR, fname)
#         result = model(img_path, conf=CONF, verbose=False)[0]

#         h, w = result.orig_shape

#         coco["images"].append({
#             "id": img_id,
#             "file_name": fname,
#             "width": w,
#             "height": h
#         })

#         # Visualization
#         cv2.imwrite(os.path.join(VIS_DIR, fname), result.plot())

#         if result.masks is not None:
#             for poly in result.masks.xy:
#                 poly = np.array(poly)
#                 if poly.shape[0] < 3:
#                     continue

#                 area = polygon_area(poly[:, 0], poly[:, 1])

#                 coco["annotations"].append({
#                     "id": ann_id,
#                     "image_id": img_id,
#                     "category_id": 1,
#                     "segmentation": [poly.flatten().tolist()],
#                     "area": float(area),
#                     "iscrowd": 0
#                 })
#                 ann_id += 1

#         img_id += 1

#         if img_id % 10 == 0:
#             print(f"⏳ {img_id}/{len(images)} | {(time.time()-start)/img_id:.2f}s/img")

#     with open(COCO_JSON, "w") as f:
#         json.dump(coco, f, indent=2)

#     print(f"✅ COCO saved → {COCO_JSON}")

# # =========================================================
# # CVAT MANIFEST
# # =========================================================
# def create_cvat_manifest():
#     print("📄 Creating CVAT manifest...")
#     images = sorted(os.listdir(IMAGE_DIR))

#     with open(MANIFEST_PATH, "w") as f:
#         for img in images:
#             if img.lower().endswith((".jpg", ".jpeg", ".png")):
#                 f.write(json.dumps({"name": img}) + "\n")

#     print(f"✅ Manifest saved → {MANIFEST_PATH}")

# # =========================================================
# # POLYGON-TIGHT CROPPING
# # =========================================================
# def crop_from_coco():
#     print("✂️ Polygon-tight cropping...")
#     coco = COCO(COCO_JSON)

#     for img_id in tqdm(coco.getImgIds()):
#         img_info = coco.loadImgs(img_id)[0]
#         img_path = os.path.join(IMAGE_DIR, img_info["file_name"])
#         img = cv2.imread(img_path)

#         if img is None:
#             continue

#         H, W = img.shape[:2]
#         anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

#         for idx, ann in enumerate(anns):
#             xs, ys = [], []
#             for poly in ann["segmentation"]:
#                 xs.extend(poly[0::2])
#                 ys.extend(poly[1::2])

#             if not xs:
#                 continue

#             x1, x2 = int(min(xs)), int(max(xs))
#             y1, y2 = int(min(ys)), int(max(ys))

#             if (x2 - x1) * (y2 - y1) < MIN_AREA:
#                 continue

#             px = int((x2 - x1) * PAD_RATIO)
#             py = int((y2 - y1) * PAD_RATIO)

#             x1, y1 = max(0, x1 - px), max(0, y1 - py)
#             x2, y2 = min(W, x2 + px), min(H, y2 + py)

#             crop = img[y1:y2, x1:x2]
#             if crop.size == 0:
#                 continue

#             out_dir = os.path.join(CROP_DIR, CLASS_NAME)
#             os.makedirs(out_dir, exist_ok=True)

#             out_name = f"{os.path.splitext(img_info['file_name'])[0]}_obj{idx}.jpg"
#             cv2.imwrite(os.path.join(out_dir, out_name), crop)

#     print("✅ Cropping completed")

# # =========================================================
# # MAIN
# # =========================================================
# if __name__ == "__main__":
#     run_yolo_and_create_coco()
#     create_cvat_manifest()
#     crop_from_coco()


import os
import json
import time
import cv2
import numpy as np
from ultralytics import YOLO
from pycocotools.coco import COCO
from tqdm import tqdm

# =========================================================
# PATH SETUP (ABSOLUTE & PIPELINE-SAFE)
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "auto_annotation_pipeline")

IMAGE_DIR = "//home/wi/Avinash_Works/Auto_Annotation/auto_annotation_pipeline/new"
OUTPUT_DIR = os.path.join(BASE_DIR, "output_v6")

VIS_DIR = os.path.join(OUTPUT_DIR, "yolo_vis")
CROP_DIR = os.path.join(OUTPUT_DIR, "cropped_by_label_polygon")
COCO_JSON = os.path.join(OUTPUT_DIR, "coco.json")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.jsonl")

MODEL_PATH = "/home/wi/Avinash_Works/waste-masknet/outputs/runs/yolov8m-seg_seg_20260205_162236/weights/best.pt"

CONF = 0.20
MIN_AREA = 200
PAD_RATIO = 0.05
CLASS_NAME = "yolo_object"

os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

# =========================================================
# UTILS
# =========================================================
def polygon_area(x, y):
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# =========================================================
# YOLO → COCO CREATION
# =========================================================
def run_yolo_and_create_coco():
    print("🚀 Running YOLOv8 segmentation...")
    model = YOLO(MODEL_PATH)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": CLASS_NAME}]
    }

    ann_id = 1
    img_id = 0
    start = time.time()

    images = sorted(
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    for fname in images:
        img_path = os.path.join(IMAGE_DIR, fname)
        result = model(img_path, conf=CONF, verbose=False)[0]

        h, w = result.orig_shape

        coco["images"].append({
            "id": img_id,
            "file_name": fname,
            "width": w,
            "height": h
        })

        cv2.imwrite(os.path.join(VIS_DIR, fname), result.plot())

        if result.masks is not None:
            for poly in result.masks.xy:
                poly = np.asarray(poly)
                if poly.shape[0] < 3:
                    continue

                area = polygon_area(poly[:, 0], poly[:, 1])

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": [poly.flatten().tolist()],
                    "area": float(area),
                    "iscrowd": 0
                })
                ann_id += 1

        img_id += 1

        if img_id % 10 == 0:
            print(f"⏳ {img_id}/{len(images)} | {(time.time()-start)/img_id:.2f}s/img")

    with open(COCO_JSON, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"✅ COCO annotations CREATED at:\n{COCO_JSON}")

# =========================================================
# CVAT MANIFEST
# =========================================================
def create_cvat_manifest():
    print("📄 Creating CVAT manifest...")

    with open(MANIFEST_PATH, "w") as f:
        for img in sorted(os.listdir(IMAGE_DIR)):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                f.write(json.dumps({"name": img}) + "\n")

    print(f"✅ Manifest saved → {MANIFEST_PATH}")

# =========================================================
# POLYGON-TIGHT CROPPING (READS COCO)
# =========================================================
def crop_from_coco():
    print("✂️ Polygon-tight cropping...")

    if not os.path.exists(COCO_JSON):
        raise FileNotFoundError(
            f"❌ COCO file not found.\n"
            f"Expected at: {COCO_JSON}\n"
            f"👉 Run YOLO step first."
        )

    coco = COCO(COCO_JSON)

    for img_id in tqdm(coco.getImgIds()):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(IMAGE_DIR, img_info["file_name"])
        img = cv2.imread(img_path)

        if img is None:
            continue

        H, W = img.shape[:2]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

        for idx, ann in enumerate(anns):
            xs, ys = [], []
            for poly in ann["segmentation"]:
                xs.extend(poly[0::2])
                ys.extend(poly[1::2])

            if not xs:
                continue

            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))

            if (x2 - x1) * (y2 - y1) < MIN_AREA:
                continue

            px = int((x2 - x1) * PAD_RATIO)
            py = int((y2 - y1) * PAD_RATIO)

            x1, y1 = max(0, x1 - px), max(0, y1 - py)
            x2, y2 = min(W, x2 + px), min(H, y2 + py)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            out_dir = os.path.join(CROP_DIR, CLASS_NAME)
            os.makedirs(out_dir, exist_ok=True)

            out_name = f"{os.path.splitext(img_info['file_name'])[0]}_obj{idx}.jpg"
            cv2.imwrite(os.path.join(out_dir, out_name), crop)

    print("✅ Cropping completed")

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    run_yolo_and_create_coco()
    create_cvat_manifest()
    crop_from_coco()
