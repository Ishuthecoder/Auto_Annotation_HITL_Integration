import os
import json
import numpy as np
from tqdm import tqdm

# =========================================================
# PATHS
# =========================================================
COCO_PATH = "./output_pipeline/coco_final_updated.json"
OUTPUT_LABEL_DIR = "/media/wi/ssd_hub/output_v8/yolo_labels"

os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# =========================================================
# YOUR LABEL.TXT (ORDER MATTERS)
# =========================================================
CLASS_NAMES = [
    "Blue_HD","Red_HD","White_HD","Color_HD","Unknown","Film","Oil_Pouch",
    "Milk_Packaging","HMLD","Clear_PP","White_PP","Black_PP","Color_PP",
    "Red_PP","HIPS","MLP","Newspaper","Cardboard","Paper","Greyboard",
    "Aluminium_Cans","Aluminium_Foil","Steel","PET","Glass_PP",
    "Green_HD","Black_HD","Blue_PP","Green_PP"
]

# name → index mapping
name_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}

# =========================================================
# LOAD COCO
# =========================================================
print("📄 Loading COCO...")
with open(COCO_PATH) as f:
    coco = json.load(f)

# category_id → name
cat_id_to_name = {
    cat["id"]: cat["name"]
    for cat in coco["categories"]
}

# image_id → info
image_id_to_info = {
    img["id"]: img
    for img in coco["images"]
}

# group annotations by image
anns_by_image = {}
for ann in coco["annotations"]:
    anns_by_image.setdefault(ann["image_id"], []).append(ann)

# =========================================================
# CONVERT TO YOLO SEG
# =========================================================
print("🔄 Converting to YOLOv8 segmentation format...")

for img_id, anns in tqdm(anns_by_image.items()):

    img_info = image_id_to_info[img_id]
    w, h = img_info["width"], img_info["height"]

    label_lines = []

    for ann in anns:

        cat_name = cat_id_to_name[ann["category_id"]]

        # skip unknown classes not in label.txt
        if cat_name not in name_to_index:
            continue

        class_id = name_to_index[cat_name]

        for seg in ann["segmentation"]:

            coords = np.array(seg).reshape(-1, 2)

            # normalize
            coords[:, 0] /= w
            coords[:, 1] /= h

            coords = coords.flatten()

            line = [str(class_id)] + [f"{x:.6f}" for x in coords]
            label_lines.append(" ".join(line))

    # save label file
    txt_name = os.path.splitext(img_info["file_name"])[0] + ".txt"
    txt_path = os.path.join(OUTPUT_LABEL_DIR, txt_name)

    with open(txt_path, "w") as f:
        f.write("\n".join(label_lines))

print("✅ Conversion complete")