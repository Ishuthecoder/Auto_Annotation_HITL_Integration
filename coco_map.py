# # # import os
# # # import json
# # # import copy
# # # import re
# # # from ultralytics import YOLO
# # # from tqdm import tqdm

# # # # =========================================================
# # # # PATHS
# # # # =========================================================
# # # CROP_DIR = "/media/wi/ssd_hub/Ishika_works/color_dataset/cropped_objects"
# # # # IMPORTANT: Use the JSON that has the 29 categories!
# # # ORIGINAL_COCO = "/media/wi/ssd_hub/Ishika_works/color_dataset/coco_added_cats.json"
# # # OUTPUT_COCO = "/media/wi/ssd_hub/Ishika_works/coco_final_updated.json"
# # # MODEL_PATH = "/media/wi/ssd_hub/Ishika_works/training_runs/yolov8m_color_seg/weights/best.pt"

# # # # =========================================================
# # # # INITIALIZATION
# # # # =========================================================
# # # print("Loading YOLO model...")
# # # model = YOLO(MODEL_PATH)

# # # print("Loading COCO file...")
# # # with open(ORIGINAL_COCO) as f:
# # #     coco_original = json.load(f)

# # # coco_updated = copy.deepcopy(coco_original)

# # # # Build Lookups
# # # category_name_to_id = {cat["name"]: cat["id"] for cat in coco_updated["categories"]}
# # # image_name_to_id = {img["file_name"]: img["id"] for img in coco_updated["images"]}

# # # annotations_by_image = {}
# # # for ann in coco_updated["annotations"]:
# # #     annotations_by_image.setdefault(ann["image_id"], []).append(ann)

# # # # =========================================================
# # # # REGEX DEFINITION
# # # # =========================================================
# # # # This safely extracts: 1) Base Name, 2) Material (HD/PP), 3) Index
# # # pattern = re.compile(r"(.*)__Color_(HD|PP)_(\d+)\.(jpg|jpeg|png)", re.IGNORECASE)

# # # # =========================================================
# # # # PROCESS CROPS
# # # # =========================================================
# # # updated = 0
# # # skipped = 0

# # # for folder in ["Color_HD", "Color_PP"]:
# # #     folder_path = os.path.join(CROP_DIR, folder)
# # #     if not os.path.exists(folder_path):
# # #         continue

# # #     print(f"\nProcessing folder: {folder}")
# # #     for crop_name in tqdm(os.listdir(folder_path)):
        
# # #         match = pattern.match(crop_name)
# # #         if not match:
# # #             continue

# # #         base, material, idx_str, ext = match.groups()
# # #         idx = int(idx_str)

# # #         # Find the parent image ID
# # #         parent_image_name = None
# # #         for test_ext in [".jpg", ".jpeg", ".png"]:
# # #             if (base + test_ext) in image_name_to_id:
# # #                 parent_image_name = base + test_ext
# # #                 break

# # #         if parent_image_name is None:
# # #             skipped += 1
# # #             continue

# # #         image_id = image_name_to_id[parent_image_name]
# # #         anns = annotations_by_image.get(image_id, [])

# # #         # Filter only the annotations that are currently Color_HD (3) or Color_PP (12)
# # #         filtered_anns = [a for a in anns if a["category_id"] in [3, 12]]

# # #         if idx >= len(filtered_anns):
# # #             skipped += 1
# # #             continue

# # #         ann = filtered_anns[idx]
# # #         crop_path = os.path.join(folder_path, crop_name)

# # #         # Run AI Inference
# # #         result = model(crop_path, conf=0.25, verbose=False)[0]

# # #         if result.boxes is None or len(result.boxes.cls) == 0:
# # #             skipped += 1
# # #             continue

# # #         # Format the prediction (e.g., 'blue' + 'PP' -> 'Blue_PP')
# # #         pred_id = int(result.boxes.cls[0].item())
# # #         pred_color = result.names[pred_id].strip().capitalize()
        
# # #         new_class_name = f"{pred_color}_{material}"

# # #         # If the combo exists (e.g., Blue_PP is in our 29 categories), update it!
# # #         if new_class_name in category_name_to_id:
# # #             ann["category_id"] = category_name_to_id[new_class_name]
# # #             updated += 1
# # #         else:
# # #             skipped += 1

# # # # =========================================================
# # # # SAVE FILE
# # # # =========================================================
# # # with open(OUTPUT_COCO, "w") as f:
# # #     json.dump(coco_updated, f, indent=2)

# # # print("\n===== SUMMARY =====")
# # # print(f"✅ Successfully updated: {updated} annotations")
# # # print(f"⚠️ Skipped crops: {skipped}")
# # # print(f"💾 Saved final JSON to: {OUTPUT_COCO}")




# # import os
# # import json
# # import copy
# # from ultralytics import YOLO
# # from tqdm import tqdm

# # # =========================================================
# # # PATHS
# # # =========================================================
# # CROP_DIR = "/media/wi/ssd_hub/Ishika_works/color_dataset/cropped_objects"
# # ORIGINAL_COCO = "/media/wi/ssd_hub/Ishika_works/color_dataset/coco_added_cats.json"
# # OUTPUT_COCO = "/media/wi/ssd_hub/Ishika_works/coco_final_updated.json"
# # MODEL_PATH = "/media/wi/ssd_hub/Ishika_works/training_runs/yolov8m_color_seg/weights/best.pt"

# # # =========================================================
# # # INITIALIZATION
# # # =========================================================
# # print("Loading YOLO model...")
# # model = YOLO(MODEL_PATH)

# # print("Loading COCO file...")
# # with open(ORIGINAL_COCO) as f:
# #     coco_original = json.load(f)

# # coco_updated = copy.deepcopy(coco_original)

# # category_name_to_id = {cat["name"]: cat["id"] for cat in coco_updated["categories"]}
# # image_name_to_id = {img["file_name"]: img["id"] for img in coco_updated["images"]}

# # annotations_by_image = {}
# # for ann in coco_updated["annotations"]:
# #     annotations_by_image.setdefault(ann["image_id"], []).append(ann)

# # # =========================================================
# # # TRACKERS
# # # =========================================================
# # stats = {
# #     "updated": 0,
# #     "err_invalid_name": 0,
# #     "err_missing_parent": 0,
# #     "err_idx_out_of_bounds": 0,
# #     "err_no_ai_prediction": 0,
# #     "err_class_not_in_coco": 0
# # }

# # # =========================================================
# # # PROCESS CROPS
# # # =========================================================
# # for folder in ["Color_HD", "Color_PP"]:
# #     folder_path = os.path.join(CROP_DIR, folder)
# #     if not os.path.exists(folder_path):
# #         continue

# #     # Determine material purely based on folder name
# #     material = "HD" if "HD" in folder else "PP"

# #     print(f"\nProcessing folder: {folder} (Target Material: {material})")
    
# #     for crop_name in tqdm(os.listdir(folder_path)):
        
# #         name_no_ext, ext = os.path.splitext(crop_name)

# #         # 1. Safely split exactly at the last "__" 
# #         # e.g., "Image_Name__7" -> base="Image_Name", idx_str="7"
# #         if "__" not in name_no_ext:
# #             stats["err_invalid_name"] += 1
# #             continue

# #         try:
# #             base, idx_str = name_no_ext.rsplit("__", 1)
# #             idx = int(idx_str)
# #         except ValueError:
# #             stats["err_invalid_name"] += 1
# #             continue

# #         # 2. Find Parent Image in COCO
# #         parent_image_name = None
# #         for test_ext in [".jpg", ".jpeg", ".png"]:
# #             if (base + test_ext) in image_name_to_id:
# #                 parent_image_name = base + test_ext
# #                 break

# #         if parent_image_name is None:
# #             stats["err_missing_parent"] += 1
# #             continue

# #         # 3. Get Annotations
# #         image_id = image_name_to_id[parent_image_name]
# #         anns = annotations_by_image.get(image_id, [])
        
# #         # We assume the index '7' refers to the target annotations
# #         filtered_anns = [a for a in anns if a["category_id"] in [3, 12]]

# #         # Fallback: if '7' is the absolute index in the image rather than the filtered list
# #         if idx < len(filtered_anns):
# #             ann = filtered_anns[idx]
# #         elif idx < len(anns):
# #             ann = anns[idx]
# #         else:
# #             stats["err_idx_out_of_bounds"] += 1
# #             continue

# #         crop_path = os.path.join(folder_path, crop_name)

# #         # 4. Run AI Inference
# #         result = model(crop_path, conf=0.25, verbose=False)[0]

# #         if result.boxes is None or len(result.boxes.cls) == 0:
# #             stats["err_no_ai_prediction"] += 1
# #             continue

# #         # 5. Format and Update Class
# #         pred_id = int(result.boxes.cls[0].item())
# #         pred_color = result.names[pred_id].strip().capitalize()
        
# #         # Creates "Blue_HD" or "Green_PP"
# #         new_class_name = f"{pred_color}_{material}"

# #         if new_class_name in category_name_to_id:
# #             ann["category_id"] = category_name_to_id[new_class_name]
# #             stats["updated"] += 1
# #         else:
# #             stats["err_class_not_in_coco"] += 1

# # # =========================================================
# # # SAVE FILE
# # # =========================================================
# # with open(OUTPUT_COCO, "w") as f:
# #     json.dump(coco_updated, f, indent=2)

# # print("\n===== DETAILED SUMMARY =====")
# # print(f"✅ Successfully updated     : {stats['updated']}")
# # print(f"❌ Skipped (Bad Filename)   : {stats['err_invalid_name']}")
# # print(f"❌ Skipped (No Parent Img)  : {stats['err_missing_parent']}")
# # print(f"❌ Skipped (Index Mismatch) : {stats['err_idx_out_of_bounds']}")
# # print(f"❌ Skipped (AI Failed)      : {stats['err_no_ai_prediction']}")
# # print(f"❌ Skipped (Class Not Found): {stats['err_class_not_in_coco']}")
# # print(f"💾 Saved final JSON to      : {OUTPUT_COCO}")





# import os
# import json
# import copy
# from ultralytics import YOLO
# from tqdm import tqdm

# # =========================================================
# # PATHS
# # =========================================================
# CROP_DIR = "/media/wi/ssd_hub/Ishika_works/color_dataset/cropped_objects"
# ORIGINAL_COCO = "/media/wi/ssd_hub/Ishika_works/color_dataset/coco_added_cats.json"
# OUTPUT_COCO = "/media/wi/ssd_hub/Ishika_works/coco_final_updated.json"
# MODEL_PATH = "/media/wi/ssd_hub/Ishika_works/training_runs/yolov8m_color_seg/weights/best.pt"

# # =========================================================
# # INITIALIZATION
# # =========================================================
# print("Loading YOLO model...")
# model = YOLO(MODEL_PATH)

# print("Loading COCO file...")
# with open(ORIGINAL_COCO) as f:
#     coco_original = json.load(f)

# coco_updated = copy.deepcopy(coco_original)

# category_name_to_id = {cat["name"]: cat["id"] for cat in coco_updated["categories"]}
# image_name_to_id = {img["file_name"]: img["id"] for img in coco_updated["images"]}

# annotations_by_image = {}
# for ann in coco_updated["annotations"]:
#     annotations_by_image.setdefault(ann["image_id"], []).append(ann)

# # =========================================================
# # TRACKERS
# # =========================================================
# stats = {
#     "updated": 0,
#     "err_invalid_name": 0,
#     "err_missing_parent": 0,
#     "err_idx_out_of_bounds": 0,
#     "err_no_ai_prediction": 0,
#     "err_class_not_in_coco": 0
# }

# # =========================================================
# # PROCESS CROPS
# # =========================================================
# for folder in ["Color_HD", "Color_PP"]:
#     folder_path = os.path.join(CROP_DIR, folder)
#     if not os.path.exists(folder_path):
#         continue

#     # Determine material purely based on folder name
#     material = "HD" if "HD" in folder else "PP"

#     print(f"\nProcessing folder: {folder} (Target Material: {material})")
    
#     for crop_name in tqdm(os.listdir(folder_path)):
        
#         name_no_ext, ext = os.path.splitext(crop_name)

#         if "__" not in name_no_ext:
#             stats["err_invalid_name"] += 1
#             continue

#         try:
#             base, idx_str = name_no_ext.rsplit("__", 1)
#             idx = int(idx_str)
#         except ValueError:
#             stats["err_invalid_name"] += 1
#             continue

#         # 2. Find Parent Image in COCO (Now tests for the swallowed underscore!)
#         parent_image_name = None
#         test_variations = [
#             f"{base}.jpg", f"{base}.jpeg", f"{base}.png",
#             f"{base}_.jpg", f"{base}_.jpeg", f"{base}_.png"
#         ]

#         for test_name in test_variations:
#             if test_name in image_name_to_id:
#                 parent_image_name = test_name
#                 break

#         if parent_image_name is None:
#             stats["err_missing_parent"] += 1
#             continue

#         # 3. Get Annotations
#         image_id = image_name_to_id[parent_image_name]
#         anns = annotations_by_image.get(image_id, [])
        
#         # Original generation script only incremented index for HD (3) and PP (12)
#         filtered_anns = [a for a in anns if a["category_id"] in [3, 12]]

#         if idx < len(filtered_anns):
#             ann = filtered_anns[idx]
#         else:
#             stats["err_idx_out_of_bounds"] += 1
#             continue

#         crop_path = os.path.join(folder_path, crop_name)

#         # 4. Run AI Inference
#         result = model(crop_path, conf=0.25, verbose=False)[0]

#         if result.boxes is None or len(result.boxes.cls) == 0:
#             stats["err_no_ai_prediction"] += 1
#             continue

#         # 5. Format and Update Class
#         pred_id = int(result.boxes.cls[0].item())
#         pred_color = result.names[pred_id].strip().capitalize()
        
#         # Creates "Blue_HD" or "Green_PP"
#         new_class_name = f"{pred_color}_{material}"

#         if new_class_name in category_name_to_id:
#             ann["category_id"] = category_name_to_id[new_class_name]
#             stats["updated"] += 1
#         else:
#             stats["err_class_not_in_coco"] += 1

# # =========================================================
# # SAVE FILE
# # =========================================================
# with open(OUTPUT_COCO, "w") as f:
#     json.dump(coco_updated, f, indent=2)

# print("\n===== DETAILED SUMMARY =====")
# print(f"✅ Successfully updated     : {stats['updated']}")
# print(f"❌ Skipped (Bad Filename)   : {stats['err_invalid_name']}")
# print(f"❌ Skipped (No Parent Img)  : {stats['err_missing_parent']}")
# print(f"❌ Skipped (Index Mismatch) : {stats['err_idx_out_of_bounds']}")
# print(f"❌ Skipped (AI Failed)      : {stats['err_no_ai_prediction']}")
# print(f"❌ Skipped (Class Not Found): {stats['err_class_not_in_coco']}")
# print(f"💾 Saved final JSON to      : {OUTPUT_COCO}")






# import os
# import json
# import copy
# from ultralytics import YOLO
# from tqdm import tqdm

# # =========================================================
# # PATHS
# # =========================================================
# CROP_DIR = "/media/wi/ssd_hub/output_v8/cropped_by_label_polygon/yolo_object"
# ORIGINAL_COCO = "/media/wi/ssd_hub/output_v8/coco_added_cats.json"
# OUTPUT_COCO = "/media/wi/ssd_hub/output_v8/coco_updated.json"
# MODEL_PATH = "/media/wi/ssd_hub/Ishika_works/training_runs/yolov8m_color_seg/weights/best.pt"

# # =========================================================
# # INITIALIZATION
# # =========================================================
# print("Loading YOLO model...")
# model = YOLO(MODEL_PATH)

# print("Loading COCO file...")
# with open(ORIGINAL_COCO) as f:
#     coco_original = json.load(f)

# coco_updated = copy.deepcopy(coco_original)

# # Creates a dynamic dictionary of all 29 categories
# category_name_to_id = {cat["name"]: cat["id"] for cat in coco_updated["categories"]}
# image_name_to_id = {img["file_name"]: img["id"] for img in coco_updated["images"]}

# annotations_by_image = {}
# for ann in coco_updated["annotations"]:
#     annotations_by_image.setdefault(ann["image_id"], []).append(ann)

# # =========================================================
# # TRACKERS
# # =========================================================
# stats = {
#     "updated": 0,
#     "err_invalid_name": 0,
#     "err_missing_parent": 0,
#     "err_idx_out_of_bounds": 0,
#     "err_no_ai_prediction": 0,
#     "err_class_not_in_coco": 0
# }

# # =========================================================
# # PROCESS CROPS
# # =========================================================
# for folder in ["Color_HD", "Color_PP"]:
#     folder_path = os.path.join(CROP_DIR, folder)
#     if not os.path.exists(folder_path):
#         continue

#     # Determine material purely based on folder name
#     material = "HD" if "HD" in folder else "PP"

#     print(f"\nProcessing folder: {folder} (Target Material: {material})")
    
#     for crop_name in tqdm(os.listdir(folder_path)):
        
#         name_no_ext, ext = os.path.splitext(crop_name)

#         if "__" not in name_no_ext:
#             stats["err_invalid_name"] += 1
#             continue

#         try:
#             base, idx_str = name_no_ext.rsplit("__", 1)
#             idx = int(idx_str)
#         except ValueError:
#             stats["err_invalid_name"] += 1
#             continue

#         # 2. Find Parent Image in COCO
#         parent_image_name = None
#         test_variations = [
#             f"{base}.jpg", f"{base}.jpeg", f"{base}.png",
#             f"{base}_.jpg", f"{base}_.jpeg", f"{base}_.png"
#         ]

#         for test_name in test_variations:
#             if test_name in image_name_to_id:
#                 parent_image_name = test_name
#                 break

#         if parent_image_name is None:
#             stats["err_missing_parent"] += 1
#             continue

#         # 3. Get Annotations
#         image_id = image_name_to_id[parent_image_name]
#         anns = annotations_by_image.get(image_id, [])
        
#         filtered_anns = [a for a in anns if a["category_id"] in [3, 12]]

#         # Smart Index Matcher
#         if idx < len(filtered_anns):
#             ann = filtered_anns[idx]
#         elif idx < len(anns) and anns[idx]["category_id"] in [3, 12]:
#             ann = anns[idx]
#         else:
#             stats["err_idx_out_of_bounds"] += 1
#             continue

#         crop_path = os.path.join(folder_path, crop_name)

#         # 4. Run AI Inference
#         result = model(crop_path, conf=0.25, verbose=False)[0]

#         if result.boxes is None or len(result.boxes.cls) == 0:
#             stats["err_no_ai_prediction"] += 1
#             continue

#         # 5. Format and Update Class
#         pred_id = int(result.boxes.cls[0].item())
#         pred_color = result.names[pred_id].strip().capitalize()
        
#         new_class_name = f"{pred_color}_{material}"

#         # 6. Global Category Check
#         # If the AI predicts ANY valid category from your list of 29 (e.g., Red_HD, Green_PP)
#         if new_class_name in category_name_to_id:
#             ann["category_id"] = category_name_to_id[new_class_name]
#             stats["updated"] += 1
#         else:
#             # If it predicts something weird like "Purple_HD", skip it.
#             stats["err_class_not_in_coco"] += 1

# # =========================================================
# # SAVE FILE
# # =========================================================
# with open(OUTPUT_COCO, "w") as f:
#     json.dump(coco_updated, f, indent=2)

# print("\n===== DETAILED SUMMARY =====")
# print(f"✅ Successfully updated     : {stats['updated']}")
# print(f"❌ Skipped (Bad Filename)   : {stats['err_invalid_name']}")
# print(f"❌ Skipped (No Parent Img)  : {stats['err_missing_parent']}")
# print(f"❌ Skipped (Index Mismatch) : {stats['err_idx_out_of_bounds']}")
# print(f"❌ Skipped (AI Failed)      : {stats['err_no_ai_prediction']}")
# print(f"❌ Skipped (Class Not Found): {stats['err_class_not_in_coco']}")
# print(f"💾 Saved final JSON to      : {OUTPUT_COCO}")




import os
import json
import copy
from ultralytics import YOLO
from tqdm import tqdm

# =========================================================
# PATHS
# =========================================================
CROP_DIR = "/media/wi/ssd_hub/output_v8/cropped_by_label_polygon/yolo_object"
ORIGINAL_COCO = "/media/wi/ssd_hub/output_v8/coco_added_cats.json"
OUTPUT_COCO = "/media/wi/ssd_hub/output_v8/coco_updated.json"
MODEL_PATH = "/home/wi/Avinash_Works/waste-masknet/outputs/runs/yolov8m-seg_seg_20260217_123427/weights/best.pt"

# =========================================================
# INITIALIZATION
# =========================================================
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

print("Loading COCO file...")
with open(ORIGINAL_COCO) as f:
    coco_original = json.load(f)

coco_updated = copy.deepcopy(coco_original)

# =========================================================
# FAST LOOKUPS
# =========================================================
category_name_to_id = {cat["name"]: cat["id"] for cat in coco_updated["categories"]}
image_name_to_id = {img["file_name"]: img["id"] for img in coco_updated["images"]}
ann_id_to_ann = {ann["id"]: ann for ann in coco_updated["annotations"]}

# =========================================================
# TRACKERS
# =========================================================
stats = {
    "updated": 0,
    "err_invalid_name": 0,
    "err_missing_parent": 0,
    "err_missing_ann": 0,
    "err_no_ai_prediction": 0,
    "err_class_not_in_coco": 0
}

# =========================================================
# PROCESS CROPS
# =========================================================
for folder in ["Color_HD", "Color_PP"]:
    folder_path = os.path.join(CROP_DIR, folder)

    if not os.path.exists(folder_path):
        print(f"⚠️ Folder missing: {folder_path}")
        continue

    print(f"\nProcessing folder: {folder}")

    for crop_name in tqdm(os.listdir(folder_path)):

        name_no_ext, ext = os.path.splitext(crop_name)

        if "__ann" not in name_no_ext:
            stats["err_invalid_name"] += 1
            continue

        try:
            base, ann_str = name_no_ext.rsplit("__ann", 1)
            ann_id = int(ann_str)
        except ValueError:
            stats["err_invalid_name"] += 1
            continue

        parent_image_name = base

        if parent_image_name not in image_name_to_id:
            stats["err_missing_parent"] += 1
            continue

        ann = ann_id_to_ann.get(ann_id)
        if ann is None:
            stats["err_missing_ann"] += 1
            continue

        crop_path = os.path.join(folder_path, crop_name)

        # AI Inference
        result = model(crop_path, conf=0.25, verbose=False)[0]

        if result.boxes is None or len(result.boxes.cls) == 0:
            stats["err_no_ai_prediction"] += 1
            continue

        pred_id = int(result.boxes.cls[0].item())
        pred_class_name = result.names[pred_id].strip()  # already full e.g. "White_PP"

        # Direct match — no material appending needed
        if pred_class_name in category_name_to_id:
            ann["category_id"] = category_name_to_id[pred_class_name]
            stats["updated"] += 1
        else:
            stats["err_class_not_in_coco"] += 1

# =========================================================
# SAVE FILE
# =========================================================
with open(OUTPUT_COCO, "w") as f:
    json.dump(coco_updated, f, indent=2)

# =========================================================
# SUMMARY
# =========================================================
print("\n===== DETAILED SUMMARY =====")
print(f"✅ Successfully updated     : {stats['updated']}")
print(f"❌ Skipped (Bad Filename)   : {stats['err_invalid_name']}")
print(f"❌ Skipped (No Parent Img)  : {stats['err_missing_parent']}")
print(f"❌ Skipped (Missing Ann)    : {stats['err_missing_ann']}")
print(f"❌ Skipped (AI Failed)      : {stats['err_no_ai_prediction']}")
print(f"❌ Skipped (Class Not Found): {stats['err_class_not_in_coco']}")
print(f"💾 Saved final JSON to      : {OUTPUT_COCO}")