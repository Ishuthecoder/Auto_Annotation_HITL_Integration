import json
import os

# === CONFIG ===
INPUT_COCO = "/media/wi/ssd_hub/output_v8/coco_all.json"
OUTPUT_COCO = "/media/wi/ssd_hub/output_v8/coco_added_cats.json"

# The 4 new categories you want to add
NEW_CATEGORIES = [
    "Green_HD", 
    "Black_HD", 
    "Blue_PP", 
    "Green_PP"
]

def add_categories_to_coco():
    if not os.path.exists(INPUT_COCO):
        print(f"❌ Error: File not found at {INPUT_COCO}")
        return

    # 1. Load the existing COCO JSON
    with open(INPUT_COCO, 'r') as f:
        coco_data = json.load(f)

    categories = coco_data.get('categories', [])
    
    # 2. Find the current highest category ID to avoid conflicts
    if len(categories) > 0:
        current_max_id = max(cat['id'] for cat in categories)
    else:
        current_max_id = -1

    # 3. Add the new categories
    next_id = current_max_id + 1
    added_count = 0

    # Get a list of existing category names to prevent duplicates
    existing_names = [cat['name'] for cat in categories]

    for new_name in NEW_CATEGORIES:
        if new_name in existing_names:
            print(f"⚠️ Skipping '{new_name}': Category already exists.")
            continue
            
        new_cat = {
            "id": next_id,
            "name": new_name,
            "supercategory": "None"
        }
        categories.append(new_cat)
        print(f"✅ Added: ID {next_id} | Name: {new_name}")
        
        next_id += 1
        added_count += 1

    # Update the dictionary
    coco_data['categories'] = categories

    # 4. Save the updated COCO JSON
    with open(OUTPUT_COCO, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"\n🎉 Successfully added {added_count} new categories.")
    print(f"💾 Saved updated file to: {OUTPUT_COCO}")

if __name__ == "__main__":
    add_categories_to_coco()