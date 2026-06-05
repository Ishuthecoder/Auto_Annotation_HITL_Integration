#!/usr/bin/env python3
"""
merge_corrections.py — Standalone script to merge corrections into COCO JSON.

Usage:
    python merge_corrections.py \
        --coco    /path/to/original_coco.json \
        --corr    /path/to/output/corrections.json \
        --output  /path/to/coco_corrected.json
"""
import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("merge")


def merge(coco_path: Path, corr_path: Path, output_path: Path) -> None:
    log.info("Loading COCO: %s", coco_path)
    with open(coco_path) as f:
        coco = json.load(f)

    log.info("Loading corrections: %s", corr_path)
    with open(corr_path) as f:
        corr = json.load(f)

    edits     = {e["annotation_id"]: e["new_category_id"] for e in corr.get("edits", [])}
    deletions = set(corr.get("deletions", []))
    additions = corr.get("additions", [])

    log.info(
        "Applying: %d edits | %d deletions | %d additions",
        len(edits), len(deletions), len(additions),
    )

    # Validate category IDs in edits
    valid_cats = {c["id"] for c in coco.get("categories", [])}
    bad_edits = [cat for cat in edits.values() if cat not in valid_cats]
    if bad_edits:
        log.warning("Unknown category IDs in edits: %s", bad_edits)

    # Apply edits + deletions
    new_anns = []
    n_edited = n_deleted = 0
    for ann in coco.get("annotations", []):
        if ann["id"] in deletions:
            n_deleted += 1
            continue
        if ann["id"] in edits:
            ann = dict(ann)
            ann["category_id"] = edits[ann["id"]]
            n_edited += 1
        new_anns.append(ann)

    # Assign new IDs for additions (start after current max)
    max_id = max((a["id"] for a in new_anns), default=0)
    image_ids = {img["id"] for img in coco.get("images", [])}
    n_added = 0
    for add in additions:
        if add["image_id"] not in image_ids:
            log.warning("Skipping addition: image_id=%d not in COCO", add["image_id"])
            continue
        max_id += 1
        new_anns.append({
            "id":           max_id,
            "image_id":     add["image_id"],
            "category_id":  add["category_id"],
            "segmentation": add["segmentation"],
            "bbox":         [],
            "area":         0,
            "iscrowd":      0,
        })
        n_added += 1

    coco["annotations"] = new_anns

    # Atomic write
    tmp = output_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(coco, f, ensure_ascii=False)
    tmp.replace(output_path)

    log.info(
        "Done — edited: %d | deleted: %d | added: %d | total anns: %d",
        n_edited, n_deleted, n_added, len(new_anns),
    )
    log.info("Output: %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge HITL corrections into COCO JSON")
    parser.add_argument("--coco",   required=True, help="Original COCO JSON")
    parser.add_argument("--corr",   required=True, help="corrections.json from output dir")
    parser.add_argument("--output", required=True, help="Output path for merged COCO JSON")
    args = parser.parse_args()
    merge(Path(args.coco), Path(args.corr), Path(args.output))
