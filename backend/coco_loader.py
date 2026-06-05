from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Set

log = logging.getLogger("hitl.loader")


class CocoLoader:
    """Loads a COCO JSON and builds efficient in-memory indexes."""

    def __init__(self, coco_path: Path, batch_size: int = 1000):
        log.info("Loading COCO: %s", coco_path)
        with open(coco_path) as f:
            coco = json.load(f)

        self.batch_size = batch_size
        self.categories: List[Dict] = coco.get("categories", [])
        self.cat_ids: Set[int] = {c["id"] for c in self.categories}

        # image_id → image metadata
        self.image_by_id: Dict[int, Dict] = {
            img["id"]: img for img in coco.get("images", [])
        }

        # annotation indexes
        self.ann_by_id: Dict[int, Dict] = {}
        self.anns_by_image: Dict[int, List[Dict]] = {}
        for ann in coco.get("annotations", []):
            self.ann_by_id[ann["id"]] = ann
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.total_annotations = len(self.ann_by_id)

        # Only batch images that have at least one annotation
        annotated = sorted(
            iid for iid in self.image_by_id if self.anns_by_image.get(iid)
        )
        self.batches: List[List[int]] = [
            annotated[i : i + batch_size]
            for i in range(0, len(annotated), batch_size)
        ]

        log.info(
            "Indexed: %d images | %d annotations | %d categories | %d batches",
            len(self.image_by_id),
            self.total_annotations,
            len(self.categories),
            len(self.batches),
        )
