from __future__ import annotations
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("hitl.corrections")

_EMPTY: Dict[str, List] = {"edits": [], "deletions": [], "additions": []}


class CorrectionStore:
    """Thread-safe, append-only correction storage with atomic file writes."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.corrections_file = output_dir / "corrections.json"
        self.progress_file = output_dir / "progress.json"
        self._lock = threading.Lock()

        if not self.corrections_file.exists():
            self._atomic_write(self.corrections_file, dict(_EMPTY))
        if not self.progress_file.exists():
            self._atomic_write(self.progress_file, {"completed_batches": []})

        log.info("CorrectionStore ready at %s", output_dir)

    # ── internal helpers ────────────────────────────────────────────────────────

    def _atomic_write(self, path: Path, data: Any) -> None:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)

    def _read(self, path: Path, default: Any) -> Any:
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            log.warning("Could not read %s: %s — using default", path, e)
            return default

    # ── public read ─────────────────────────────────────────────────────────────

    def load(self) -> Dict:
        return self._read(self.corrections_file, dict(_EMPTY))

    def load_progress(self) -> Dict:
        return self._read(self.progress_file, {"completed_batches": []})

    # ── public write ─────────────────────────────────────────────────────────────

    def add_edit(self, annotation_id: int, new_category_id: int, attributes: dict = None) -> None:
        if attributes is None:
            attributes = {}
        with self._lock:
            data = self.load()
            # Deduplicate: replace any previous edit for same annotation
            data["edits"] = [
                e for e in data["edits"] if e["annotation_id"] != annotation_id
            ]
            data["edits"].append(
                {
                    "annotation_id": annotation_id,
                    "new_category_id": new_category_id,
                    "attributes": attributes,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            )
            self._atomic_write(self.corrections_file, data)

    def add_deletion(self, annotation_id: int) -> None:
        with self._lock:
            data = self.load()
            if annotation_id not in data["deletions"]:
                data["deletions"].append(annotation_id)
            # Remove any pending edit for deleted annotation
            data["edits"] = [
                e for e in data["edits"] if e["annotation_id"] != annotation_id
            ]
            self._atomic_write(self.corrections_file, data)

    def undo_deletion(self, annotation_id: int) -> None:
        with self._lock:
            data = self.load()
            data["deletions"] = [d for d in data["deletions"] if d != annotation_id]
            self._atomic_write(self.corrections_file, data)

    def add_annotation(
        self, image_id: int, segmentation: list, category_id: int, attributes: dict = None
    ) -> int:
        if attributes is None:
            attributes = {}
        with self._lock:
            data = self.load()
            # Assign a temporary negative ID to track additions
            temp_id = -(len(data["additions"]) + 1)
            data["additions"].append(
                {
                    "temp_id": temp_id,
                    "image_id": image_id,
                    "segmentation": segmentation,
                    "category_id": category_id,
                    "attributes": attributes,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            )
            self._atomic_write(self.corrections_file, data)
            return temp_id

    def mark_completed(self, batch_idx: int) -> None:
        with self._lock:
            progress = self.load_progress()
            completed = set(progress["completed_batches"])
            completed.add(batch_idx)
            progress["completed_batches"] = sorted(completed)
            progress["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._atomic_write(self.progress_file, progress)

    def mark_uncompleted(self, batch_idx: int) -> None:
        with self._lock:
            progress = self.load_progress()
            completed = set(progress["completed_batches"])
            completed.discard(batch_idx)
            progress["completed_batches"] = sorted(completed)
            self._atomic_write(self.progress_file, progress)
