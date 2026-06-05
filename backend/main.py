#!/usr/bin/env python3
"""
HITL Annotation Review Server (FastAPI)

Usage:
    python backend/main.py \
        --coco       /path/to/coco.json \
        --images     /path/to/train /path/to/val \
        --output_dir /path/to/output \
        --batch_size 1000 \
        --port       8000
"""
from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from coco_loader import CocoLoader
from corrections import CorrectionStore
from image_server import ImageServer
from models import AddRequest, DeleteRequest, EditRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hitl")

# ── Global state (populated before uvicorn starts) ─────────────────────────────
_cfg: Dict[str, Any] = {}
loader: Optional[CocoLoader] = None
store: Optional[CorrectionStore] = None
imgserver: Optional[ImageServer] = None


# ── App startup / shutdown ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global loader, store, imgserver
    loader = CocoLoader(Path(_cfg["coco"]), _cfg["batch_size"])
    store = CorrectionStore(Path(_cfg["output_dir"]))
    imgserver = ImageServer([Path(p) for p in _cfg["images"]])
    log.info(
        "Server ready — %d images | %d annotations | %d batches",
        len(loader.image_by_id),
        loader.total_annotations,
        len(loader.batches),
    )
    yield
    log.info("Shutdown.")


app = FastAPI(title="HITL Annotation Review", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _build_annotation_list(image_id: int) -> List[Dict]:
    """Merge base annotations with applied corrections."""
    base = loader.anns_by_image.get(image_id, [])
    corr = store.load()

    deleted = set(corr.get("deletions", []))
    edits = {
        e["annotation_id"]: {"category_id": e["new_category_id"], "attributes": e.get("attributes", {})}
        for e in corr.get("edits", [])
    }
    additions = [a for a in corr.get("additions", []) if a["image_id"] == image_id]

    result = []
    for ann in base:
        if ann["id"] in deleted:
            continue
        
        edit_data = edits.get(ann["id"], {})
        final_cat = edit_data.get("category_id", ann["category_id"])
        final_attrs = edit_data.get("attributes", ann.get("attributes", {}))

        entry = {
            "id":           ann["id"],
            "category_id":  final_cat,
            "segmentation": ann.get("segmentation", []),
            "bbox":         ann.get("bbox", []),
            "attributes":   final_attrs,
            "corrected":    ann["id"] in edits,
            "added":        False,
        }
        result.append(entry)

    for a in additions:
        temp_id = a["temp_id"]
        if temp_id in deleted:          # skip deleted additions too
            continue
        edit_data = edits.get(temp_id, {})
        final_cat = edit_data.get("category_id", a["category_id"])
        final_attrs = edit_data.get("attributes", a.get("attributes", {}))

        result.append({
            "id":           temp_id,
            "category_id":  final_cat,
            "segmentation": a["segmentation"],
            "bbox":         [],
            "attributes":   final_attrs,
            "corrected":    temp_id in edits,
            "added":        True,
        })

    return result


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/api/info")
def api_info():
    progress = store.load_progress()
    corr = store.load()
    return {
        "total_images":      len(loader.image_by_id),
        "total_annotations": loader.total_annotations,
        "total_batches":     len(loader.batches),
        "batch_size":        loader.batch_size,
        "completed_batches": sorted(set(progress.get("completed_batches", []))),
        "total_edits":       len(corr.get("edits", [])),
        "total_deletions":   len(corr.get("deletions", [])),
        "total_additions":   len(corr.get("additions", [])),
        "categories":        loader.categories,
    }


@app.get("/api/batch/{batch_idx}")
def api_batch(batch_idx: int):
    if batch_idx < 0 or batch_idx >= len(loader.batches):
        raise HTTPException(404, "Invalid batch index")
    progress = store.load_progress()
    completed = set(progress.get("completed_batches", []))
    items = []
    for image_id in loader.batches[batch_idx]:
        img = loader.image_by_id.get(image_id)
        if not img:
            continue
        items.append({
            "image_id":    image_id,
            "file_name":   img["file_name"],
            "width":       img.get("width", 1920),
            "height":      img.get("height", 1080),
            "annotations": _build_annotation_list(image_id),
        })
    return {
        "batch_idx":     batch_idx,
        "total_batches": len(loader.batches),
        "completed":     batch_idx in completed,
        "items":         items,
    }


@app.get("/api/image/{image_id}")
def api_image(image_id: int):
    img = loader.image_by_id.get(image_id)
    if not img:
        raise HTTPException(404, "Image not in COCO")
    data, mime = imgserver.serve(img["file_name"])
    if data is None:
        raise HTTPException(404, f"File not found on disk: {img['file_name']}")
    return Response(content=data, media_type=mime)


@app.post("/api/edit")
def api_edit(req: EditRequest):
    if req.annotation_id >= 0 and req.annotation_id not in loader.ann_by_id:
        raise HTTPException(404, f"Annotation {req.annotation_id} not found")
    if req.new_category_id not in loader.cat_ids:
        raise HTTPException(400, f"Category {req.new_category_id} does not exist")
    store.add_edit(req.annotation_id, req.new_category_id, req.attributes)
    return {"status": "ok", "annotation_id": req.annotation_id}


@app.post("/api/delete")
def api_delete(req: DeleteRequest):
    store.add_deletion(req.annotation_id)
    return {"status": "ok", "annotation_id": req.annotation_id}


@app.post("/api/undo_delete")
def api_undo_delete(req: DeleteRequest):
    store.undo_deletion(req.annotation_id)
    return {"status": "ok", "annotation_id": req.annotation_id}


@app.post("/api/add")
def api_add(req: AddRequest):
    if req.category_id not in loader.cat_ids:
        raise HTTPException(400, f"Category {req.category_id} does not exist")
    if req.image_id not in loader.image_by_id:
        raise HTTPException(404, f"Image {req.image_id} not found")
    temp_id = store.add_annotation(req.image_id, req.segmentation, req.category_id, req.attributes)
    return {"status": "ok", "temp_id": temp_id}


@app.post("/api/approve/{batch_idx}")
def api_approve(batch_idx: int):
    if batch_idx < 0 or batch_idx >= len(loader.batches):
        raise HTTPException(404, "Invalid batch index")
    store.mark_completed(batch_idx)
    return {"status": "approved", "batch_idx": batch_idx}


@app.post("/api/unapprove/{batch_idx}")
def api_unapprove(batch_idx: int):
    store.mark_uncompleted(batch_idx)
    return {"status": "unapproved", "batch_idx": batch_idx}


@app.get("/api/debug/stem")
def api_debug_stem():
    sample = [img["file_name"] for img in list(loader.image_by_id.values())[:10]]
    resolved = [
        {"file_name": fn, "found": imgserver.find(fn) is not None}
        for fn in sample[:5]
    ]
    return {
        "stem_index_size": len(imgserver.stem_index),
        "image_dirs":      [str(d) for d in imgserver.dirs],
        "sample_filenames": sample,
        "resolved":        resolved,
    }


# ── Entry point ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HITL Annotation Review Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--coco",       required=True, help="Path to COCO JSON file")
    parser.add_argument("--images",     required=True, nargs="+",
                        help="One or more image directories (train, val, etc.)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to store corrections.json and progress.json")
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--port",       type=int, default=8001)
    parser.add_argument("--host",       default="0.0.0.0")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    _cfg.update(vars(args))

    log.info("Starting HITL server → http://%s:%d", args.host, args.port)
    log.info("COCO:       %s", args.coco)
    log.info("Images:     %s", args.images)
    log.info("Output dir: %s", args.output_dir)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
