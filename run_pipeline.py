#!/usr/bin/env python3
"""
run_pipeline.py — Master Orchestrator
======================================
Runs the full Auto-Annotation pipeline (Steps 1-4), then launches the
HITL review server with the generated COCO file.

Usage
-----
  python hitl_review/run_pipeline.py \\
    --image_dir      /path/to/images \\
    --shape_model    /path/to/segmentation_model.pt \\
    --color_model    /path/to/color_model.pt \\
    --output_dir     /path/to/pipeline_output \\
    --batch_size     2000 \\
    --port           8000

Timestamp filtering — any date/time range is supported:
    --start_time "2026-01-15 09:00:00"   --end_time "2026-03-31 23:59:59"
    --start_time "2026-01-15"            --end_time "2026-01-15"
    --start_time "20260115_090000"       --end_time "20260115_180000"
    --start_time "09:00:00"             --end_time "18:00:00"

  The filter checks:
    1. File modification time (mtime)   — always available, primary source
    2. Timestamp embedded in filename   — secondary, if a pattern is found
  Omit both flags to process all images.

Resume from a step (skip already-done work):
    --start_step 3       (choices: 1-4, default 1)

Skip HITL auto-launch:
    --no_launch
"""

# python3 run_pipeline.py \
#   --image_dir   /media/wi/ssd_hub/Avinash_work/dataset/images/train
#   --shape_model /media/wi/ssd_hub/training_runs/yolov8m_seg_real_only_20260423_175631/weights/best.pt \
#   --color_model /media/wi/ssd_hub/Ishika_works/training_runs/yolov8m_color_seg/weights/best.pt \
#   --output_dir  /media/wi/ssd_hub/Ishika_works/pipeline_outputs/output/pipeline_output \
#   --start_time  "2026-03-15 09:00:00" \
#   --end_time    "2026-03-31 18:00:00"

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Timestamp filtering helpers
# ─────────────────────────────────────────────────────────────────────────────

# Every format tried when parsing a user-supplied timestamp string.
# Formats are tried in order; the first match wins.
_USER_DT_FORMATS = [
    "%Y-%m-%d %H:%M:%S",   # 2026-01-15 09:30:00
    "%Y-%m-%dT%H:%M:%S",   # 2026-01-15T09:30:00
    "%Y-%m-%d %H:%M",      # 2026-01-15 09:30
    "%Y-%m-%d",            # 2026-01-15   (time = 00:00:00)
    "%Y%m%d_%H%M%S",       # 20260115_093000
    "%Y%m%d_%H%M",         # 20260115_0930
    "%Y%m%d",              # 20260115     (time = 00:00:00)
    "%d/%m/%Y %H:%M:%S",   # 15/01/2026 09:30:00
    "%d/%m/%Y %H:%M",      # 15/01/2026 09:30
    "%d/%m/%Y",            # 15/01/2026
    "%d-%m-%Y %H:%M:%S",   # 15-01-2026 09:30:00
    "%d-%m-%Y",            # 15-01-2026
    "%H:%M:%S",            # 09:30:00   (date = 1970-01-01, time only)
    "%H:%M",               # 09:30      (date = 1970-01-01, time only)
    "%H%M%S",              # 093000     (date = 1970-01-01, time only)
]


def _parse_user_dt(s: str) -> datetime:
    """
    Parse a user-supplied date/time string.
    Accepts many common formats so users can type whatever feels natural.
    Raises ValueError with a clear message listing all accepted formats.
    """
    s = s.strip()
    for fmt in _USER_DT_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue

    accepted = "\n".join(f"  {fmt}" for fmt in _USER_DT_FORMATS)
    raise ValueError(
        f"Cannot parse timestamp: '{s}'\n"
        f"Accepted formats:\n{accepted}\n"
        f"Examples: '2026-01-15 09:30:00'  '2026-01-15'  '20260115_093000'  '09:30:00'"
    )


# Regex patterns for extracting a datetime from an image filename.
# Primary timestamp source is always file mtime (see filter function);
# this is a secondary bonus that works when the filename contains a stamp.
_FNAME_TS_PATTERNS = [
    # YYYYMMDD_HHMMSS  e.g.  frame_20260115_093045.jpg
    (re.compile(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})"),
     lambda g: datetime(int(g[0]), int(g[1]), int(g[2]),
                        int(g[3]), int(g[4]), int(g[5]))),
    # YYYYMMDD  e.g.  capture_20260115_001.jpg
    (re.compile(r"(\d{4})(\d{2})(\d{2})"),
     lambda g: datetime(int(g[0]), int(g[1]), int(g[2]))),
]


def _dt_from_filename(fname: str) -> Optional[datetime]:
    """Try to extract a datetime from a filename. Returns None if not found."""
    for pattern, builder in _FNAME_TS_PATTERNS:
        m = pattern.search(fname)
        if m:
            try:
                return builder(m.groups())
            except ValueError:
                continue
    return None


def filter_images_by_timestamp(
    image_dir: str,
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
) -> Optional[List[str]]:
    """
    Select image filenames whose timestamp falls within [start_dt, end_dt].

    Timestamp resolution order per file:
      1. File modification time (mtime) — always available, always used first.
      2. Timestamp found in the filename — used only when mtime is epoch (1970),
         which can happen on some filesystems.

    Returns:
      - None  →  no filtering requested; the pipeline processes all images.
      - List  →  the filenames (in sorted order) that passed the filter.
    """
    if start_dt is None and end_dt is None:
        log.info("No timestamp filter — all images in %s will be processed.", image_dir)
        return None

    exts = (".jpg", ".jpeg", ".png")
    all_files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(exts))
    selected: List[str] = []
    skipped_before = 0
    skipped_after  = 0

    for fname in all_files:
        full_path = os.path.join(image_dir, fname)

        # Primary: file modification time
        mtime = os.path.getmtime(full_path)
        dt = datetime.fromtimestamp(mtime)

        # Secondary: if mtime looks like epoch (unlikely but possible on some
        # network/mounted filesystems), fall back to filename pattern.
        if dt.year == 1970:
            fname_dt = _dt_from_filename(fname)
            if fname_dt is not None:
                dt = fname_dt

        if start_dt and dt < start_dt:
            skipped_before += 1
            continue
        if end_dt and dt > end_dt:
            skipped_after += 1
            continue

        selected.append(fname)

    log.info(
        "Timestamp filter  start=%s  end=%s",
        start_dt.strftime("%Y-%m-%d %H:%M:%S") if start_dt else "(none)",
        end_dt.strftime("%Y-%m-%d %H:%M:%S")   if end_dt   else "(none)",
    )
    log.info(
        "  Selected: %d / %d  (before range: %d  |  after range: %d)",
        len(selected), len(all_files), skipped_before, skipped_after,
    )
    if not selected:
        log.warning(
            "  ⚠️  No images matched the timestamp range. "
            "Check --start_time / --end_time values and image directory."
        )
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> str:
    """
    Execute Steps 1-4 and return the path to coco_updated.json.
    """
    # Late import so the module loads fast even if ultralytics isn't installed.
    from auto_annotation.pipeline_steps import (
        step1_run_yolo_segmentation,
        step2_add_categories,
        step3_generate_crops,
        step4_run_color_classification,
    )

    output_dir: str = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Timestamp filtering ──────────────────────────────────────────────────
    start_dt = _parse_user_dt(args.start_time) if args.start_time else None
    end_dt   = _parse_user_dt(args.end_time)   if args.end_time   else None
    image_files = filter_images_by_timestamp(args.image_dir, start_dt, end_dt)

    # ── Parse class-folder mapping  e.g. "3:Color_HD,12:Color_PP" ───────────
    class_id_to_folder: Dict[int, str] = {}
    valid_class_ids: Set[int] = set()
    for token in args.class_folder_map.split(","):
        token = token.strip()
        if not token:
            continue
        cid_str, _, folder = token.partition(":")
        cid = int(cid_str.strip())
        class_id_to_folder[cid] = folder.strip()
        valid_class_ids.add(cid)

    extra_categories: List[str] = [
        c.strip() for c in args.extra_categories.split(",") if c.strip()
    ]

    log.info("=" * 60)
    log.info("Class → Crop-folder mapping (Step 3 / Step 4 only):")
    for cid, folder in sorted(class_id_to_folder.items()):
        log.info("  class_id=%d  →  crops/%s", cid, folder)
    log.info("NOTE: Step 1 segments ALL classes so every object appears in HITL.")
    log.info("NOTE: Run Step 1 first, then check logs for the actual class IDs")
    log.info("      and update --class_folder_map if any ID is wrong.")
    log.info("=" * 60)

    # ── File paths carried between steps ────────────────────────────────────
    coco_all_path        = os.path.join(output_dir, "coco_all.json")
    coco_added_cats_path = os.path.join(output_dir, "coco_added_cats.json")
    coco_updated_path    = os.path.join(output_dir, "coco_updated.json")

    start_step: int = args.start_step

    # ── STEP 1 ───────────────────────────────────────────────────────────────
    # valid_class_ids is intentionally NOT passed here — Step 1 always
    # segments ALL objects so every detection reaches the HITL review UI
    # with its correct model label.  Class filtering is only for crop
    # generation (Step 3) and colour classification (Step 4).
    if start_step <= 1:
        coco_all_path = step1_run_yolo_segmentation(
            image_dir      = args.image_dir,
            model_path     = args.shape_model,
            output_dir     = output_dir,
            conf           = args.shape_conf,
            min_area       = args.min_area,
            valid_class_ids= None,   # ALL classes → HITL
            image_files    = image_files,
        )
    else:
        log.info("Skipping Step 1 — using existing %s", coco_all_path)
        if not os.path.exists(coco_all_path):
            sys.exit(f"❌ Expected {coco_all_path} but not found. Lower --start_step.")

    # ── STEP 2 ───────────────────────────────────────────────────────────────
    if start_step <= 2:
        coco_added_cats_path = step2_add_categories(
            coco_path      = coco_all_path,
            new_categories = extra_categories,
            output_dir     = output_dir,
        )
    else:
        log.info("Skipping Step 2 — using existing %s", coco_added_cats_path)
        if not os.path.exists(coco_added_cats_path):
            sys.exit(f"❌ Expected {coco_added_cats_path} but not found. Lower --start_step.")

    # ── STEP 3 ───────────────────────────────────────────────────────────────
    crop_dir = os.path.join(output_dir, "crops")
    if start_step <= 3:
        crop_dir = step3_generate_crops(
            image_dir         = args.image_dir,
            coco_path         = coco_added_cats_path,
            output_dir        = output_dir,
            valid_class_ids   = valid_class_ids,
            class_id_to_folder= class_id_to_folder,
        )
    else:
        log.info("Skipping Step 3 — using existing crops at %s", crop_dir)
        if not os.path.exists(crop_dir):
            sys.exit(f"❌ Expected {crop_dir} but not found. Lower --start_step.")

    # ── STEP 4 ───────────────────────────────────────────────────────────────
    if start_step <= 4:
        crop_folders = list(class_id_to_folder.values()) or None
        coco_updated_path = step4_run_color_classification(
            crop_dir     = crop_dir,
            coco_path    = coco_added_cats_path,
            model_path   = args.color_model,
            output_dir   = output_dir,
            conf         = args.color_conf,
            crop_folders = crop_folders,
        )
    else:
        log.info("Skipping Step 4 — using existing %s", coco_updated_path)
        if not os.path.exists(coco_updated_path):
            sys.exit(f"❌ Expected {coco_updated_path} but not found. Lower --start_step.")

    return coco_updated_path


# ─────────────────────────────────────────────────────────────────────────────
# HITL launcher
# ─────────────────────────────────────────────────────────────────────────────

def launch_hitl(args: argparse.Namespace, coco_updated_path: str) -> None:
    """Start the HITL FastAPI backend as a subprocess (blocks until Ctrl+C)."""
    hitl_output = os.path.join(args.output_dir, "hitl_corrections")
    os.makedirs(hitl_output, exist_ok=True)

    backend_main = Path(__file__).parent / "backend" / "main.py"

    cmd = [
        sys.executable,
        str(backend_main),
        "--coco",       coco_updated_path,
        "--images",     args.image_dir,
        "--output_dir", hitl_output,
        "--batch_size", str(args.batch_size),
        "--port",       str(args.port),
        "--host",       args.host,
    ]

    log.info("=" * 60)
    log.info("Launching HITL Review Server")
    log.info("  COCO file  : %s", coco_updated_path)
    log.info("  Images dir : %s", args.image_dir)
    log.info("  Batch size : %d", args.batch_size)
    log.info("  URL        : http://%s:%d", args.host, args.port)
    log.info("  Corrections: %s", hitl_output)
    log.info("  (Press Ctrl+C to stop the server)")
    log.info("=" * 60)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        log.info("HITL server stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Auto-Annotation → HITL Integrated Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required paths ───────────────────────────────────────────────────────
    p.add_argument("--image_dir",   required=True,
                   help="Directory containing source images (frames).")
    p.add_argument("--shape_model", required=True,
                   help="YOLOv8 segmentation model (.pt) for Step 1.")
    p.add_argument("--color_model", required=True,
                   help="YOLOv8 color-classification model (.pt) for Step 4.")
    p.add_argument("--output_dir",  required=True,
                   help="Root output directory for all pipeline artefacts.")

    # ── Timestamp filtering ───────────────────────────────────────────────────
    p.add_argument(
        "--start_time", default=None,
        metavar="DATETIME",
        help=(
            "Start of the image time range (inclusive). "
            "Any date/time format is accepted, e.g.:\n"
            "  '2026-01-15 09:30:00'  (YYYY-MM-DD HH:MM:SS)\n"
            "  '2026-01-15'           (date only, time=00:00:00)\n"
            "  '20260115_093000'      (YYYYMMDD_HHMMSS)\n"
            "  '15/01/2026 09:30'     (DD/MM/YYYY HH:MM)\n"
            "  '09:30:00'             (time only, any date)\n"
            "Timestamp is read from file mtime (primary) or filename (secondary). "
            "Omit to include all images."
        ),
    )
    p.add_argument(
        "--end_time", default=None,
        metavar="DATETIME",
        help=(
            "End of the image time range (inclusive). "
            "Same flexible format as --start_time. "
            "Omit to include all images up to the latest."
        ),
    )

    # ── Class / category config ───────────────────────────────────────────────
    p.add_argument(
        "--class_folder_map",
        default="3:Color_HD,12:Color_PP",
        help=(
            "Comma-separated class_id:folder_name pairs for CROP GENERATION ONLY "
            "(Steps 3 & 4). Does NOT affect which objects appear in HITL — Step 1 "
            "always segments all classes. "
            "E.g. '3:Color_HD,12:Color_PP' means YOLO class 3 → Color_HD folder "
            "and YOLO class 12 → Color_PP folder. "
            "IMPORTANT: verify the actual YOLO class IDs in the Step 1 log output "
            "('Per-class annotation counts') and adjust these IDs to match your model. "
            "A wrong class ID is the most common cause of a missing Color_PP folder."
        ),
    )
    p.add_argument(
        "--extra_categories",
        default="Green_HD,Black_HD,Blue_PP,Green_PP",
        help="Comma-separated category names to add in Step 2.",
    )

    # ── YOLO thresholds ───────────────────────────────────────────────────────
    p.add_argument("--shape_conf", type=float, default=0.20,
                   help="Confidence threshold for the segmentation model (Step 1).")
    p.add_argument("--color_conf", type=float, default=0.25,
                   help="Confidence threshold for the color model (Step 4).")
    p.add_argument("--min_area",   type=float, default=200.0,
                   help="Minimum polygon area (px²) — smaller detections are dropped.")

    # ── Pipeline control ──────────────────────────────────────────────────────
    p.add_argument("--start_step", type=int, default=1, choices=[1, 2, 3, 4],
                   help="Resume pipeline from this step (1=start from scratch).")

    # ── HITL server ───────────────────────────────────────────────────────────
    p.add_argument("--batch_size", type=int, default=2000,
                   help="Number of images per HITL review batch.")
    p.add_argument("--port",       type=int, default=8000,
                   help="Port for the HITL FastAPI server.")
    p.add_argument("--host",       default="0.0.0.0",
                   help="Host for the HITL FastAPI server.")
    p.add_argument("--no_launch",  action="store_true",
                   help="Run the annotation pipeline only — do not start HITL server.")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Auto-Annotation → HITL Pipeline")
    log.info("  image_dir   : %s", args.image_dir)
    log.info("  shape_model : %s", args.shape_model)
    log.info("  color_model : %s", args.color_model)
    log.info("  output_dir  : %s", args.output_dir)
    log.info("  start_time  : %s", args.start_time or "(all images)")
    log.info("  end_time    : %s", args.end_time   or "(all images)")
    log.info("  batch_size  : %d", args.batch_size)
    log.info("=" * 60)

    # ── Run Steps 1-4 ────────────────────────────────────────────────────────
    coco_updated_path = run_pipeline(args)

    log.info("")
    log.info("Pipeline complete!")
    log.info("HITL input COCO → %s", coco_updated_path)

    if args.no_launch:
        log.info("")
        log.info("To start the HITL server, run:")
        log.info(
            "  python backend/main.py \\\n"
            "    --coco       %s \\\n"
            "    --images     %s \\\n"
            "    --output_dir %s \\\n"
            "    --batch_size %d \\\n"
            "    --port       %d",
            coco_updated_path,
            args.image_dir,
            os.path.join(args.output_dir, "hitl_corrections"),
            args.batch_size,
            args.port,
        )
    else:
        launch_hitl(args, coco_updated_path)


if __name__ == "__main__":
    main()
