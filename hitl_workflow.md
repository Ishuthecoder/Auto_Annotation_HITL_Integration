# HITL Pipeline — Complete Integration Workflow

## Big Picture

```
Raw Images
    │
    ▼
[run_pipeline.py]  ← single command that drives everything
    │
    ├─ STEP 1 → coco_all.json          (YOLO segmentation)
    ├─ STEP 2 → coco_added_cats.json   (category expansion)
    ├─ STEP 3 → crops/                 (object crop images)
    ├─ STEP 4 → coco_updated.json      (color classification)
    │
    ▼
[backend/main.py]  ← FastAPI server (auto-launched)
    │
    ▼
[frontend/]        ← React + Vite review UI
    │
    ▼
hitl_corrections/
    ├─ corrections.json   (all human edits)
    └─ progress.json      (batch approval state)
```

---

## PHASE 1 — Auto-Annotation Pipeline (Steps 1–4)

### Entry Point

```bash
python hitl_review/run_pipeline.py \
  --image_dir   /path/to/images \
  --shape_model /path/to/segmentation.pt \
  --color_model /path/to/color_model.pt \
  --output_dir  /path/to/pipeline_output \
  --batch_size  2000 \
  --port        8000
```

---

### STEP 0 — Timestamp Filtering (pre-step)

**File:** `run_pipeline.py` → `filter_images_by_timestamp()`

- Optional — activated via `--start_time` / `--end_time`
- Primary timestamp: **file mtime**. Falls back to filename pattern (e.g. `frame_20260115_093045.jpg`) if mtime is epoch
- Returns a filtered filename list → passed as `image_files` to Step 1
- If omitted → **all images** in `--image_dir` are processed

> [!TIP]
> Flexible formats accepted: `"2026-01-15 09:30:00"`, `"2026-01-15"`, `"20260115_093000"`, `"09:30:00"`

---

### STEP 1 — YOLOv8 Segmentation → `coco_all.json`

**File:** `auto_annotation/pipeline_steps.py` → `step1_run_yolo_segmentation()`

**What it does:**
1. Loads the **segmentation YOLO model** (`.pt` weights)
2. Runs inference on every image (confidence threshold: `--shape_conf`, default `0.20`)
3. For each detected object:
   - Extracts the **polygon mask** (`.xy` from YOLO result masks)
   - Computes polygon area via Shoelace formula
   - **Drops polygons smaller than `--min_area`** (default 200 px²)
4. **ALL detected classes are kept** — no class filtering here
5. Writes everything to COCO format

**Output:** `<output_dir>/coco_all.json`

```
coco_all.json
├── images:       [{id, file_name, width, height}, ...]
├── annotations:  [{id, image_id, category_id, segmentation, area}, ...]
└── categories:   [{id, name}, ...]   ← from YOLO model.names
```

> [!IMPORTANT]
> After Step 1, check the log for **"Per-class annotation counts"**.
> It shows the actual YOLO class IDs (e.g. `class 3 = Color_HD`).
> You must match these IDs in `--class_folder_map` for Steps 3 & 4 to work.

---

### STEP 2 — Add Fine-Grained Categories → `coco_added_cats.json`

**File:** `pipeline_steps.py` → `step2_add_categories()`

**What it does:**
1. Opens `coco_all.json`
2. Appends extra category names from `--extra_categories` (default: `Green_HD, Black_HD, Blue_PP, Green_PP`)
3. Skips duplicates — no double insertion
4. Assigns new sequential IDs

**Output:** `<output_dir>/coco_added_cats.json`

These extra categories are the **target labels** the color classifier predicts. Without them, Step 4 predictions can't be mapped to valid COCO category IDs.

---

### STEP 3 — Polygon-Tight Crop Generation → `crops/`

**File:** `pipeline_steps.py` → `step3_generate_crops()`

**What it does:**
1. Reads `coco_added_cats.json`
2. For each annotation whose `category_id` is in `--class_folder_map`:
   - Reads the source image from disk
   - Computes the tight bounding box of the polygon
   - Saves a **cropped image** of just that object
3. Crop filename format: `<image_stem>__ann<ann_id>.jpg`
   - e.g. `frame_001__ann452.jpg` → annotation ID 452
4. Organized into subfolders: `crops/Color_HD/`, `crops/Color_PP/`, etc.

**Output:** `<output_dir>/crops/<folder>/<stem>__ann<id>.jpg`

> [!NOTE]
> The `__ann<id>` suffix is the link between crop image and COCO annotation.
> Step 4 parses this ID to know which annotation to update.

> [!WARNING]
> **ZERO crops saved?** → Your `--class_folder_map` class IDs don't match the actual YOLO class IDs from Step 1. Check the Step 1 "Per-class annotation counts" log.

---

### STEP 4 — Color Classification → `coco_updated.json`

**File:** `pipeline_steps.py` → `step4_run_color_classification()`

**What it does:**
1. Loads the **color classifier YOLO model**
2. Iterates over all crop images in `crops/<folder>/`
3. For each crop:
   - Parses `ann_id` from filename
   - Looks up that annotation in the COCO JSON
   - Runs color classifier on the crop
   - Maps predicted class name → COCO `category_id`
   - **Overwrites** `category_id` with the color prediction
4. Writes final result to `coco_updated.json`

**Output:** `<output_dir>/coco_updated.json` ← **this is the HITL input file**

**Step 4 logged stats:**

| Stat | Meaning |
|---|---|
| `updated` | Annotation successfully re-labeled by color |
| `err_no_prediction` | Color model gave no confident detection |
| `err_class_not_found` | Predicted class not in COCO categories |
| `err_missing_ann` | Crop's ann_id not found in COCO |
| `err_invalid_name` | Crop filename not in `__ann<id>` format |

---

## PHASE 2 — HITL Review Server

### STEP 5 — FastAPI Backend

**File:** `backend/main.py`

Auto-launched by `run_pipeline.py` after Step 4. Can be skipped with `--no_launch` and run manually.

**On startup, 3 services initialize:**
- **`CocoLoader`** — loads `coco_updated.json`, partitions images into batches of `--batch_size`
- **`CorrectionStore`** — sets up `corrections.json` and `progress.json` in `hitl_corrections/`
- **`ImageServer`** — builds filename→disk-path index from all `--images` directories

**API Routes:**

| Route | Purpose |
|---|---|
| `GET /api/info` | Total images, annotations, batches, completion stats |
| `GET /api/batch/{idx}` | Fetch one batch's images + merged annotations |
| `GET /api/image/{id}` | Stream image file bytes to browser |
| `POST /api/edit` | Change annotation category + attributes |
| `POST /api/delete` | Mark annotation as deleted |
| `POST /api/undo_delete` | Restore a deleted annotation |
| `POST /api/add` | Save a new manually-drawn polygon |
| `POST /api/approve/{idx}` | Mark batch complete |
| `POST /api/unapprove/{idx}` | Re-open a completed batch |

**How `GET /api/batch` builds the annotation list (`_build_annotation_list`):**

```
coco_updated.json  (base truth — never modified)
         +
corrections.json   (human edits overlay)
         ↓
  1. Filter out deleted annotation IDs
  2. Apply category/attribute edits
  3. Filter out deleted added-annotations (temp_id)
  4. Append remaining newly-added annotations
         ↓
  final merged list → sent to frontend
```

---

### STEP 6 — Corrections Persistence

**File:** `backend/corrections.py` → `CorrectionStore`

All human changes are stored in `hitl_corrections/corrections.json`:

```json
{
  "edits":     [{ "annotation_id": 42, "new_category_id": 7, "attributes": {...} }],
  "deletions": [55, 102, 340],
  "additions": [{ "temp_id": -1, "image_id": 12, "segmentation": [[...]], ... }]
}
```

Key properties:
- **Atomic writes** — written to `.tmp` first, then renamed → no corruption on crash
- **Thread-safe** — protected by a threading lock
- **Edit deduplication** — re-editing the same annotation replaces the old edit entry

`progress.json` tracks approved batches:
```json
{ "completed_batches": [0, 1, 4] }
```

---

## PHASE 3 — Frontend Review UI

**Stack:** React + Vite (dev server port 5173, Vite proxies API calls to backend port 8000)

### Component Tree

```
App.jsx
  ├─ BatchList         ← left sidebar: all batches + completion indicators
  └─ ReviewPanel       ← main workspace
       ├─ AnnotationCanvas   ← Konva canvas: image + polygon overlays
       └─ AnnotationPanel    ← right sidebar: selected annotation editor + list
```

### User Actions → API Calls

| Action | Frontend behavior | Backend call |
|---|---|---|
| Select batch | Fetch and display all images | `GET /api/batch/{idx}` |
| Click polygon on canvas | Highlight + show in sidebar | Local state only |
| Change category dropdown | Save immediately | `POST /api/edit` |
| Change Material / Color | Save immediately | `POST /api/edit` |
| 🗑 Delete button (sidebar) | Hide polygon instantly → persist | `POST /api/delete` |
| `Delete` / `Backspace` key | Same as delete button | `POST /api/delete` |
| ✕ button on list row | Same as delete button | `POST /api/delete` |
| Draw polygon (finish) | Submit new polygon | `POST /api/add` |
| ✅ Approve Batch | Mark batch done | `POST /api/approve/{idx}` |
| Re-open | Un-mark batch | `POST /api/unapprove/{idx}` |

### Polygon Drawing Flow

1. Click **"+ Draw Polygon"** → draw mode on (crosshair cursor)
2. **Single-click** → places a vertex (debounced 220ms to avoid false clicks from double-click)
3. **Double-click** → finishes the polygon (minimum 3 points / 6 coords required)
4. Polygon sent to `POST /api/add` → appears on canvas immediately

### Delete Flow (Why It's Instant & Permanent)

```
User triggers delete
       │
       ▼
setLocalDeleted(annId)    ← polygon hidden from canvas IMMEDIATELY
setSelectedAnn(null)
       │
       ▼
postDelete(annId)         ← persisted to corrections.json in background
       │
       ▼
onBatchUpdate()           ← update header stats (non-blocking)

localDeleted is NEVER cleared during the session
  → polygon stays hidden no matter what
  → reset only when user navigates to a different image
```

---

## Resume / Partial Re-runs

```bash
# Resume from Step 3 (Steps 1 & 2 already done)
python run_pipeline.py ... --start_step 3

# Run pipeline only, don't launch HITL server
python run_pipeline.py ... --no_launch

# Launch HITL server manually (after pipeline already ran)
python backend/main.py \
  --coco       /path/to/pipeline_output/coco_updated.json \
  --images     /path/to/images \
  --output_dir /path/to/pipeline_output/hitl_corrections \
  --batch_size 2000 \
  --port       8000
```

---

## Output Directory Structure

```
pipeline_output/
├── coco_all.json             ← Step 1: raw YOLO detections
├── coco_added_cats.json      ← Step 2: + extra color categories
├── coco_updated.json         ← Step 4: color-classified (HITL input)
├── crops/
│   ├── Color_HD/
│   │   └── frame_001__ann452.jpg
│   └── Color_PP/
│       └── frame_003__ann801.jpg
└── hitl_corrections/
    ├── corrections.json      ← human edits, deletions, additions
    └── progress.json         ← approved batch indices
```
