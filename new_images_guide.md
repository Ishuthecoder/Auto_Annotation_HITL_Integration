# How to Auto-Annotate New Images (Material, Color & Grade)

## What Happens Under the Hood

When you run the pipeline on new images, here's what the system does automatically:

```
Your New Images (downloaded to a folder)
         │
         ▼
 STEP 1 — YOLOv8 Segmentation Model
         Detects every object → draws polygon masks
         Identifies material/shape class (HD, PP, etc.)
         Output: coco_all.json
         │
         ▼
 STEP 2 — Category Expansion
         Adds fine-grained color labels (Green_HD, Black_HD, Blue_PP, Green_PP…)
         Output: coco_added_cats.json
         │
         ▼
 STEP 3 — Crop Generation
         Cuts out each detected object as a tight crop image
         Organizes them: crops/Color_HD/, crops/Color_PP/, …
         │
         ▼
 STEP 4 — Color Classifier Model
         Runs on each crop → assigns color prediction
         e.g. "this is Green_HD", "this is Blue_PP"
         Output: coco_updated.json  ← your auto-annotations are DONE
         │
         ▼
 HITL Review Server (auto-launches)
         Opens in browser → you review, fix mistakes, approve batches
         All human corrections saved to corrections.json
```

---

## Step-by-Step: What You Do

### ① Put your new images in a folder

```
/path/to/your/new_images/
    ├── img_001.jpg
    ├── img_002.jpg
    └── ...
```

### ② Run ONE command

```bash
cd /home/wi/Documents/Ishika_work/auto_hitl_integration

python run_pipeline.py \
  --image_dir   /path/to/your/new_images \
  --shape_model /media/wi/ssd_hub/training_runs/yolov8m_seg_real_only_20260423_175631/weights/best.pt \
  --color_model /media/wi/ssd_hub/Ishika_works/training_runs/yolov8m_color_seg/weights/best.pt \
  --output_dir  /path/to/your/new_images_output \
  --batch_size  500 \
  --port        8000
```

> **Replace** `/path/to/your/new_images` with your actual folder.
> **Replace** `/path/to/your/new_images_output` with where you want results saved.
> The model paths are your trained `.pt` weights — use the ones that match above.

---

### ③ Optional: Process only images from a specific date range

If you downloaded images captured between specific dates, use timestamp filtering so only those are processed:

```bash
python run_pipeline.py \
  --image_dir   /path/to/your/new_images \
  --shape_model /path/to/segmentation.pt \
  --color_model /path/to/color_model.pt \
  --output_dir  /path/to/output \
  --start_time  "2026-06-01" \
  --end_time    "2026-06-05"
```

Flexible formats accepted:
| What you type | Meaning |
|---|---|
| `"2026-06-01"` | From midnight on June 1 |
| `"2026-06-01 09:00:00"` | From 9 AM on June 1 |
| `"20260601_090000"` | Same, compact format |
| `"2026-06-05 18:00:00"` | Until 6 PM on June 5 |

---

### ④ Wait for Steps 1–4 to finish (terminal shows progress)

You'll see logs like:
```
12:30:01 | INFO     | Step 1 — YOLO segmentation …
12:30:45 | INFO     | Per-class annotation counts: {3: 842, 12: 310}
12:30:46 | INFO     | Step 2 — Adding extra categories …
12:31:00 | INFO     | Step 3 — Generating crops …
12:31:55 | INFO     | Step 4 — Color classification …
12:32:10 | INFO     | Pipeline complete!
12:32:10 | INFO     | Launching HITL Review Server
```

> [!IMPORTANT]
> Check the **"Per-class annotation counts"** line from Step 1.
> It shows your actual YOLO class IDs (e.g. `{3: 842, 12: 310}`).
> The default `--class_folder_map "3:Color_HD,12:Color_PP"` must match these IDs.
> If they differ, re-run with `--class_folder_map "YOUR_ID:Color_HD,YOUR_ID2:Color_PP"`.

---

### ⑤ HITL Review opens automatically in terminal

The backend server starts. Now open your browser:

```
http://localhost:8000
```

Or open the **React frontend** (in a second terminal):
```bash
cd frontend
npm run dev
# → opens at http://localhost:5173
```

---

## What You See in HITL Review (Material / Color / Grade)

Once the browser is open:

```
Left Sidebar          Main Canvas              Right Sidebar
─────────────         ─────────────────────    ─────────────────────
Batch 1  ✅           [Image with colored      Selected Annotation:
Batch 2  ✅            polygon overlays]        ┌──────────────────┐
Batch 3  (current) →                            │ Category: Color_HD│
Batch 4                                         │ Material: [HD  ▼]│
Batch 5                                         │ Color:  [Green ▼]│
                                                │ Grade:  [A    ▼]│
                                                └──────────────────┘
                                                [🗑 Delete] [✅ Approve]
```

### What each field means

| Field | Auto-filled by | What you do |
|---|---|---|
| **Category** | Step 1 YOLO segmentation model | Verify it's correct (HD vs PP) |
| **Color** | Step 4 Color classifier model | Fix if wrong (Green/Black/Blue) |
| **Material** | Derived from category | Adjust if needed |
| **Grade** | ⚠️ Not auto-predicted — defaults | **You must fill this in HITL** |

---

## Human-in-the-Loop (HITL) Actions

| What you want to do | How |
|---|---|
| Fix wrong color label | Click polygon → change **Color** dropdown → auto-saves |
| Fix wrong material | Click polygon → change **Category/Material** dropdown → auto-saves |
| Delete a wrong detection | Click polygon → press `Delete` key or click 🗑 button |
| Add a missed object | Click **"+ Draw Polygon"** → click vertices → double-click to finish |
| Move to next image | Scroll or navigate — corrections already saved |
| Mark batch as done | Click ✅ **Approve Batch** |

---

## Output Files After You're Done

```
new_images_output/
├── coco_all.json             ← Step 1 raw detections
├── coco_added_cats.json      ← Step 2 expanded categories  
├── coco_updated.json         ← Step 4 auto-annotated (color classified)
├── crops/
│   ├── Color_HD/             ← cropped HD objects
│   └── Color_PP/             ← cropped PP objects
└── hitl_corrections/
    ├── corrections.json      ← YOUR human edits (edits, deletions, additions)
    └── progress.json         ← which batches you approved
```

> [!TIP]
> `coco_updated.json` = pure machine output
> `corrections.json` = your human corrections layered on top
> The backend **merges both** before showing you the review UI — you always see the final merged view.

---

## Quick Reference: Most Common Flags

| Flag | Default | When to change |
|---|---|---|
| `--batch_size` | 2000 | Lower to 200-500 if you want smaller review chunks |
| `--shape_conf` | 0.20 | Raise to 0.3-0.5 to reduce false positives |
| `--color_conf` | 0.25 | Raise to 0.4+ if color predictions are noisy |
| `--min_area` | 200 px² | Raise to ignore tiny fragments |
| `--start_step` | 1 | Set to 3 or 4 to resume if pipeline crashed mid-way |
| `--no_launch` | off | Add to skip auto-launching HITL server |

---

## Resuming If Pipeline Crashes Mid-Way

If the pipeline stops in the middle (e.g., crash during Step 3), don't re-run from scratch:

```bash
# Resume from Step 3 (Steps 1 and 2 already done)
python run_pipeline.py \
  --image_dir   /path/to/new_images \
  --shape_model /path/to/seg.pt \
  --color_model /path/to/color.pt \
  --output_dir  /path/to/output \
  --start_step  3
```

---

## TL;DR — 3-Step Summary

```
1. Drop images in a folder
2. Run: python run_pipeline.py --image_dir <folder> --shape_model <seg.pt> --color_model <color.pt> --output_dir <output>
3. Open browser → http://localhost:5173 → review, fix, approve
```
