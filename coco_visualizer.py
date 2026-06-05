"""
coco_visualizer.py
──────────────────────────────────────────────────────────────────
Local COCO annotation visualizer — no install beyond Python stdlib.

Usage
─────
    python coco_visualizer.py \
        --coco  /home/wi/Avinash_Works/auto_annotation_pipeline/merged_coco.json \
        --images /home/wi/Avinash_Works/waste-masknet/waste/data/images

Then open  http://localhost:8765  in your browser.

Features
────────
  • Browse all 53 k images with pagination
  • Click any image → full overlay with coloured polygon masks per category
  • Filter by category name
  • Search by filename
  • Per-category annotation counts shown in the sidebar
  • Keyboard shortcuts: ← → to navigate, Esc to close overlay
"""

import argparse
import base64
import http.server
import json
import os
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Optional
from typing import Dict, List

# ── tiny in-process data store ────────────────────────────────────────────────

COCO: dict = {}
IMG_DIR: Path = Path()
CAT_MAP: dict = {}     # id → name
CAT_COLOR: dict = {}   # name → hex color

# 21 distinct colours — one per category (wraps if more)
PALETTE = [
    "#3B82F6","#EF4444","#F59E0B","#10B981","#8B5CF6",
    "#F97316","#06B6D4","#84CC16","#EC4899","#6366F1",
    "#14B8A6","#F43F5E","#A855F7","#22C55E","#EAB308",
    "#64748B","#0EA5E9","#D97706","#DC2626","#7C3AED","#059669",
]


def load_coco(coco_path: Path) -> None:
    global COCO, CAT_MAP, CAT_COLOR
    with coco_path.open() as f:
        COCO = json.load(f)
    for cat in COCO.get("categories", []):
        CAT_MAP[cat["id"]] = cat["name"]
    for i, cat in enumerate(COCO.get("categories", [])):
        CAT_COLOR[cat["name"]] = PALETTE[i % len(PALETTE)]

def img_to_b64(path: Path) -> Optional[str]:
# def img_to_b64(path: Path) -> str | None:
    """Read an image file and return a base64 data-URI."""
    if not path.exists():
        return None
    ext = path.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
            "webp": "webp", "bmp": "bmp"}.get(ext, "jpeg")
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"data:image/{mime};base64,{b64}"


# ── build lookup tables once ──────────────────────────────────────────────────

# _ann_by_image: dict[int, list] = {}    # image_id → [annotations]
# _img_by_id:    dict[int, dict] = {}    # image_id → image record

_ann_by_image: Dict[int, List] = {}
_img_by_id:    Dict[int, dict] = {}


def build_index() -> None:
    global _ann_by_image, _img_by_id
    _img_by_id = {img["id"]: img for img in COCO.get("images", [])}
    for ann in COCO.get("annotations", []):
        _ann_by_image.setdefault(ann["image_id"], []).append(ann)


# ── HTML fragments ────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>COCO Visualizer</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,sans-serif;background:#f8f8f7;color:#1a1a1a;display:flex;height:100vh;overflow:hidden}
  #sidebar{width:260px;min-width:260px;background:#fff;border-right:1px solid #e5e5e5;display:flex;flex-direction:column;overflow:hidden}
  #sidebar h1{font-size:15px;font-weight:600;padding:16px;border-bottom:1px solid #e5e5e5}
  #stats{padding:12px 16px;font-size:12px;color:#666;border-bottom:1px solid #e5e5e5;line-height:1.8}
  #controls{padding:12px 16px;border-bottom:1px solid #e5e5e5;display:flex;flex-direction:column;gap:8px}
  #controls input,#controls select{width:100%;padding:6px 10px;border:1px solid #d5d5d5;border-radius:6px;font-size:13px;outline:none}
  #controls input:focus,#controls select:focus{border-color:#3B82F6}
  #cat-list{overflow-y:auto;flex:1;padding:8px}
  .cat-item{display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:6px;cursor:pointer;font-size:12px}
  .cat-item:hover{background:#f0f0f0}
  .cat-item.active{background:#EFF6FF}
  .cat-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
  .cat-count{margin-left:auto;color:#999;font-size:11px}
  #main{flex:1;display:flex;flex-direction:column;overflow:hidden}
  #toolbar{padding:10px 16px;background:#fff;border-bottom:1px solid #e5e5e5;display:flex;align-items:center;gap:12px;font-size:13px}
  #grid{flex:1;overflow-y:auto;padding:16px;display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;align-content:start}
  .thumb{position:relative;border-radius:8px;overflow:hidden;background:#e8e8e8;aspect-ratio:4/3;cursor:pointer;border:2px solid transparent;transition:border-color .15s}
  .thumb:hover{border-color:#3B82F6}
  .thumb img{width:100%;height:100%;object-fit:cover}
  .thumb .badge{position:absolute;bottom:4px;right:4px;background:rgba(0,0,0,.6);color:#fff;font-size:10px;padding:2px 6px;border-radius:4px}
  .thumb .fname{position:absolute;top:0;left:0;right:0;background:rgba(0,0,0,.45);color:#fff;font-size:10px;padding:3px 6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  #pagination{padding:10px 16px;background:#fff;border-top:1px solid #e5e5e5;display:flex;align-items:center;gap:8px;font-size:13px}
  #pagination button{padding:5px 12px;border:1px solid #d5d5d5;border-radius:6px;background:#fff;cursor:pointer;font-size:13px}
  #pagination button:hover{background:#f0f0f0}
  #pagination button:disabled{opacity:.4;cursor:default}
  #overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:100;flex-direction:column;align-items:center;justify-content:center}
  #overlay.open{display:flex}
  #ov-box{background:#fff;border-radius:12px;overflow:hidden;max-width:92vw;max-height:92vh;display:flex;flex-direction:column}
  #ov-header{padding:10px 16px;border-bottom:1px solid #e5e5e5;display:flex;align-items:center;gap:12px;font-size:13px}
  #ov-header strong{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  #ov-header button{padding:4px 10px;border:1px solid #d5d5d5;border-radius:6px;background:#fff;cursor:pointer}
  #ov-canvas-wrap{flex:1;overflow:auto;position:relative}
  canvas#ov-canvas{display:block}
  #ov-legend{padding:10px 16px;border-top:1px solid #e5e5e5;display:flex;flex-wrap:wrap;gap:8px;font-size:11px;max-height:80px;overflow-y:auto}
  .leg-item{display:flex;align-items:center;gap:4px}
  .leg-dot{width:8px;height:8px;border-radius:50%}
  #loading{position:fixed;inset:0;background:#fff;display:flex;align-items:center;justify-content:center;font-size:14px;color:#666;z-index:200}
</style>
</head>
<body>
<div id="loading">Loading dataset…</div>

<div id="sidebar">
  <h1>COCO Visualizer</h1>
  <div id="stats"></div>
  <div id="controls">
    <input id="search" placeholder="Search filename…" type="search">
    <select id="cat-filter"><option value="">All categories</option></select>
  </div>
  <div id="cat-list"></div>
</div>

<div id="main">
  <div id="toolbar">
    <span id="page-info">–</span>
    <span style="flex:1"></span>
    <label style="display:flex;align-items:center;gap:6px">
      <input type="checkbox" id="show-masks" checked> Show masks
    </label>
  </div>
  <div id="grid"></div>
  <div id="pagination">
    <button id="btn-prev" disabled>← Prev</button>
    <span id="pag-label"></span>
    <button id="btn-next">Next →</button>
    <select id="page-size" style="margin-left:auto;padding:4px 8px;border:1px solid #d5d5d5;border-radius:6px;font-size:13px">
      <option value="30" selected>30 / page</option>
      <option value="60">60 / page</option>
      <option value="100">100 / page</option>
    </select>
  </div>
</div>

<div id="overlay">
  <div id="ov-box">
    <div id="ov-header">
      <button id="ov-prev">←</button>
      <strong id="ov-title"></strong>
      <button id="ov-next">→</button>
      <button id="ov-close">✕</button>
    </div>
    <div id="ov-canvas-wrap"><canvas id="ov-canvas"></canvas></div>
    <div id="ov-legend"></div>
  </div>
</div>

<script>
const CATS = __CATS_JSON__;
const CAT_COLOR = __CAT_COLOR_JSON__;
const CAT_COUNTS = __CAT_COUNTS_JSON__;
const PAGE_SIZE_DEFAULT = 30;

let allImages = [];
let filtered = [];
let page = 0;
let pageSize = PAGE_SIZE_DEFAULT;
let activeCat = "";
let searchQ = "";
let overlayIdx = -1;

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${alpha})`;
}

async function init() {
  const resp = await fetch("/api/images");
  allImages = await resp.json();
  document.getElementById("loading").style.display = "none";

  const statsEl = document.getElementById("stats");
  statsEl.innerHTML = `<b>${allImages.length.toLocaleString()}</b> images<br><b>__TOTAL_ANNS__</b> annotations<br><b>${CATS.length}</b> categories`;

  const catFilter = document.getElementById("cat-filter");
  const catList   = document.getElementById("cat-list");
  CATS.forEach(c => {
    const opt = document.createElement("option");
    opt.value = c.name; opt.textContent = c.name;
    catFilter.appendChild(opt);

    const div = document.createElement("div");
    div.className = "cat-item";
    div.dataset.cat = c.name;
    div.innerHTML = `<span class="cat-dot" style="background:${CAT_COLOR[c.name]}"></span>
      <span>${c.name}</span><span class="cat-count">${(CAT_COUNTS[c.name]||0).toLocaleString()}</span>`;
    div.onclick = () => setCatFilter(c.name === activeCat ? "" : c.name);
    catList.appendChild(div);
  });

  applyFilters();
}

function setCatFilter(name) {
  activeCat = name;
  document.getElementById("cat-filter").value = name;
  document.querySelectorAll(".cat-item").forEach(el => {
    el.classList.toggle("active", el.dataset.cat === name);
  });
  applyFilters();
}

function applyFilters() {
  const q = searchQ.toLowerCase();
  filtered = allImages.filter(img => {
    if (q && !img.file_name.toLowerCase().includes(q)) return false;
    if (activeCat && !img.cats.includes(activeCat)) return false;
    return true;
  });
  page = 0;
  render();
}

function render() {
  const start = page * pageSize;
  const slice = filtered.slice(start, start + pageSize);
  const totalPages = Math.ceil(filtered.length / pageSize);

  document.getElementById("page-info").textContent =
    `${filtered.length.toLocaleString()} images (filtered)`;
  document.getElementById("pag-label").textContent =
    `Page ${page+1} / ${Math.max(1,totalPages)}`;
  document.getElementById("btn-prev").disabled = page === 0;
  document.getElementById("btn-next").disabled = page >= totalPages - 1;

  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  slice.forEach((img, i) => {
    const div = document.createElement("div");
    div.className = "thumb";
    div.innerHTML = `<img src="/api/thumb/${img.id}" loading="lazy">
      <div class="fname">${img.file_name.split('/').pop()}</div>
      <div class="badge">${img.ann_count} ann</div>`;
    div.onclick = () => openOverlay(start + i);
    grid.appendChild(div);
  });
}

// ── overlay ────────────────────────────────────────────────────────────────

async function openOverlay(idx) {
  overlayIdx = idx;
  const img = filtered[idx];
  document.getElementById("ov-title").textContent = img.file_name.split('/').pop();
  document.getElementById("overlay").classList.add("open");
  await drawOverlay(img);
}

async function drawOverlay(img) {
  const resp = await fetch(`/api/annotations/${img.id}`);
  const anns = await resp.json();
  const imgResp = await fetch(`/api/image_b64/${img.id}`);
  const {b64, width, height} = await imgResp.json();

  const canvas = document.getElementById("ov-canvas");
  const maxW = window.innerWidth * 0.85;
  const maxH = window.innerHeight * 0.72;
  const scale = Math.min(1, maxW / width, maxH / height);
  canvas.width  = Math.round(width  * scale);
  canvas.height = Math.round(height * scale);

  const ctx = canvas.getContext("2d");
  const image = new Image();
  image.onload = () => {
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    if (document.getElementById("show-masks").checked) {
      anns.forEach(ann => {
        const catName = ann.cat_name;
        const color   = CAT_COLOR[catName] || "#888";
        ann.segmentation.forEach(seg => {
          if (!seg.length) return;
          ctx.beginPath();
          ctx.moveTo(seg[0]*scale, seg[1]*scale);
          for (let i=2;i<seg.length;i+=2) ctx.lineTo(seg[i]*scale, seg[i+1]*scale);
          ctx.closePath();
          ctx.fillStyle   = hexToRgba(color, 0.35);
          ctx.strokeStyle = color;
          ctx.lineWidth   = 1.5;
          ctx.fill(); ctx.stroke();
        });
      });
    }
    renderLegend(anns);
  };
  image.src = b64;
}

function renderLegend(anns) {
  const seen = {};
  anns.forEach(a => { seen[a.cat_name] = (seen[a.cat_name]||0)+1; });
  const leg = document.getElementById("ov-legend");
  leg.innerHTML = Object.entries(seen).map(([name,cnt]) =>
    `<span class="leg-item"><span class="leg-dot" style="background:${CAT_COLOR[name]||'#888'}"></span>${name} (${cnt})</span>`
  ).join("");
}

function closeOverlay() {
  document.getElementById("overlay").classList.remove("open");
  overlayIdx = -1;
}

// ── events ─────────────────────────────────────────────────────────────────

document.getElementById("search").addEventListener("input", e => {
  searchQ = e.target.value; applyFilters();
});
document.getElementById("cat-filter").addEventListener("change", e => {
  setCatFilter(e.target.value);
});
document.getElementById("btn-prev").onclick = () => { page--; render(); };
document.getElementById("btn-next").onclick = () => { page++; render(); };
document.getElementById("page-size").onchange = e => { pageSize = +e.target.value; page=0; render(); };
document.getElementById("ov-close").onclick = closeOverlay;
document.getElementById("ov-prev").onclick = () => { if(overlayIdx>0) openOverlay(overlayIdx-1); };
document.getElementById("ov-next").onclick = () => { if(overlayIdx<filtered.length-1) openOverlay(overlayIdx+1); };
document.getElementById("overlay").onclick = e => { if(e.target===document.getElementById("overlay")) closeOverlay(); };
document.getElementById("show-masks").onchange = () => {
  if(overlayIdx>=0) drawOverlay(filtered[overlayIdx]);
};
document.addEventListener("keydown", e => {
  if(e.key==="Escape") closeOverlay();
  if(e.key==="ArrowRight" && overlayIdx>=0 && overlayIdx<filtered.length-1) openOverlay(overlayIdx+1);
  if(e.key==="ArrowLeft"  && overlayIdx>0) openOverlay(overlayIdx-1);
});

init();
</script>
</body>
</html>
"""

class Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # suppress request logs

    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html: str):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path

        # ── root ─────────────────────────────────────────────────────────────
        if path == "/" or path == "/index.html":
            cats      = COCO.get("categories", [])
            cat_color = {name: CAT_COLOR.get(name, "#888") for name in CAT_MAP.values()}

            # count annotations per category name
            cat_counts: dict[str, int] = {c["name"]: 0 for c in cats}
            for ann in COCO.get("annotations", []):
                name = CAT_MAP.get(ann["category_id"], "Unknown")
                cat_counts[name] = cat_counts.get(name, 0) + 1

            html = (HTML_PAGE
                    .replace("__CATS_JSON__",       json.dumps(cats))
                    .replace("__CAT_COLOR_JSON__",  json.dumps(cat_color))
                    .replace("__CAT_COUNTS_JSON__", json.dumps(cat_counts))
                    .replace("__TOTAL_ANNS__",
                             f"{len(COCO.get('annotations', [])):,}"))
            self.send_html(html)
            return

        # ── /api/images  — lightweight list for the grid ──────────────────────
        if path == "/api/images":
            # build per-image category set
            img_cats: dict[int, set] = {}
            img_ann_count: dict[int, int] = {}
            for ann in COCO.get("annotations", []):
                iid = ann["image_id"]
                img_cats.setdefault(iid, set()).add(
                    CAT_MAP.get(ann["category_id"], "Unknown"))
                img_ann_count[iid] = img_ann_count.get(iid, 0) + 1

            result = [
                {
                    "id":        img["id"],
                    "file_name": img["file_name"],
                    "ann_count": img_ann_count.get(img["id"], 0),
                    "cats":      list(img_cats.get(img["id"], [])),
                }
                for img in COCO.get("images", [])
            ]
            self.send_json(result)
            return

        # ── /api/thumb/<id> ───────────────────────────────────────────────────
        if path.startswith("/api/thumb/"):
            img_id = int(path.split("/")[-1])
            img    = _img_by_id.get(img_id)
            if not img:
                self.send_response(404); self.end_headers(); return
            file_path = IMG_DIR / img["file_name"]
            if not file_path.exists():
                # try just the basename
                file_path = IMG_DIR / Path(img["file_name"]).name
            if file_path.exists():
                body = file_path.read_bytes()
                ext  = file_path.suffix.lower().lstrip(".")
                mime = {"jpg":"jpeg","jpeg":"jpeg","png":"png","webp":"webp"}.get(ext,"jpeg")
                self.send_response(200)
                self.send_header("Content-Type", f"image/{mime}")
                self.send_header("Content-Length", len(body))
                self.end_headers()
                self.wfile.write(body)
            else:
                # grey placeholder
                self.send_response(200)
                self.send_header("Content-Type", "image/svg+xml")
                svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="180" height="135"><rect width="180" height="135" fill="#ddd"/><text x="90" y="72" text-anchor="middle" font-size="12" fill="#999">no image</text></svg>'
                self.send_header("Content-Length", len(svg))
                self.end_headers()
                self.wfile.write(svg)
            return

        # ── /api/image_b64/<id> ───────────────────────────────────────────────
        if path.startswith("/api/image_b64/"):
            img_id = int(path.split("/")[-1])
            img    = _img_by_id.get(img_id)
            if not img:
                self.send_json({"error": "not found"}, 404); return
            file_path = IMG_DIR / img["file_name"]
            if not file_path.exists():
                file_path = IMG_DIR / Path(img["file_name"]).name
            b64 = img_to_b64(file_path) or "data:image/svg+xml;base64," + base64.b64encode(
                b'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="640" height="480" fill="#ccc"/></svg>'
            ).decode()
            self.send_json({
                "b64":    b64,
                "width":  img.get("width",  640),
                "height": img.get("height", 480),
            })
            return

        # ── /api/annotations/<id> ─────────────────────────────────────────────
        if path.startswith("/api/annotations/"):
            img_id = int(path.split("/")[-1])
            anns   = _ann_by_image.get(img_id, [])
            result = [
                {
                    "id":           ann["id"],
                    "cat_name":     CAT_MAP.get(ann["category_id"], "Unknown"),
                    "segmentation": ann.get("segmentation", []),
                    "bbox":         ann.get("bbox", []),
                    "attributes":   ann.get("attributes", {}),
                }
                for ann in anns
            ]
            self.send_json(result)
            return

        self.send_response(404); self.end_headers()


def main() -> None:
    ap = argparse.ArgumentParser(description="Local COCO annotation visualizer")
    ap.add_argument("--coco",   required=True,  help="Path to merged_coco.json")
    ap.add_argument("--images", required=True,  help="Root image directory")
    ap.add_argument("--port",   type=int, default=8765)
    args = ap.parse_args()

    global IMG_DIR
    IMG_DIR = Path(args.images)

    coco_path = Path(args.coco)
    if not coco_path.exists():
        print(f"ERROR: COCO file not found: {coco_path}")
        sys.exit(1)
    if not IMG_DIR.is_dir():
        print(f"ERROR: Image directory not found: {IMG_DIR}")
        sys.exit(1)

    print(f"Loading {coco_path} …")
    load_coco(coco_path)
    build_index()

    n_imgs = len(COCO.get("images", []))
    n_anns = len(COCO.get("annotations", []))
    n_cats = len(COCO.get("categories", []))
    print(f"  {n_imgs:,} images  |  {n_anns:,} annotations  |  {n_cats} categories")

    url = f"http://localhost:{args.port}"
    print(f"\nStarting server → {url}")
    print("Press Ctrl+C to stop.\n")

    server = http.server.HTTPServer(("", args.port), Handler)
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()