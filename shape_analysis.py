"""
STEP 3 — Shape Analysis Visualization
======================================
Reads step2_output/detections.json (produced by Step 2) and
generates a single comprehensive visualization:

  Panel 1 — Spatial heatmap  : where on the image detections fire
  Panel 2 — W vs H scatter   : detection shape vs GT shape
  Panel 3 — Width histogram  : detected widths vs GT widths
  Panel 4 — Aspect ratio     : det AR vs GT AR (Haar locked at 1.0)
  Panel 5 — Size comparison  : det sizes vs GT sizes as bar chart
  Panel 6 — Miss analysis    : where in the size spectrum Haar misses

Output: step3_output/shape_analysis.png

Run:
    python step3_shape_analysis.py
"""

import cv2
import json
import os
import numpy as np
from pathlib import Path

IN_JSON  = "step2_output/detections.json"
OUT_DIR  = "step3_output"
OUT_PATH = os.path.join(OUT_DIR, "shape_analysis.png")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load detections.json ──────────────────────────────────
print("=" * 58)
print("  STEP 3 — Shape Analysis Visualization")
print("=" * 58)

with open(IN_JSON) as f:
    data = json.load(f)

# Collect all detection boxes and GT boxes
det_ws, det_hs, det_ars = [], [], []
det_cxs, det_cys        = [], []

gt_ws,  gt_hs,  gt_ars  = [], [], []
gt_cxs, gt_cys          = [], []

missed_ws, missed_hs     = [], []   # GT boxes that were NOT matched

IOU_THRESH = data["accuracy"]["iou_threshold"]

def iou(a, b):
    ax1,ay1,aw,ah = a;  bx1,by1,bw,bh = b
    ax2,ay2 = ax1+aw,ay1+ah;  bx2,by2 = bx1+bw,by1+bh
    ix = max(0, min(ax2,bx2)-max(ax1,bx1))
    iy = max(0, min(ay2,by2)-max(ay1,by1))
    inter = ix*iy;  union = aw*ah+bw*bh-inter
    return inter/union if union>0 else 0.0

for item in data["per_image"]:
    gt_boxes  = [tuple(b) for b in item["gt"]]
    det_boxes = [tuple(b) for b in item["dets"]]

    # Collect det stats
    for (x,y,w,h) in det_boxes:
        det_ws.append(w);  det_hs.append(h)
        det_ars.append(w/h if h>0 else 1.0)
        det_cxs.append(x + w//2)
        det_cys.append(y + h//2)

    # Collect GT stats + find missed boxes
    matched_gt = set()
    if gt_boxes and det_boxes:
        mat = np.array([[iou(g,d) for d in det_boxes] for g in gt_boxes])
        tmp = mat.copy()
        while tmp.max() >= IOU_THRESH:
            gi,di = np.unravel_index(tmp.argmax(), tmp.shape)
            matched_gt.add(gi)
            tmp[gi,:] = -1;  tmp[:,di] = -1

    for i,(x,y,w,h) in enumerate(gt_boxes):
        gt_ws.append(w);  gt_hs.append(h)
        gt_ars.append(w/h if h>0 else 1.0)
        gt_cxs.append(x + w//2)
        gt_cys.append(y + h//2)
        if i not in matched_gt:
            missed_ws.append(w);  missed_hs.append(h)

print(f"  Detections  : {len(det_ws)}")
print(f"  GT faces    : {len(gt_ws)}")
print(f"  Missed GT   : {len(missed_ws)}")

# ── Canvas setup ─────────────────────────────────────────
CW, CH = 1500, 960
canvas = np.full((CH, CW, 3), 248, dtype=np.uint8)

DARK   = (30,  30,  30)
MID    = (100, 100, 100)
LIGHT  = (180, 180, 180)
BLUE   = (200, 120,  40)    # BGR — detection colour
GREEN  = ( 60, 160,  60)    # GT colour
RED    = ( 50,  50, 210)    # missed colour
ORANGE = ( 30, 140, 200)

font  = cv2.FONT_HERSHEY_SIMPLEX
fontB = cv2.FONT_HERSHEY_DUPLEX

def txt(s, x, y, scale=0.42, col=DARK, bold=False):
    cv2.putText(canvas, s, (x,y), fontB if bold else font,
                scale, col, 1, cv2.LINE_AA)

def panel(px,py,pw,ph,title=""):
    cv2.rectangle(canvas,(px,py),(px+pw,py+ph),(210,210,210),1)
    if title:
        cv2.putText(canvas,title,(px+10,py+20),fontB,0.48,DARK,1,cv2.LINE_AA)

def legend_dot(x,y,col,label):
    cv2.circle(canvas,(x,y),6,col,-1)
    txt(label, x+10, y+4, 0.38)

# Title bar
cv2.rectangle(canvas,(0,0),(CW,46),(50,50,50),-1)
n_img = data["n_images"]
cv2.putText(canvas,
    f"Shape Analysis — Haar Cascade Baseline   "
    f"({len(det_ws)} dets / {len(gt_ws)} GT faces / {n_img} images)",
    (14,30), fontB, 0.62, (230,230,230), 1, cv2.LINE_AA)

PAD=14; TOP=54
PW=(CW-4*PAD)//3;  PH=(CH-TOP-3*PAD)//2
panels=[
    (PAD,          TOP,        PW,PH),
    (PAD*2+PW,     TOP,        PW,PH),
    (PAD*3+PW*2,   TOP,        PW,PH),
    (PAD,          TOP+PH+PAD, PW,PH),
    (PAD*2+PW,     TOP+PH+PAD, PW,PH),
    (PAD*3+PW*2,   TOP+PH+PAD, PW,PH),
]
ptitles=[
    "1. Spatial heatmap — where detections fire",
    "2. Detected vs GT bounding-box shape",
    "3. Width distribution: det vs GT",
    "4. Aspect ratio: det vs GT (Haar = always 1.0)",
    "5. Face size coverage: what Haar misses",
    "6. Summary statistics",
]
for (px,py,pw,ph),t in zip(panels,ptitles):
    panel(px,py,pw,ph,t)

# inner areas
def inner(idx):
    px,py,pw,ph = panels[idx]
    return px+8, py+28, pw-16, ph-36

# ═══════════════════════════════════════════════════════════
# PANEL 1 — Spatial heatmap
# ═══════════════════════════════════════════════════════════
ix,iy,iw,ih = inner(0)

# Use a representative image size (1024×1024 for train, 1024×576 for val)
# normalise cx/cy to [0,1] relative to image
# We'll use per-image sizes from GT — approximate with 1024x1024
IMG_W, IMG_H = 1024, 1024

hmap = np.zeros((IMG_H, IMG_W), np.float32)
for cx2,cy2 in zip(det_cxs, det_cys):
    cx2 = max(0, min(IMG_W-1, int(cx2)))
    cy2 = max(0, min(IMG_H-1, int(cy2)))
    hmap[cy2, cx2] += 1

hmap = cv2.GaussianBlur(hmap, (81, 81), 28)
hmap = cv2.normalize(hmap, None, 0, 255, cv2.NORM_MINMAX)
hmap_col = cv2.applyColorMap(hmap.astype(np.uint8), cv2.COLORMAP_JET)

# grid overlay
grid = np.full((IMG_H, IMG_W, 3), 230, np.uint8)
for gx in range(0, IMG_W, IMG_W//8):
    cv2.line(grid,(gx,0),(gx,IMG_H),(200,200,200),1)
for gy in range(0, IMG_H, IMG_H//8):
    cv2.line(grid,(0,gy),(IMG_W,gy),(200,200,200),1)

blended = cv2.addWeighted(grid, 0.25, hmap_col, 0.75, 0)
blended = cv2.resize(blended, (iw, ih))
canvas[iy:iy+ih, ix:ix+iw] = blended

txt("x →", ix+iw//2-12, iy+ih+12, 0.36, MID)
txt("y", ix-14, iy+ih//2+4, 0.36, MID)
# Colorbar label
txt("cold=few  hot=many", ix+2, iy+ih-6, 0.33, (220,220,220))

# ═══════════════════════════════════════════════════════════
# PANEL 2 — W vs H scatter (det=blue, GT=green)
# ═══════════════════════════════════════════════════════════
ix,iy,iw,ih = inner(1)
canvas[iy:iy+ih, ix:ix+iw] = 255

max_dim = max(max(det_ws+gt_ws+[1]), max(det_hs+gt_hs+[1]))

# GT dots (sample 1500)
step_gt = max(1, len(gt_ws)//1500)
for w2,h2 in zip(gt_ws[::step_gt], gt_hs[::step_gt]):
    sx = ix + int(w2/max_dim*(iw-4))
    sy = iy + ih - int(h2/max_dim*(ih-4))
    cv2.circle(canvas,(sx,sy),2,GREEN,-1)

# Det dots (sample 1000)
step_d = max(1, len(det_ws)//1000)
for w2,h2 in zip(det_ws[::step_d], det_hs[::step_d]):
    sx = ix + int(w2/max_dim*(iw-4))
    sy = iy + ih - int(h2/max_dim*(ih-4))
    cv2.circle(canvas,(sx,sy),2,BLUE,-1)

# w=h diagonal
cv2.line(canvas,(ix,iy+ih),(ix+iw,iy),(180,180,180),1)
txt("w=h", ix+iw-36, iy+14, 0.35, LIGHT)

legend_dot(ix+4,    iy+ih-22, GREEN, "GT")
legend_dot(ix+50,   iy+ih-22, BLUE,  "Det")

txt("width (px)",  ix+iw//2-25, iy+ih+12, 0.36, MID)
txt("height",      ix-44,       iy+ih//2, 0.36, MID)

# ═══════════════════════════════════════════════════════════
# PANEL 3 — Width histogram det vs GT
# ═══════════════════════════════════════════════════════════
ix,iy,iw,ih = inner(2)
canvas[iy:iy+ih, ix:ix+iw] = 255

bins    = np.linspace(0, 400, 25)
gt_cnt, _  = np.histogram(np.clip(gt_ws,  0, 400), bins=bins)
det_cnt, _ = np.histogram(np.clip(det_ws, 0, 400), bins=bins)
max_c   = max(gt_cnt.max(), det_cnt.max(), 1)
bw      = iw // (len(bins)-1)

for bi in range(len(bins)-1):
    bx2 = ix + bi*bw
    # GT bar (behind)
    bh2 = int(gt_cnt[bi]/max_c*(ih-22))
    cv2.rectangle(canvas,(bx2,iy+ih-bh2-1),(bx2+bw-2,iy+ih-1),GREEN,-1)
    # Det bar (in front, semi-transparent via blend)
    bh3 = int(det_cnt[bi]/max_c*(ih-22))
    overlay_rect = canvas[iy+ih-bh3-1:iy+ih, bx2:bx2+bw-2].copy()
    cv2.rectangle(canvas,(bx2,iy+ih-bh3-1),(bx2+bw-2,iy+ih-1),BLUE,-1)
    cv2.addWeighted(canvas[iy+ih-bh3-1:iy+ih, bx2:bx2+bw-2],
                    0.6, overlay_rect, 0.4, 0,
                    canvas[iy+ih-bh3-1:iy+ih, bx2:bx2+bw-2])
    if bi % 5 == 0:
        txt(f"{int(bins[bi])}", bx2, iy+ih+12, 0.32, MID)

legend_dot(ix+4,  iy+12, GREEN, "GT")
legend_dot(ix+40, iy+12, BLUE,  "Det")
txt("width (px, capped 400)", ix+iw//2-50, iy+ih+12, 0.36, MID)

# ═══════════════════════════════════════════════════════════
# PANEL 4 — Aspect ratio histogram det vs GT
# ═══════════════════════════════════════════════════════════
ix,iy,iw,ih = inner(3)
canvas[iy:iy+ih, ix:ix+iw] = 255

ar_bins    = np.linspace(0.2, 2.2, 30)
gt_ar_cnt, _  = np.histogram(np.clip(gt_ars,  0.2, 2.2), bins=ar_bins)
det_ar_cnt, _ = np.histogram(np.clip(det_ars, 0.2, 2.2), bins=ar_bins)
max_ar = max(gt_ar_cnt.max(), det_ar_cnt.max(), 1)
bw2    = iw // (len(ar_bins)-1)

for bi in range(len(ar_bins)-1):
    bx2 = ix + bi*bw2
    bh2 = int(gt_ar_cnt[bi]/max_ar*(ih-22))
    cv2.rectangle(canvas,(bx2,iy+ih-bh2-1),(bx2+bw2-2,iy+ih-1),GREEN,-1)
    bh3 = int(det_ar_cnt[bi]/max_ar*(ih-22))
    overlay_r = canvas[iy+ih-bh3-1:iy+ih, bx2:bx2+bw2-2].copy()
    cv2.rectangle(canvas,(bx2,iy+ih-bh3-1),(bx2+bw2-2,iy+ih-1),BLUE,-1)
    cv2.addWeighted(canvas[iy+ih-bh3-1:iy+ih, bx2:bx2+bw2-2],
                    0.6, overlay_r, 0.4, 0,
                    canvas[iy+ih-bh3-1:iy+ih, bx2:bx2+bw2-2])
    if bi % 5 == 0:
        txt(f"{ar_bins[bi]:.1f}", bx2, iy+ih+12, 0.32, MID)

# Mark AR=1.0 (Haar's fixed window)
haar_x = ix + int((1.0-0.2)/(2.2-0.2) * iw)
cv2.line(canvas,(haar_x,iy+4),(haar_x,iy+ih-1),(0,0,180),2)
txt("Haar AR=1.0", haar_x-28, iy+ih-4, 0.33, (0,0,180))

legend_dot(ix+4,  iy+12, GREEN, "GT")
legend_dot(ix+40, iy+12, BLUE,  "Det")
txt("aspect ratio w/h", ix+iw//2-40, iy+ih+12, 0.36, MID)

# ═══════════════════════════════════════════════════════════
# PANEL 5 — Miss analysis: detected vs missed GT by size
# ═══════════════════════════════════════════════════════════
ix,iy,iw,ih = inner(4)
canvas[iy:iy+ih, ix:ix+iw] = 255

size_bins  = np.linspace(0, 300, 20)
hit_ws     = [w for w,h in zip(gt_ws,gt_hs)
              if (w,h) not in zip(missed_ws,missed_hs)]

miss_cnt, _ = np.histogram(np.clip(missed_ws, 0, 300), bins=size_bins)
gt_sz_cnt,_ = np.histogram(np.clip(gt_ws,     0, 300), bins=size_bins)
hit_cnt     = gt_sz_cnt - miss_cnt
hit_cnt     = np.clip(hit_cnt, 0, None)
max_s       = max(gt_sz_cnt.max(), 1)
bw3         = iw // (len(size_bins)-1)

for bi in range(len(size_bins)-1):
    bx2  = ix + bi*bw3
    # Stacked bar: hit (green) bottom, missed (red) on top
    h_hit  = int(hit_cnt[bi] /max_s*(ih-22))
    h_miss = int(miss_cnt[bi]/max_s*(ih-22))
    cv2.rectangle(canvas,
                  (bx2, iy+ih-h_hit-1),
                  (bx2+bw3-2, iy+ih-1), GREEN, -1)
    cv2.rectangle(canvas,
                  (bx2, iy+ih-h_hit-h_miss-1),
                  (bx2+bw3-2, iy+ih-h_hit-1), RED, -1)
    if bi % 4 == 0:
        txt(f"{int(size_bins[bi])}", bx2, iy+ih+12, 0.32, MID)

legend_dot(ix+4,  iy+12, GREEN, "Detected")
legend_dot(ix+70, iy+12, RED,   "Missed")
txt("GT face width (px, capped 300)", ix+iw//2-65, iy+ih+12, 0.36, MID)

# ═══════════════════════════════════════════════════════════
# PANEL 6 — Summary stats text
# ═══════════════════════════════════════════════════════════
ix,iy,iw,ih = inner(5)
canvas[iy:iy+ih, ix:ix+iw] = 252

acc  = data["accuracy"]
lines = [
    ("DETECTION RESULTS",             True,  DARK),
    (f"  Images         : {n_img}",   False, DARK),
    (f"  GT faces       : {len(gt_ws)}",False,DARK),
    (f"  Detections     : {len(det_ws)}",False,DARK),
    ("",False,DARK),
    ("ACCURACY",                       True,  DARK),
    (f"  Precision : {acc['precision']:.4f}",False,DARK),
    (f"  Recall    : {acc['recall']:.4f}",   False,DARK),
    (f"  F1        : {acc['f1']:.4f}",       False,DARK),
    (f"  TP={acc['TP']}  FP={acc['FP']}  FN={acc['FN']}",False,MID),
    ("",False,DARK),
    ("DETECTION SHAPE",                True,  DARK),
    (f"  Width  : {np.mean(det_ws):.1f} ± {np.std(det_ws):.1f} px",False,DARK),
    (f"  Height : {np.mean(det_hs):.1f} ± {np.std(det_hs):.1f} px",False,DARK),
    (f"  AR     : {np.mean(det_ars):.3f} ± {np.std(det_ars):.3f}  (always ~1.0)",False,MID),
    ("",False,DARK),
    ("GT FACE SHAPE",                  True,  DARK),
    (f"  Width  : {np.mean(gt_ws):.1f} ± {np.std(gt_ws):.1f} px",False,DARK),
    (f"  Height : {np.mean(gt_hs):.1f} ± {np.std(gt_hs):.1f} px",False,DARK),
    (f"  AR     : {np.mean(gt_ars):.3f} ± {np.std(gt_ars):.3f}  (faces ~0.80)",False,MID),
    ("",False,DARK),
    ("KEY INSIGHT",                    True,  (0,0,180)),
    (f"  Haar window = square (AR 1.0)",False,(0,0,180)),
    (f"  GT faces = portrait (AR 0.80)",False,(0,0,180)),
    (f"  Small faces (<50px) mostly missed",False,(0,0,180)),
]
for li,(line,bold,col) in enumerate(lines):
    cv2.putText(canvas, line, (ix+6, iy+18+li*20),
                fontB if bold else font,
                0.40 if bold else 0.38,
                col, 1, cv2.LINE_AA)

# ── Save ─────────────────────────────────────────────────
cv2.imwrite(OUT_PATH, canvas)
print(f"\n  Shape analysis saved → {OUT_PATH}")
print(f"\n  STEP 3 complete ✓")
print(f"  Next: Step 4 — apply optimisations and compare FPS + accuracy\n")