"""
STEP 5 — Final Results Dashboard + final.json
==============================================
Reads step4_output/results.json and produces:

  step5_output/dashboard.png   — full visual comparison
  step5_output/final.json      — clean before/after summary

Run:
    python step5_final.py
"""

import cv2
import json
import os
import numpy as np

IN_JSON  = "step4_output/results.json"
OUT_DIR  = "step5_output"
DASH_OUT = os.path.join(OUT_DIR, "dashboard.png")
JSON_OUT = os.path.join(OUT_DIR, "final.json")

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 58)
print("  STEP 5 — Final Dashboard + final.json")
print("=" * 58)

with open(IN_JSON) as f:
    data = json.load(f)

meta   = data["meta"]
before = data["before"]["HaarDefault_baseline"]
after  = data["after"]

# ── Build final.json ──────────────────────────────────────
best_fps_name = max(after, key=lambda k: after[k]["fps"])
best_f1_name  = max(after, key=lambda k: after[k]["f1"])
best_bal_name = max(after, key=lambda k: after[k]["fps"] * after[k]["f1"])

final = {
    "dataset": {
        "root":          meta["dataset_root"],
        "split":         meta["split"],
        "images":        meta["n_images"],
        "gt_faces":      meta["n_gt_faces"],
        "iou_threshold": meta["iou_threshold"],
    },
   "before": {
    "model":      "HaarCascade haarcascade_frontalface_default.xml",
    "params":     "scaleFactor=1.1  minNeighbors=5  minSize=(30,30)",
    "fps":        before["fps"],
    "ms_per_frame": before["ms_per_frame"],
    "precision":  before["precision"],
    "recall":     before["recall"],
    "f1":         before["f1"],
    "accuracy":   round(before["TP"] / (before["TP"] + before["FP"] + before["FN"]), 4),
    "TP":         before["TP"],
    "FP":         before["FP"],
    "FN":         before["FN"],
},
    "after": {},
    "best": {
        "highest_fps":     {"method": best_fps_name,
                            "fps":    after[best_fps_name]["fps"],
                            "f1":     after[best_fps_name]["f1"],
                            "speedup":round(after[best_fps_name]["fps"]/before["fps"],2)},
        "highest_f1":      {"method": best_f1_name,
                            "fps":    after[best_f1_name]["fps"],
                            "f1":     after[best_f1_name]["f1"],
                            "speedup":round(after[best_f1_name]["fps"]/before["fps"],2)},
        "best_balanced":   {"method": best_bal_name,
                            "fps":    after[best_bal_name]["fps"],
                            "f1":     after[best_bal_name]["f1"],
                            "speedup":round(after[best_bal_name]["fps"]/before["fps"],2)},
    }
}

for name, r in after.items():
    acc = r["TP"] / (r["TP"] + r["FP"] + r["FN"])
    final["after"][name] = {
        "description": r.get("description", ""),
        "fps":         r["fps"],
        "ms_per_frame":r["ms_per_frame"],
        "speedup_vs_baseline": round(r["fps"] / before["fps"], 2),
        "precision":   r["precision"],
        "recall":      r["recall"],
        "f1":          r["f1"],
        "accuracy":    round(acc, 4),
        "TP":          r["TP"],
        "FP":          r["FP"],
        "FN":          r["FN"],
        "f1_delta":    round(r["f1"] - before["f1"], 4),
        "fps_delta":   round(r["fps"] - before["fps"], 2),
    }

with open(JSON_OUT, "w") as f:
    json.dump(final, f, indent=2)
print(f"\n  final.json saved → {JSON_OUT}")
print("\nAccuracy Summary")
print("-" * 40)

before_acc = before["TP"] / (before["TP"] + before["FP"] + before["FN"])
print(f"Baseline Accuracy : {before_acc:.4f}")

for name, r in after.items():
    acc = r["TP"] / (r["TP"] + r["FP"] + r["FN"])
    print(f"{name:20s} : {acc:.4f}")

# ── Canvas ────────────────────────────────────────────────
CW, CH = 1500, 1020
canvas  = np.full((CH, CW, 3), 245, dtype=np.uint8)
font    = cv2.FONT_HERSHEY_SIMPLEX
fontB   = cv2.FONT_HERSHEY_DUPLEX

DARK  = (30,  30,  30)
MID   = (110, 110, 110)
LIGHT = (190, 190, 190)
BLUE  = (180, 100,  30)   # BGR
GREEN = ( 40, 160,  40)
RED   = ( 40,  40, 200)
GOLD  = ( 20, 160, 200)
TEAL  = (160, 130,  20)

def txt(s, x, y, sc=0.42, col=DARK, bold=False, thickness=1):
    cv2.putText(canvas, str(s), (x,y),
                fontB if bold else font, sc, col, thickness, cv2.LINE_AA)

def hrule(y, x1=0, x2=None, col=LIGHT, t=1):
    cv2.line(canvas, (x1,y), (x2 or CW, y), col, t)

def vrule(x, y1=0, y2=None, col=LIGHT, t=1):
    cv2.line(canvas, (x,y1), (x, y2 or CH), col, t)

# ── Title ─────────────────────────────────────────────────
cv2.rectangle(canvas, (0,0), (CW,52), (40,40,40), -1)
txt("Haar Cascade Face Detection — Optimisation Results Dashboard",
    14, 33, 0.70, (230,230,230), bold=True)
txt(f"Dataset: {meta['n_images']} images | {meta['n_gt_faces']} GT faces | "
    f"IoU≥{meta['iou_threshold']} | split={meta['split']}",
    CW-480, 33, 0.38, (180,180,180))

# ── Section labels ────────────────────────────────────────
ROWS = [
    ("Baseline",       before,              (70,70,70)),
    ("Opt-1 Downscale",after["Opt1_Downscale_0.5x"],  TEAL),
    ("Opt-2 FastParams",after["Opt2_FastParams"],      BLUE),
    ("Opt-3 FrameSkip", after["Opt3_FrameSkip_x3"],   MID),
    ("Opt-4 ROI Track", after["Opt4_ROI_Tracking"],   MID),
    ("Opt-5 Combined",  after["Opt5_Combined"],        GREEN),
]

base_fps = before["fps"]
base_f1  = before["f1"]

# ═══════════════════════════════════════════════════════
# SECTION A — FPS BAR CHART  (top-left)
# ═══════════════════════════════════════════════════════
AX, AY, AW, AH = 20, 70, 680, 280
cv2.rectangle(canvas,(AX,AY),(AX+AW,AY+AH),(220,220,220),1)
txt("FPS by Method  (higher = faster)", AX+10, AY+20, 0.50, DARK, bold=True)

max_fps = max(r["fps"] for _,r,_ in ROWS)
BAR_X   = AX + 150
BAR_W   = AW - 170
BAR_H   = 30
GAP     = 8
START_Y = AY + 36

for i,(name,r,col) in enumerate(ROWS):
    by    = START_Y + i*(BAR_H+GAP)
    blen  = int(r["fps"]/max_fps * BAR_W)
    # bar
    cv2.rectangle(canvas,(BAR_X,by),(BAR_X+blen,by+BAR_H),col,-1)
    cv2.rectangle(canvas,(BAR_X,by),(BAR_X+BAR_W,by+BAR_H),LIGHT,1)
    # label left
    txt(name, AX+4, by+20, 0.38, DARK)
    # fps value inside/right of bar
    val_x = BAR_X + blen + 4
    speedup = r["fps"]/base_fps
    label = f"{r['fps']:.1f} fps  ({speedup:.1f}×)"
    txt(label, val_x, by+20, 0.38, DARK)

# ═══════════════════════════════════════════════════════
# SECTION B — F1 BAR CHART  (top-right)
# ═══════════════════════════════════════════════════════
BX, BY, BW, BH2 = CW-700, 70, 680, 280
cv2.rectangle(canvas,(BX,BY),(BX+BW,BY+BH2),(220,220,220),1)
txt("F1 Score by Method  (higher = better accuracy)", BX+10, BY+20, 0.50, DARK, bold=True)

max_f1 = max(r["f1"] for _,r,_ in ROWS)
BBAR_X  = BX + 155
BBAR_W  = BW - 175

for i,(name,r,col) in enumerate(ROWS):
    by   = START_Y + i*(BAR_H+GAP)
    blen = int(r["f1"]/max(max_f1,0.01) * BBAR_W)
    cv2.rectangle(canvas,(BBAR_X,by),(BBAR_X+blen,by+BAR_H),col,-1)
    cv2.rectangle(canvas,(BBAR_X,by),(BBAR_X+BBAR_W,by+BAR_H),LIGHT,1)
    txt(name, BX+4, by+20, 0.38, DARK)
    delta = r["f1"] - base_f1
    delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
    delta_col = GREEN if delta >= 0 else RED
    txt(f"{r['f1']:.4f}  ({delta_str})", BBAR_X+blen+4, by+20, 0.38, delta_col)

# ═══════════════════════════════════════════════════════
# SECTION C — Precision / Recall breakdown table
# ═══════════════════════════════════════════════════════
CX2, CY2 = 20, 370
txt("Precision · Recall · F1  Breakdown", CX2+10, CY2+20, 0.52, DARK, bold=True)

# Table header
cols_x = [CX2+10, CX2+160, CX2+240, CX2+330, CX2+420, CX2+510, CX2+590, CX2+670, CX2+760]
headers = ["Method","FPS","ms/f","Speedup","Precision","Recall","F1","TP | FP | FN",""]
cv2.rectangle(canvas,(CX2,CY2+28),(CX2+860,CY2+48),(60,60,60),-1)
for hx,h in zip(cols_x,headers):
    txt(h, hx, CY2+43, 0.38, (220,220,220))

ROW_H = 28
for i,(name,r,col) in enumerate(ROWS):
    ry  = CY2 + 52 + i*ROW_H
    bg  = (235,235,235) if i%2==0 else (245,245,245)
    cv2.rectangle(canvas,(CX2,ry),(CX2+860,ry+ROW_H),bg,-1)
    # highlight best F1 row
    if name == "Opt-2 FastParams":
        cv2.rectangle(canvas,(CX2,ry),(CX2+860,ry+ROW_H),(220,240,220),-1)
    vals = [
        name,
        f"{r['fps']:.1f}",
        f"{r['ms_per_frame']:.1f}",
        f"{r['fps']/base_fps:.1f}×",
        f"{r['precision']:.4f}",
        f"{r['recall']:.4f}",
        f"{r['f1']:.4f}",
        f"{r['TP']} | {r['FP']} | {r['FN']}",
    ]
    for vx,v in zip(cols_x,vals):
        c = GREEN if (v.startswith("+") or
                      (name=="Opt-2 FastParams" and v==f"{r['f1']:.4f}")) else DARK
        txt(v, vx, ry+19, 0.38, c)

# ═══════════════════════════════════════════════════════
# SECTION D — Speed vs Accuracy scatter
# ═══════════════════════════════════════════════════════
DX, DY, DW, DH = 20, 590, 440, 340
cv2.rectangle(canvas,(DX,DY),(DX+DW,DY+DH),(220,220,220),1)
txt("Speed vs Accuracy  (ideal = top-right)", DX+10, DY+20, 0.48, DARK, bold=True)

# axes
PLOT_X1 = DX+50; PLOT_Y1 = DY+30
PLOT_X2 = DX+DW-20; PLOT_Y2 = DY+DH-30
cv2.line(canvas,(PLOT_X1,PLOT_Y2),(PLOT_X2,PLOT_Y2),DARK,1)
cv2.line(canvas,(PLOT_X1,PLOT_Y1),(PLOT_X1,PLOT_Y2),DARK,1)
txt("FPS →", PLOT_X1+(PLOT_X2-PLOT_X1)//2-15, PLOT_Y2+18, 0.36, MID)
txt("F1", PLOT_X1-30, PLOT_Y1+(PLOT_Y2-PLOT_Y1)//2, 0.36, MID)

all_fps_s = [r["fps"] for _,r,_ in ROWS]
all_f1_s  = [r["f1"]  for _,r,_ in ROWS]
fps_max_s = max(all_fps_s); fps_min_s = min(all_fps_s)
f1_max_s  = max(all_f1_s);  f1_min_s  = min(all_f1_s)
f1_range  = max(f1_max_s - f1_min_s, 0.05)
fps_range = max(fps_max_s - fps_min_s, 1)

for i,(name,r,col) in enumerate(ROWS):
    sx = PLOT_X1 + int((r["fps"]  - fps_min_s) / fps_range * (PLOT_X2-PLOT_X1))
    sy = PLOT_Y2 - int((r["f1"]   - f1_min_s)  / f1_range  * (PLOT_Y2-PLOT_Y1))
    cv2.circle(canvas,(sx,sy),9,col,-1)
    cv2.circle(canvas,(sx,sy),9,DARK,1)
    short = name.replace("Opt-","O").replace(" ","")
    txt(short, sx-18, sy-13, 0.33, DARK)

# ═══════════════════════════════════════════════════════
# SECTION E — Key insights text
# ═══════════════════════════════════════════════════════
EX, EY = 480, 590
cv2.rectangle(canvas,(EX,EY),(EX+540,EY+340),(220,220,220),1)
txt("Key Insights", EX+10, EY+20, 0.52, DARK, bold=True)

insights = [
    ("Baseline",
     f"F1={before['f1']:.3f}  FPS={before['fps']:.0f}  →  Reference point"),
    ("",""),
    ("Opt-1 Downscale 0.5×",
     f"FPS ↑{after['Opt1_Downscale_0.5x']['fps']/base_fps:.1f}×  "
     f"but F1 drops (smaller faces missed at 0.5×)"),
    ("Opt-2 FastParams ✓ BEST F1",
     f"F1={after['Opt2_FastParams']['f1']:.3f}  FPS={after['Opt2_FastParams']['fps']:.0f}  "
     f"sf=1.2 mn=3 gives best accuracy"),
    ("Opt-3 Frame-skip",
     "Big FPS gain but accuracy collapses on static images"),
    ("Opt-4 ROI Tracking",
     "FP explodes — designed for video, not still images"),
    ("Opt-5 Combined ✓ BEST FPS",
     f"FPS={after['Opt5_Combined']['fps']:.0f} ({after['Opt5_Combined']['fps']/base_fps:.1f}×)  "
     f"F1 drops to {after['Opt5_Combined']['f1']:.3f}"),
    ("",""),
    ("Root cause of low recall:",
     "Haar square window ≠ portrait faces (GT AR~0.80)"),
    ("","Small faces (<30px) invisible to minSize=(30,30)"),
    ("",""),
    ("Recommendation:",
     "Use Opt-2 for accuracy-critical tasks"),
    ("",
     "Use Opt-5 for real-time video streams"),
]

for li,(label,val) in enumerate(insights):
    y_pos = EY + 40 + li*20
    if label:
        txt(label, EX+10, y_pos, 0.37, TEAL if "✓" in label else DARK, bold="✓" in label)
    if val:
        txt(val, EX+16, y_pos+13, 0.35, MID)

# ═══════════════════════════════════════════════════════
# SECTION F — Before / After highlight cards
# ═══════════════════════════════════════════════════════
FX, FY = 1040, 590
card_w = 200; card_h = 155; gap = 10

# BEFORE card
cv2.rectangle(canvas,(FX,FY),(FX+card_w,FY+card_h),(200,200,200),-1)
cv2.rectangle(canvas,(FX,FY),(FX+card_w,FY+24),(80,80,80),-1)
txt("BEFORE (Baseline)", FX+5, FY+17, 0.38, (230,230,230), bold=True)
lines_b = [
    f"FPS    :  {before['fps']:.1f}",
    f"Prec   :  {before['precision']:.4f}",
    f"Recall :  {before['recall']:.4f}",
    f"F1     :  {before['f1']:.4f}",
    f"TP={before['TP']}  FP={before['FP']}",
    f"FN={before['FN']}",
]
for li,line in enumerate(lines_b):
    txt(line, FX+8, FY+38+li*18, 0.39, DARK)

# BEST F1 card
opt_f1 = after[best_f1_name]
FX2 = FX + card_w + gap
cv2.rectangle(canvas,(FX2,FY),(FX2+card_w,FY+card_h),(210,235,210),-1)
cv2.rectangle(canvas,(FX2,FY),(FX2+card_w,FY+24),(40,120,40),-1)
txt("BEST F1 (Opt-2)", FX2+5, FY+17, 0.38, (230,230,230), bold=True)
lines_f1 = [
    f"FPS    :  {opt_f1['fps']:.1f}",
    f"Prec   :  {opt_f1['precision']:.4f}",
    f"Recall :  {opt_f1['recall']:.4f}",
    f"F1     :  {opt_f1['f1']:.4f}",
    f"TP={opt_f1['TP']}  FP={opt_f1['FP']}",
    f"FN={opt_f1['FN']}",
]
for li,line in enumerate(lines_f1):
    txt(line, FX2+8, FY+38+li*18, 0.39, DARK)

# BEST FPS card  (below)
opt_fps = after[best_fps_name]
FY2 = FY + card_h + gap
cv2.rectangle(canvas,(FX,FY2),(FX+card_w,FY2+card_h),(210,225,240),-1)
cv2.rectangle(canvas,(FX,FY2),(FX+card_w,FY2+24),(30,80,160),-1)
txt("BEST FPS (Opt-5)", FX+5, FY2+17, 0.38, (230,230,230), bold=True)
lines_fps = [
    f"FPS    :  {opt_fps['fps']:.1f}",
    f"Speedup:  {opt_fps['fps']/base_fps:.1f}×",
    f"Prec   :  {opt_fps['precision']:.4f}",
    f"Recall :  {opt_fps['recall']:.4f}",
    f"F1     :  {opt_fps['f1']:.4f}",
    f"TP={opt_fps['TP']}  FP={opt_fps['FP']}",
]
for li,line in enumerate(lines_fps):
    txt(line, FX+8, FY2+38+li*18, 0.39, DARK)

# ── Save ─────────────────────────────────────────────────
cv2.imwrite(DASH_OUT, canvas)
print(f"  Dashboard saved  → {DASH_OUT}")
print(f"\n  STEP 5 complete ✓  — all done!\n")
print(f"  Files produced:")
print(f"    step5_output/dashboard.png   ← full visual comparison")
print(f"    step5_output/final.json      ← before/after JSON summary")