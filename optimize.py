"""
STEP 4 — Optimise Haar Cascade: FPS + Accuracy Comparison
==========================================================
Reads your val images directly (same 200-image cap as Step 2).
Runs 6 variants and records FPS + Accuracy for each:

  Baseline  — haarcascade_frontalface_default, scaleFactor=1.1, minN=5
  Opt-1     — Downscale 0.5×  (process at half resolution)
  Opt-2     — Fast params     (scaleFactor=1.2, minNeighbors=3)
  Opt-3     — Frame-skip ×3   (detect every 3rd frame, reuse otherwise)
  Opt-4     — ROI Tracking    (cheap ROI confirm, full re-detect every 8)
  Opt-5     — Combined        (Opt-1 + Opt-2 + Opt-3 stacked)

Saves:
  step4_output/results.json          full before/after benchmark
  step4_output/annotated_before/     8 sample frames (baseline)
  step4_output/annotated_after/      8 sample frames (best opt)

Run:
    python step4_optimize.py
"""

import cv2
import os
import json
import time
import numpy as np
from pathlib import Path

# ── EDIT THIS ─────────────────────────────────────────────
DATASET_ROOT = r"datasets/face"
# ──────────────────────────────────────────────────────────

SPLIT      = "val"
IOU_THRESH = 0.4
MAX_IMAGES = 200
WARMUP     = 10
SAVE_N     = 8
OUT_DIR    = "step4_output"
BEFORE_DIR = os.path.join(OUT_DIR, "annotated_before")
AFTER_DIR  = os.path.join(OUT_DIR, "annotated_after")

for d in [OUT_DIR, BEFORE_DIR, AFTER_DIR]:
    os.makedirs(d, exist_ok=True)

IMG_DIR = Path(DATASET_ROOT) / "images" / SPLIT
LBL_DIR = Path(DATASET_ROOT) / "labels" / SPLIT

# ── Helpers ───────────────────────────────────────────────
def yolo_to_xywh(cx, cy, w, h, iw, ih):
    pw = w*iw;  ph = h*ih
    return int(cx*iw - pw/2), int(cy*ih - ph/2), int(pw), int(ph)

def load_gt(lbl_path, iw, ih):
    boxes = []
    if Path(lbl_path).exists():
        with open(lbl_path) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    cx,cy,w,h = map(float, p[1:5])
                    boxes.append(yolo_to_xywh(cx,cy,w,h,iw,ih))
    return boxes

def iou(a, b):
    ax1,ay1,aw,ah = a;  bx1,by1,bw,bh = b
    ix = max(0, min(ax1+aw,bx1+bw) - max(ax1,bx1))
    iy = max(0, min(ay1+ah,by1+bh) - max(ay1,by1))
    inter = ix*iy;  union = aw*ah+bw*bh-inter
    return inter/union if union>0 else 0.0

def match_boxes(gt_boxes, det_boxes):
    if not gt_boxes and not det_boxes: return 0,0,0
    if not det_boxes: return 0,0,len(gt_boxes)
    if not gt_boxes:  return 0,len(det_boxes),0
    mat = np.array([[iou(g,d) for d in det_boxes] for g in gt_boxes])
    mg, md = set(), set()
    while mat.max() >= IOU_THRESH:
        gi,di = np.unravel_index(mat.argmax(), mat.shape)
        mg.add(gi); md.add(di)
        mat[gi,:] = -1; mat[:,di] = -1
    tp = len(mg)
    return tp, len(det_boxes)-len(md), len(gt_boxes)-len(mg)

def calc_metrics(tp, fp, fn):
    pr = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rc = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1 = 2*pr*rc/(pr+rc) if (pr+rc)>0 else 0.0
    return round(pr,4), round(rc,4), round(f1,4)

# ── Load dataset ──────────────────────────────────────────
print("="*60)
print("  STEP 4 — Optimisation Benchmark")
print("="*60)

exts  = {".jpg",".jpeg",".png",".bmp"}
paths = sorted(p for p in IMG_DIR.iterdir()
               if p.suffix.lower() in exts)[:MAX_IMAGES]

print(f"\n  Loading {len(paths)} val images ...")
dataset = []
for p in paths:
    img = cv2.imread(str(p))
    if img is None: continue
    ih,iw = img.shape[:2]
    gt = load_gt(LBL_DIR/(p.stem+".txt"), iw, ih)
    dataset.append({"fname":p.name,"img":img,"gt":gt,"h":ih,"w":iw})

frames  = [d["img"] for d in dataset]
gts     = [d["gt"]  for d in dataset]
print(f"  Loaded: {len(dataset)} images | "
      f"{sum(len(g) for g in gts)} GT faces")

# ── Detectors ─────────────────────────────────────────────
haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
haar_alt = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

def raw_detect(img, cascade=None, sf=1.1, mn=5, ms=(30,30)):
    if cascade is None: cascade = haar
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d = cascade.detectMultiScale(gray, scaleFactor=sf,
                                  minNeighbors=mn, minSize=ms)
    return [list(map(int,b)) for b in d] if len(d) else []

# ── Optimisation state (reset between runs) ───────────────
_state = {"fs_last":[], "fs_idx":0,
          "roi_tracked":[], "roi_age":0,
          "cb_last":[], "cb_idx":0}

def reset_state():
    _state.update({"fs_last":[], "fs_idx":0,
                   "roi_tracked":[], "roi_age":0,
                   "cb_last":[], "cb_idx":0})

def opt1_downscale(img):
    h,w = img.shape[:2]
    small = cv2.resize(img,(w//2, h//2))
    raw = raw_detect(small, haar_alt, sf=1.1, mn=5, ms=(15,15))
    return [[x*2,y*2,bw*2,bh*2] for x,y,bw,bh in raw]

def opt2_fast_params(img):
    return raw_detect(img, haar_alt, sf=1.2, mn=3, ms=(30,30))

def opt3_frame_skip(img):
    _state["fs_idx"] += 1
    if _state["fs_idx"] % 3 == 0:
        _state["fs_last"] = raw_detect(img, haar_alt)
    return _state["fs_last"]

def opt4_roi_tracking(img):
    h,w = img.shape[:2]
    _state["roi_age"] += 1
    if not _state["roi_tracked"] or _state["roi_age"] >= 8:
        _state["roi_tracked"] = raw_detect(img, haar_alt)
        _state["roi_age"] = 0
        return _state["roi_tracked"]
    pad = 40; confirmed = []
    for (x,y,bw,bh) in _state["roi_tracked"]:
        rx1=max(0,x-pad); ry1=max(0,y-pad)
        rx2=min(w,x+bw+pad); ry2=min(h,y+bh+pad)
        if rx2 <= rx1 or ry2 <= ry1:
            confirmed.append([x,y,bw,bh])
            continue
        roi = img[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            confirmed.append([x,y,bw,bh])
            continue
        sub = raw_detect(roi, haar_alt, ms=(20,20))
        if sub:
            sx,sy,sw,sh = sub[0]
            confirmed.append([rx1+sx,ry1+sy,sw,sh])
        else:
            confirmed.append([x,y,bw,bh])
    _state["roi_tracked"] = confirmed
    return confirmed

def opt5_combined(img):
    _state["cb_idx"] += 1
    if _state["cb_idx"] % 3 != 0:
        return _state["cb_last"]
    h,w = img.shape[:2]
    small = cv2.resize(img,(w//2,h//2))
    raw = raw_detect(small, haar_alt, sf=1.2, mn=3, ms=(15,15))
    _state["cb_last"] = [[x*2,y*2,bw*2,bh*2] for x,y,bw,bh in raw]
    return _state["cb_last"]

# ── Benchmark runner ──────────────────────────────────────
def benchmark(fn, label):
    reset_state()
    # warmup
    for img in frames[:WARMUP]:
        fn(img)
    reset_state()

    t0   = time.perf_counter()
    dets = [fn(img) for img in frames]
    elapsed = time.perf_counter() - t0

    fps = len(frames)/elapsed
    ms  = elapsed/len(frames)*1000

    tp_t=fp_t=fn_t=0
    for gt,det in zip(gts,dets):
        tp,fp,fn = match_boxes(gt,det)
        tp_t+=tp; fp_t+=fp; fn_t+=fn

    pr,rc,f1 = calc_metrics(tp_t,fp_t,fn_t)

    print(f"\n  [{label}]")
    print(f"    FPS={fps:.2f}  ms/f={ms:.2f}  "
          f"P={pr:.4f}  R={rc:.4f}  F1={f1:.4f}  "
          f"TP={tp_t} FP={fp_t} FN={fn_t}")

    return {
        "fps":          round(fps,2),
        "ms_per_frame": round(ms,2),
        "TP":tp_t, "FP":fp_t, "FN":fn_t,
        "precision":pr, "recall":rc, "f1":f1,
        "dets": dets
    }

# ── Run all variants ──────────────────────────────────────
print("\n── BASELINE ─────────────────────────────────────────")
r_base = benchmark(lambda img: raw_detect(img), "Baseline HaarDefault")

print("\n── OPTIMISATIONS ────────────────────────────────────")
r_o1 = benchmark(opt1_downscale,    "Opt-1  Downscale 0.5×")
r_o2 = benchmark(opt2_fast_params,  "Opt-2  Fast params (sf=1.2, mn=3)")
r_o3 = benchmark(opt3_frame_skip,   "Opt-3  Frame-skip ×3")
r_o4 = benchmark(opt4_roi_tracking, "Opt-4  ROI Tracking")
r_o5 = benchmark(opt5_combined,     "Opt-5  Combined (1+2+3)")

# ── Summary table ─────────────────────────────────────────
rows = [
    ("Baseline",      r_base),
    ("Opt-1 Downscale", r_o1),
    ("Opt-2 FastParams", r_o2),
    ("Opt-3 FrameSkip",  r_o3),
    ("Opt-4 ROI Track",  r_o4),
    ("Opt-5 Combined",   r_o5),
]

base_fps = r_base["fps"]
print("\n── SUMMARY ──────────────────────────────────────────")
print(f"  {'Method':<22} {'FPS':>7} {'Speedup':>8} "
      f"{'Prec':>7} {'Rec':>7} {'F1':>7}")
print("  " + "─"*62)
for name,r in rows:
    sx = r["fps"]/base_fps
    print(f"  {name:<22} {r['fps']:>7.2f} {sx:>7.1f}x "
          f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f}")

best_fps = max(rows, key=lambda x: x[1]["fps"])
best_f1  = max(rows, key=lambda x: x[1]["f1"])
print(f"\n  Best FPS : {best_fps[0]} → {best_fps[1]['fps']:.2f} FPS "
      f"({best_fps[1]['fps']/base_fps:.1f}× speedup)")
print(f"  Best F1  : {best_f1[0]} → {best_f1[1]['f1']:.4f}")

# ── Save annotated samples ────────────────────────────────
def save_samples(result_dets, out_dir, tag):
    for i in range(min(SAVE_N, len(dataset))):
        vis = dataset[i]["img"].copy()
        for x,y,w,h in dataset[i]["gt"]:
            cv2.rectangle(vis,(x,y),(x+w,y+h),(0,210,0),2)
        for x,y,w,h in result_dets[i]:
            cv2.rectangle(vis,(x,y),(x+w,y+h),(30,30,220),2)
        cv2.rectangle(vis,(8,8),(22,22),(0,210,0),-1)
        cv2.putText(vis,"GT",(26,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,210,0),1)
        cv2.rectangle(vis,(8,28),(22,42),(30,30,220),-1)
        cv2.putText(vis,"Det",(26,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(30,30,220),1)
        cv2.imwrite(os.path.join(out_dir,
                    f"s{i:02d}_{tag}_{dataset[i]['fname']}"), vis)

print(f"\n  Saving annotated samples ...")
save_samples(r_base["dets"], BEFORE_DIR, "baseline")
# find best F1 opt for after samples
save_samples(r_o1["dets"],   AFTER_DIR,  "opt1_downscale")
print(f"  Before → {BEFORE_DIR}/")
print(f"  After  → {AFTER_DIR}/")

# ── Save results JSON ─────────────────────────────────────
def strip_dets(r):
    """Remove per-image dets list before saving to JSON (too large)."""
    out = {k:v for k,v in r.items() if k != "dets"}
    return out

results_json = {
    "meta": {
        "dataset_root":  str(Path(DATASET_ROOT).resolve()),
        "split":         SPLIT,
        "n_images":      len(dataset),
        "n_gt_faces":    sum(len(g) for g in gts),
        "iou_threshold": IOU_THRESH,
        "max_images":    MAX_IMAGES,
    },
    "before": {
        "HaarDefault_baseline": strip_dets(r_base)
    },
    "after": {
        "Opt1_Downscale_0.5x": {
            "description": "Resize to 0.5x, detect, scale boxes x2",
            **strip_dets(r_o1)
        },
        "Opt2_FastParams": {
            "description": "scaleFactor=1.2, minNeighbors=3",
            **strip_dets(r_o2)
        },
        "Opt3_FrameSkip_x3": {
            "description": "Detect every 3rd frame, reuse otherwise",
            **strip_dets(r_o3)
        },
        "Opt4_ROI_Tracking": {
            "description": "ROI confirm between full re-detects every 8 frames",
            **strip_dets(r_o4)
        },
        "Opt5_Combined": {
            "description": "Downscale + FastParams + FrameSkip stacked",
            **strip_dets(r_o5)
        },
    },
    "speedups": {
        name: round(r["fps"]/base_fps, 2)
        for name,r in rows[1:]
    }
}

json_path = os.path.join(OUT_DIR, "results.json")
with open(json_path,"w") as f:
    json.dump(results_json, f, indent=2)

print(f"\n  Results saved → {json_path}")
print(f"\n  STEP 4 complete ✓")
print(f"  Share the summary table above → then Step 5 (final visualization)\n")