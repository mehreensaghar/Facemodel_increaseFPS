"""
STEP 2 — Run Haar Cascade on your real val set
===============================================
- Loads val images + YOLO ground-truth labels
- Runs haarcascade_frontalface_default on every image
- Records FPS (wall-clock benchmark)
- Saves raw detections to step2_output/detections.json
- Saves 8 annotated sample frames (green=GT, red=detection)
- Prints a quick accuracy preview (P / R / F1 at IoU 0.4)

No model download needed — Haar ships inside opencv-python.

Run:
    python step2_run_detector.py
"""

import cv2
import os
import json
import time
import numpy as np
from pathlib import Path

# ── EDIT THIS ─────────────────────────────────────────────
DATASET_ROOT = r"datasets/face"     # same as Step 1
# ──────────────────────────────────────────────────────────

SPLIT      = "val"                  # we benchmark on val
IOU_THRESH = 0.4
WARMUP     = 20                     # frames to warm up timer
SAVE_N     = 8                      # annotated samples to save
OUT_DIR    = "step2_output"

os.makedirs(OUT_DIR, exist_ok=True)

IMG_DIR = Path(DATASET_ROOT) / "images" / SPLIT
LBL_DIR = Path(DATASET_ROOT) / "labels" / SPLIT


# ── Helpers ───────────────────────────────────────────────
def yolo_to_xywh(cx, cy, w, h, iw, ih):
    pw = w * iw;  ph = h * ih
    px = cx * iw - pw / 2
    py = cy * ih - ph / 2
    return int(px), int(py), int(pw), int(ph)


def load_gt(lbl_path, iw, ih):
    boxes = []
    if Path(lbl_path).exists():
        with open(lbl_path) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    cx, cy, w, h = map(float, p[1:5])
                    boxes.append(yolo_to_xywh(cx, cy, w, h, iw, ih))
    return boxes   # list of [x, y, w, h] in pixels


def iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1+aw, ay1+ah
    bx2, by2 = bx1+bw, by1+bh
    ix = max(0, min(ax2,bx2) - max(ax1,bx1))
    iy = max(0, min(ay2,by2) - max(ay1,by1))
    inter = ix * iy
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0


def match(gt_boxes, det_boxes, thresh=IOU_THRESH):
    if not gt_boxes and not det_boxes:
        return 0, 0, 0
    if not det_boxes:
        return 0, 0, len(gt_boxes)
    if not gt_boxes:
        return 0, len(det_boxes), 0

    mat = np.array([[iou(g, d) for d in det_boxes] for g in gt_boxes])
    mg, md = set(), set()
    while mat.max() >= thresh:
        gi, di = np.unravel_index(mat.argmax(), mat.shape)
        mg.add(gi); md.add(di)
        mat[gi, :] = -1; mat[:, di] = -1
    tp = len(mg)
    return tp, len(det_boxes)-len(md), len(gt_boxes)-len(mg)


# ── Load dataset ──────────────────────────────────────────
print("=" * 58)
print("  STEP 2 — Haar Cascade on Real Val Set")
print("=" * 58)

exts  = {".jpg", ".jpeg", ".png", ".bmp"}
paths = sorted(p for p in IMG_DIR.iterdir()
               if p.suffix.lower() in exts)

print(f"\n  Val images found : {len(paths)}")
print(f"  Loading images   ...")

dataset = []
for p in paths:
    img = cv2.imread(str(p))
    if img is None:
        continue
    ih, iw = img.shape[:2]
    gt = load_gt(LBL_DIR / (p.stem + ".txt"), iw, ih)
    dataset.append({"fname": p.name, "img": img, "gt": gt})

print(f"  Loaded           : {len(dataset)} images")
print(f"  Total GT faces   : {sum(len(d['gt']) for d in dataset)}")

# ── Build detector ────────────────────────────────────────
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
haar = cv2.CascadeClassifier(cascade_path)
print(f"\n  Model: Haar Cascade (haarcascade_frontalface_default.xml)")
print(f"  Parameters: scaleFactor=1.1  minNeighbors=5  minSize=(30,30)")


def detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d    = haar.detectMultiScale(
               gray,
               scaleFactor=1.1,
               minNeighbors=5,
               minSize=(30, 30))
    return [list(map(int, b)) for b in d] if len(d) else []


# ── Warm up ───────────────────────────────────────────────
print(f"\n  Warming up ({WARMUP} frames) ...")
for d in dataset[:WARMUP]:
    detect(d["img"])

# ── Timed benchmark ───────────────────────────────────────
print(f"  Running on {len(dataset)} images ...")
t0   = time.perf_counter()
all_dets = [detect(d["img"]) for d in dataset]
elapsed  = time.perf_counter() - t0

fps       = len(dataset) / elapsed
ms_frame  = elapsed / len(dataset) * 1000

print(f"\n  ── FPS Results ──────────────────────────────")
print(f"  Total time   : {elapsed:.2f}s")
print(f"  FPS          : {fps:.2f}")
print(f"  ms / frame   : {ms_frame:.2f}")

# ── Accuracy ──────────────────────────────────────────────
tp_total = fp_total = fn_total = 0
for d, dets in zip(dataset, all_dets):
    tp, fp, fn = match(d["gt"], dets)
    tp_total += tp; fp_total += fp; fn_total += fn

prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
rec  = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0

print(f"\n  ── Accuracy (IoU ≥ {IOU_THRESH}) ─────────────")
print(f"  TP={tp_total}  FP={fp_total}  FN={fn_total}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1        : {f1:.4f}")

# ── Save annotated samples ────────────────────────────────
print(f"\n  Saving {SAVE_N} annotated samples ...")
for i in range(min(SAVE_N, len(dataset))):
    vis = dataset[i]["img"].copy()
    for x, y, w, h in dataset[i]["gt"]:
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,210,0), 2)
    for x, y, w, h in all_dets[i]:
        cv2.rectangle(vis, (x,y), (x+w,y+h), (30,30,220), 2)
    # Legend
    cv2.rectangle(vis, (8,8),  (22,22), (0,210,0), -1)
    cv2.putText(vis, "GT",  (26, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,210,0), 1)
    cv2.rectangle(vis, (8,28), (22,42), (30,30,220), -1)
    cv2.putText(vis, "Det", (26, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30,30,220), 1)
    out = os.path.join(OUT_DIR, f"sample_{i:02d}_{dataset[i]['fname']}")
    cv2.imwrite(out, vis)

print(f"  Saved → {OUT_DIR}/sample_*.jpg")

# ── Save detections JSON ──────────────────────────────────
results = {
    "model":        "HaarCascade_default",
    "split":        SPLIT,
    "n_images":     len(dataset),
    "n_gt_faces":   sum(len(d["gt"]) for d in dataset),
    "fps":          round(fps, 2),
    "ms_per_frame": round(ms_frame, 2),
    "accuracy": {
        "iou_threshold": IOU_THRESH,
        "TP": tp_total, "FP": fp_total, "FN": fn_total,
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
    },
    "per_image": [
        {"fname": d["fname"],
         "gt":    d["gt"],
         "dets":  det}
        for d, det in zip(dataset, all_dets)
    ]
}

json_path = os.path.join(OUT_DIR, "detections.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"  Detections saved → {json_path}")
print(f"\n  STEP 2 complete ✓")
print(f"  Share the FPS + accuracy numbers above → then Step 3 (shape analysis)\n")