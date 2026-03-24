"""
STEP 1 — Load & verify your YOLO-format face dataset
=====================================================
Reads images/train, images/val, labels/train, labels/val
Confirms label format, prints stats, saves 6 annotated
sample images so you can visually confirm GT boxes.

Run:
    python step1_verify_dataset.py

Edit DATASET_ROOT below to point to your dataset folder.
"""

import cv2
import os
import json
from pathlib import Path

# ── EDIT THIS ─────────────────────────────────────────────
DATASET_ROOT = r"datasets/face"          # relative or absolute path
# e.g. on Windows: r"C:\Users\you\datasets\face"
# e.g. on Linux:   "/home/you/datasets/face"
# ──────────────────────────────────────────────────────────

OUT_DIR = "step1_output"
os.makedirs(OUT_DIR, exist_ok=True)


def yolo_to_xywh(cx, cy, w, h, img_w, img_h):
    """Convert YOLO normalised cx,cy,w,h → pixel x,y,w,h (top-left)."""
    pw = w  * img_w
    ph = h  * img_h
    px = cx * img_w - pw / 2
    py = cy * img_h - ph / 2
    return int(px), int(py), int(pw), int(ph)


def load_split(split):
    """
    Load all (image_path, label_path, boxes) for a split (train/val).
    Returns list of dicts.
    """
    img_dir = Path(DATASET_ROOT) / "images" / split
    lbl_dir = Path(DATASET_ROOT) / "labels" / split

    if not img_dir.exists():
        print(f"  [!] Image dir not found: {img_dir}")
        return []

    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    items  = []
    no_lbl = 0

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in exts:
            continue

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        boxes    = []

        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls  = int(parts[0])
                        vals = list(map(float, parts[1:5]))
                        boxes.append((cls, *vals))   # (cls, cx, cy, w, h)
        else:
            no_lbl += 1

        items.append({
            "img_path": str(img_path),
            "lbl_path": str(lbl_path),
            "boxes":    boxes,         # YOLO normalised
        })

    if no_lbl:
        print(f"  [warn] {no_lbl} images have no matching label file")

    return items


def print_stats(split, items):
    n_imgs      = len(items)
    n_with_face = sum(1 for it in items if it["boxes"])
    n_faces     = sum(len(it["boxes"]) for it in items)
    classes     = sorted({b[0] for it in items for b in it["boxes"]})

    # Read one image to get resolution
    sample_img  = None
    for it in items:
        img = cv2.imread(it["img_path"])
        if img is not None:
            sample_img = img
            sample_res = f"{img.shape[1]}×{img.shape[0]}"
            break

    print(f"\n  [{split.upper()}]")
    print(f"    Images total    : {n_imgs}")
    print(f"    With faces      : {n_with_face}")
    print(f"    Empty (no face) : {n_imgs - n_with_face}")
    print(f"    Total GT boxes  : {n_faces}")
    print(f"    Avg faces/img   : {n_faces/max(n_imgs,1):.2f}")
    print(f"    Class IDs found : {classes}  (0 = face in most YOLO face datasets)")
    if sample_img is not None:
        print(f"    Sample resolution: {sample_res}")

    # Box size stats (in pixels using sample resolution)
    if sample_img is not None and n_faces > 0:
        IH, IW = sample_img.shape[:2]
        ws, hs = [], []
        for it in items:
            img2 = cv2.imread(it["img_path"])
            if img2 is None:
                continue
            h2, w2 = img2.shape[:2]
            for (_, cx, cy, w, h) in it["boxes"]:
                _, _, pw, ph = yolo_to_xywh(cx, cy, w, h, w2, h2)
                ws.append(pw); hs.append(ph)
            if len(ws) > 500:   # fast estimate from first 500
                break

        import numpy as np
        ws_arr = __import__("numpy").array(ws)
        hs_arr = __import__("numpy").array(hs)
        print(f"    Box width  (px) : mean={ws_arr.mean():.1f}  "
              f"std={ws_arr.std():.1f}  "
              f"min={ws_arr.min():.0f}  max={ws_arr.max():.0f}")
        print(f"    Box height (px) : mean={hs_arr.mean():.1f}  "
              f"std={hs_arr.std():.1f}  "
              f"min={hs_arr.min():.0f}  max={hs_arr.max():.0f}")
        ar = ws_arr / hs_arr
        print(f"    Aspect ratio    : mean={ar.mean():.3f}  "
              f"std={ar.std():.3f}  (1.0 = square)")


def save_samples(split, items, n=6):
    """Draw GT boxes on n sample images and save them."""
    saved = 0
    for it in items:
        if saved >= n:
            break
        img = cv2.imread(it["img_path"])
        if img is None:
            continue
        IH, IW = img.shape[:2]
        for (cls, cx, cy, w, h) in it["boxes"]:
            px, py, pw, ph = yolo_to_xywh(cx, cy, w, h, IW, IH)
            cv2.rectangle(img, (px, py), (px+pw, py+ph), (0, 220, 0), 2)
            cv2.putText(img, f"face cls{cls}", (px, max(0, py-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)

        stem    = Path(it["img_path"]).stem
        outname = os.path.join(OUT_DIR, f"{split}_{saved:02d}_{stem}.jpg")
        cv2.imwrite(outname, img)
        saved += 1

    print(f"    Saved {saved} annotated samples → {OUT_DIR}/")


def main():
    print("=" * 55)
    print("  STEP 1 — Dataset Verification")
    print("=" * 55)
    print(f"\n  Dataset root: {Path(DATASET_ROOT).resolve()}")

    if not Path(DATASET_ROOT).exists():
        print(f"\n  [ERROR] Path not found: {Path(DATASET_ROOT).resolve()}")
        print("  Edit DATASET_ROOT at the top of this file.")
        return

    summary = {}

    for split in ["train", "val"]:
        items = load_split(split)
        if not items:
            print(f"\n  [{split.upper()}] — no images found, skipping.")
            continue
        print_stats(split, items)
        save_samples(split, items)
        summary[split] = {
            "n_images": len(items),
            "n_faces":  sum(len(it["boxes"]) for it in items),
        }

    # Save summary JSON for use in later steps
    summary_path = os.path.join(OUT_DIR, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"dataset_root": str(Path(DATASET_ROOT).resolve()),
                   "splits": summary}, f, indent=2)
    print(f"\n  Summary saved → {summary_path}")
    print("\n  STEP 1 complete ✓")
    print("  Share the output above — then we go to Step 2.\n")


if __name__ == "__main__":
    main()