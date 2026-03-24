# Haar Cascade Face Detection — FPS Optimization Toolkit

A comprehensive benchmarking and optimization pipeline for **Haar Cascade face detection** that demonstrates 6 practical techniques to dramatically increase inference FPS while maintaining accuracy.

## Project Overview

This toolkit measures the performance of OpenCV's Haar Cascade detector on your own YOLO-format face dataset and explores **6 optimization strategies**:

| Strategy | Technique | FPS Boost | Accuracy Impact |
|----------|-----------|-----------|-----------------|
| **Baseline** | Standard params (scaleFactor=1.1, minNeighbors=5) | 1× | Reference |
| **Opt-1** | Downscaling (0.5× resolution) | ~2–4× | Minor loss |
| **Opt-2** | Fast params (scaleFactor=1.2, minNeighbors=3) | ~1.5–2× | Some loss |
| **Opt-3** | Frame-skipping (detect every 3rd frame) | ~3× | Video smoothing |
| **Opt-4** | ROI tracking (cheap confirm, re-detect every 8 frames) | ~2–3× | Context-aware |
| **Opt-5** | Combined (Opt-1 + Opt-2 + Opt-3 stacked) | ~6–8× | Best tradeoff |

## Pipeline Steps

Run the scripts in order to go from dataset verification to final optimized results:

### Step 1: Verify Dataset (`verify.py`)

```bash
python verify.py
```

- Loads your YOLO-format face dataset (images + labels)
- Validates label format and image integrity
- Generates 6 annotated sample images (visual ground-truth confirmation)
- Output: `step1_output/dataset_summary.json`

### Step 2: Run Baseline Detector (`haar.py`)

```bash
python haar.py
```

- Runs OpenCV's `haarcascade_frontalface_default.xml` on val images
- Measures wall-clock FPS
- Records all detections with IoU evaluation (ground-truth vs predictions)
- Outputs accuracy metrics: Precision, Recall, F1 @ IoU 0.4
- Output: `step2_output/detections.json` + 8 annotated frames

### Step 3: Shape Analysis (`shape_analysis.py`)

```bash
python shape_analysis.py
```

- Analyzes detection shape distribution (width, height, aspect ratio)
- Identifies size ranges where Haar performs poorly
- Generates 6-panel visualization dashboard
- Output: `step3_output/shape_analysis.png`

### Step 4: Optimize & Benchmark (`optimize.py`)

```bash
python optimize.py
```

- Tests all 6 optimization variants simultaneously
- Records FPS + accuracy for each approach
- Ranks variants by FPS gain vs accuracy retention
- Output: `step4_output/results.json` + annotated before/after samples

### Step 5: Final Dashboard (`final.py`)

```bash
python final.py
```

- Generates final comparison dashboard (visual + JSON)
- Identifies best strategy for FPS, best for F1, best balanced
- Outputs clean summary for reporting
- Output: `step5_output/final.json` + dashboard visualization

## Quick Start

### Requirements

```bash
pip install opencv-python numpy
```

### Setup Your Dataset

1. Place your YOLO-format face dataset in `datasets/face/`:

```
datasets/face/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

2. YOLO labels: `class_id center_x center_y width height` (normalized 0–1)

### Run Full Pipeline

```bash
python verify.py
python haar.py
python shape_analysis.py
python optimize.py
python final.py
```

All results populate `step*_output/` folders.

## Dataset Management with DVC

This project uses **DVC (Data Version Control)** to manage the large ~9GB dataset efficiently.

### First-Time Setup (Clone)

When you first clone this repo, the dataset folder is empty. To download the actual data:

```bash
pip install dvc
dvc pull
```

This will fetch the dataset from the remote storage.

### After Cloning

```bash
# Install dependencies
pip install -r requirements.txt  # (if created)
dvc pull  # Download dataset

# Run pipeline
python verify.py
```

### When Updating Dataset

If you modify the dataset:

```bash
# Track changes with DVC
dvc add datasets/

# Commit and push
git add datasets.dvc .gitignore
git commit -m "Update dataset"
git push
dvc push  # Upload to remote storage
```

### DVC Remote Storage

By default, DVC uses local cache. For team collaboration, configure a remote:

```bash
dvc remote add -d myremote <s3://bucket or /path/to/shared/storage>
dvc push  # Upload to remote
```

Learn more: [DVC Documentation](https://dvc.org/doc)

## Output Files

| File | Contents |
|------|----------|
| `step1_output/dataset_summary.json` | Dataset statistics |
| `step2_output/detections.json` | Baseline detections + metrics |
| `step3_output/shape_analysis.png` | Shape distribution visualization |
| `step4_output/results.json` | All 6 variants' FPS + accuracy |
| `step5_output/final.json` | Winning strategies + clean summary |
| `step5_output/dashboard.png` | Final comparison visualization |

## Configuration

Edit these constants in each script as needed:

- **`DATASET_ROOT`**: Path to your dataset (Step 1, 2, 4)
- **`SPLIT`**: `"train"` or `"val"` (default: `"val"`)
- **`IOU_THRESH`**: IoU threshold for accuracy (default: 0.4)
- **`MAX_IMAGES`**: Cap images tested (for quick runs, default: 200)
- **`WARMUP`**: Frames to warm up FPS timer (default: 10–20)

## Use Cases

- **Real-time video**: Use Opt-3 or Opt-4 for frame-skipping/tracking
- **Mobile/Edge**: Use Opt-1 + Opt-2 for maximum speed
- **Balanced**: Use Opt-5 (combined) for best FPS/accuracy tradeoff
- **Analysis**: Review shape_analysis.png to understand detector weaknesses

## Learning Value

This toolkit teaches:

- Wall-clock FPS benchmarking on real hardware
- IoU-based detection accuracy evaluation
- Practical optimization strategies (resolution scaling, parameter tuning, temporal coherence)
- Visualization of model performance across multiple dimensions

## Notes

- Haar Cascade uses fixed 1:1 aspect ratio (cannot detect stretched/tilted faces)
- Optimization impact varies by hardware (CPU, image resolution, scene complexity)
- For best results, tune parameters (`scaleFactor`, `minNeighbors`) on your dataset
- Frame-skipping assumes temporal smoothness in video input

## License

MIT License

## Author

Your Name / Organization

---

For questions or issues, check `step*_output/` folders for detailed results and visualizations.
