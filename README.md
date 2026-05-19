# Cattle Monitoring System Using YOLOv8 Detection and Instance Segmentation

This repository contains the complete project code for:

**Cattle Monitoring System Using YOLOv8 Detection and Instance Segmentation**

The project compares YOLOv8s object detection and YOLOv8s instance segmentation for cattle monitoring, then upgrades the model outputs into an end-to-end monitoring system.

## What This Project Does

This is not only a YOLO training project. The full pipeline includes:

1. **YOLOv8s detection training**
   - Trains a bounding-box detector for cattle.

2. **YOLOv8s segmentation training**
   - Trains an instance segmentation model for cattle masks.

3. **Prediction generation**
   - Runs both trained models on the same validation frame sequence.

4. **Frame-level evaluation**
   - Computes cattle count error.
   - Computes detection box occupancy.
   - Computes segmentation mask occupancy.
   - Compares detection and segmentation using a common ground-truth count.

5. **Monitoring pipeline**
   - Converts model outputs into monitoring signals:
     - cattle count
     - occupancy
     - density status
     - event status

6. **Event detection**
   - Detects:
     - high density
     - high occupancy
     - sudden count increase
     - sudden count decrease
     - empty scene

7. **Dashboard generation**
   - Creates a standalone HTML dashboard with plots, statistics, and event logs.

8. **Annotated video generation**
   - Creates an MP4 demo video with masks, boxes, count, occupancy, and events overlaid.

## Final Experiment Summary

The final comparison used 545 validation frames.

| Metric | YOLOv8s Detection | YOLOv8s Segmentation |
|---|---:|---:|
| Count MAE | 1.881 | 1.145 |
| Count RMSE | 2.371 | 1.563 |
| Occupancy MAE | 0.0936 | 0.0209 |
| Occupancy RMSE | 0.1343 | 0.0307 |
| Total predicted cattle | 2722 | 3287 |
| Common GT cattle count | 3631 | 3631 |

Segmentation performed better for both count estimation and occupancy estimation.

## Repository Structure

```text
cattle_monitoring_yolov8_FULL_UPDATED_CODE_PACKAGE/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── cattle_det.yaml
│   ├── cattle_seg.yaml
│   └── monitoring_config.json
├── scripts/
│   ├── train_detection.py
│   ├── train_segmentation.py
│   ├── predict_detection.py
│   ├── predict_segmentation.py
│   ├── evaluate_common_gt.py
│   ├── make_plots.py
│   ├── make_qualitative_figure.py
│   ├── cattle_monitoring_system.py
│   ├── build_monitoring_dashboard.py
│   ├── make_monitoring_video.py
│   └── windows/
│       ├── run_training_detection.ps1
│       ├── run_training_segmentation.ps1
│       ├── run_prediction_detection.ps1
│       ├── run_prediction_segmentation.ps1
│       └── run_full_monitoring_pipeline.ps1
├── docs/
│   ├── COMMANDS.md
│   └── PROJECT_DESCRIPTION.md
├── results/
└── figures/
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Format

The dataset should follow YOLO format:

```text
dataset/
└── annotation/
    ├── detect/
    │   ├── images/
    │   │   ├── train/
    │   │   └── val/
    │   └── labels/
    │       ├── train/
    │       └── val/
    └── segment/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
```

Both detection and segmentation use a single class:

```text
0: cattle
```

## Quick Usage

### 1. Train Detection

```bash
python scripts/train_detection.py \
  --data configs/cattle_det.yaml \
  --model yolov8s.pt \
  --imgsz 640 \
  --epochs 100 \
  --batch 32 \
  --project cattle_runs \
  --name det_yolov8s_640
```

### 2. Train Segmentation

```bash
python scripts/train_segmentation.py \
  --data configs/cattle_seg.yaml \
  --model yolov8s-seg.pt \
  --imgsz 640 \
  --epochs 100 \
  --batch 16 \
  --project cattle_runs \
  --name seg_yolov8s_640
```

### 3. Predict Detection

```bash
python scripts/predict_detection.py \
  --weights cattle_runs/det_yolov8s_640/weights/best.pt \
  --source dataset/annotation/detect/images/val/01.mp4 \
  --project cattle_project_outputs \
  --name det_yolov8s_01_labels
```

### 4. Predict Segmentation

```bash
python scripts/predict_segmentation.py \
  --weights cattle_runs/seg_yolov8s_640/weights/best.pt \
  --source dataset/annotation/segment/images/val/01.mp4 \
  --project cattle_project_outputs \
  --name seg_yolov8s_01_labels
```

### 5. Evaluate Detection vs Segmentation

```bash
python scripts/evaluate_common_gt.py \
  --det-img-dir dataset/annotation/detect/images/val/01.mp4 \
  --seg-img-dir dataset/annotation/segment/images/val/01.mp4 \
  --det-gt-dir dataset/annotation/detect/labels/val/01.mp4 \
  --seg-gt-dir dataset/annotation/segment/labels/val/01.mp4 \
  --det-pred-dir cattle_project_outputs/det_yolov8s_01_labels/labels \
  --seg-pred-dir cattle_project_outputs/seg_yolov8s_01_labels/labels \
  --out-dir cattle_project_outputs/analysis_01
```

### 6. Generate Plots

```bash
python scripts/make_plots.py \
  --frame-csv cattle_project_outputs/analysis_01/frame_level_commonGT_01.csv \
  --summary-csv cattle_project_outputs/analysis_01/summary_commonGT_01.csv \
  --out-dir cattle_project_outputs/analysis_01
```

### 7. Generate Qualitative Comparison Figure

```bash
python scripts/make_qualitative_figure.py \
  --frame-csv cattle_project_outputs/analysis_01/frame_level_commonGT_01.csv \
  --raw-img-dir dataset/annotation/detect/images/val/01.mp4 \
  --det-pred-dir cattle_project_outputs/det_yolov8s_01_labels/labels \
  --seg-pred-dir cattle_project_outputs/seg_yolov8s_01_labels/labels \
  --out cattle_project_outputs/analysis_01/qualitative_detection_vs_segmentation_5ROWS.png
```

### 8. Run Monitoring System

```bash
python scripts/cattle_monitoring_system.py \
  --frame-csv cattle_project_outputs/analysis_01/frame_level_commonGT_01.csv \
  --raw-img-dir dataset/annotation/detect/images/val/01.mp4 \
  --det-pred-dir cattle_project_outputs/det_yolov8s_01_labels/labels \
  --seg-pred-dir cattle_project_outputs/seg_yolov8s_01_labels/labels \
  --out-dir cattle_project_outputs/monitoring_system_01 \
  --save-annotated-frames
```

### 9. Build Dashboard

```bash
python scripts/build_monitoring_dashboard.py \
  --monitoring-csv cattle_project_outputs/monitoring_system_01/monitoring_summary.csv \
  --event-log cattle_project_outputs/monitoring_system_01/event_log.csv \
  --plots-dir cattle_project_outputs/monitoring_system_01/plots \
  --out-html cattle_project_outputs/monitoring_system_01/dashboard.html
```

### 10. Make Monitoring Video

```bash
python scripts/make_monitoring_video.py \
  --frames-dir cattle_project_outputs/monitoring_system_01/annotated_frames \
  --out-video cattle_project_outputs/monitoring_system_01/cattle_monitoring_demo.mp4 \
  --fps 10
```

## Important Notes

- Dataset files are not included because they are large.
- Trained weights are not included because `.pt` files are large.
- Output folders are ignored by Git through `.gitignore`.
- The project-specific code is mainly in:
  - `evaluate_common_gt.py`
  - `make_plots.py`
  - `make_qualitative_figure.py`
  - `cattle_monitoring_system.py`
  - `build_monitoring_dashboard.py`
  - `make_monitoring_video.py`

## Authors

- Hasin Ishrak — ID: 22201133
- Nusrat Lamia Faruk — ID:
- Md. Bashir Al Lazim — ID:
