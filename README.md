# Cattle Monitoring System Using YOLOv8 Detection and Instance Segmentation

This repository contains the complete project code for:

**Cattle Monitoring System Using YOLOv8 Detection and Instance Segmentation**

The project compares YOLOv8s object detection and YOLOv8s instance segmentation for cattle monitoring, then extends the model outputs into an end-to-end monitoring system with event detection, dashboard generation, annotated video output, and a local Streamlit application.

---

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
   - Converts model outputs into practical monitoring signals:
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

9. **Streamlit application**
   - Allows a user to upload a cattle video, run the trained segmentation model, and automatically generate an annotated monitoring video, frame-level CSV, and event log.

---

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

---

## Repository Structure

```text
cattle-monitoring-yolov8/
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── cattle_det.yaml
│   ├── cattle_seg.yaml
│   └── monitoring_config.json
│
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
│   │
│   └── windows/
│       ├── run_training_detection.ps1
│       ├── run_training_segmentation.ps1
│       ├── run_prediction_detection.ps1
│       ├── run_prediction_segmentation.ps1
│       └── run_full_monitoring_pipeline.ps1
│
├── docs/
│   ├── COMMANDS.md
│   └── PROJECT_DESCRIPTION.md
│
├── figures/
│   ├── README.md
│   ├── plot_count_over_time_commonGT_01.png
│   ├── plot_count_abs_error_commonGT_01.png
│   ├── plot_occupancy_abs_error_01.png
│   ├── plot_metric_summary_REPORT_READY.png
│   └── qualitative_detection_vs_segmentation_5ROWS.png
│
├── results/
│   ├── README.md
│   ├── summary_commonGT_01.csv
│   └── frame_level_commonGT_01.csv
│
├── sample_outputs/
│   └── README.md
│
└── streamlit_app/
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── models/
    │   └── segmentation_best.pt      # not included in Git
    └── outputs/                      # generated output, ignored by Git
```

---

## Installation

### Basic CPU Installation

```bash
pip install -r requirements.txt
```

For the Streamlit app:

```bash
pip install -r streamlit_app/requirements.txt
```

### NVIDIA GPU Installation

If you want to use GPU inference, install CUDA-supported PyTorch.

Recommended Python version: **Python 3.12**

```bash
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU support:

```bash
py -3.12 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Expected output should show:

```text
True
NVIDIA GPU name
```

---

## Trained Weights

The trained YOLOv8s weights are stored separately because `.pt` model files are large.

Download trained weights here:

- YOLOv8s Detection model: `<GOOGLE_DRIVE_DETECTION_WEIGHT_LINK>`
- YOLOv8s Segmentation model: `<GOOGLE_DRIVE_SEGMENTATION_WEIGHT_LINK>`

For the Streamlit app, download the segmentation model and place it at:

```text
streamlit_app/models/segmentation_best.pt
```

The app can also use another model path if you paste it into the sidebar.

---

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

---

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

---

## Running the Streamlit App

The Streamlit app allows a user to upload a cattle video and automatically generate:

- annotated segmentation video
- cattle count per frame
- mask occupancy per frame
- monitoring status
- event log
- downloadable CSV files

### Step 1: Place the Model

Download the trained segmentation model and place it here:

```text
streamlit_app/models/segmentation_best.pt
```

### Step 2: Install App Requirements

```bash
pip install -r streamlit_app/requirements.txt
```

For GPU with Python 3.12:

```bash
py -3.12 -m pip install -r streamlit_app/requirements.txt
```

### Step 3: Run the App

CPU or default Python:

```bash
python -m streamlit run streamlit_app/app.py
```

GPU with Python 3.12:

```bash
py -3.12 -m streamlit run streamlit_app/app.py
```

### Step 4: App Settings

In the sidebar:

```text
Device = 0      # for GPU
Device = cpu    # for CPU
```

For low-VRAM GPUs such as GTX 1050 Ti, recommended settings:

```text
Image size = 512
Max frames = 50 or 100 for testing
Max frames = 0 for full video
```

Generated videos, CSV files, and event logs are saved in:

```text
streamlit_app/outputs/
```

---

## Monitoring Events

The system uses simple threshold-based event rules:

| Event | Rule |
|---|---|
| Empty scene | Segmentation count = 0 |
| High density | Segmentation count >= 9 |
| High occupancy | Mask occupancy >= 0.18 |
| Sudden count increase | Frame-to-frame count change >= 4 |
| Sudden count decrease | Frame-to-frame count change <= -4 |

These thresholds can be adjusted depending on camera angle, farm layout, and monitoring requirements.

---

## Important Notes

- Dataset files are not included because they are large.
- Trained weights are not included because `.pt` files are large.
- Generated outputs are ignored by Git.
- The Streamlit app requires the trained segmentation weight file to run inference.
- CPU mode works on most machines but is slower.
- GPU mode requires a CUDA-compatible PyTorch installation.
- The project-specific code is mainly in:
  - `evaluate_common_gt.py`
  - `make_plots.py`
  - `make_qualitative_figure.py`
  - `cattle_monitoring_system.py`
  - `build_monitoring_dashboard.py`
  - `make_monitoring_video.py`
  - `streamlit_app/app.py`

---

## Authors

- Hasin Ishrak — ID: 22201133
- Nusrat Lamia Faruk — ID: `<ADD_ID_HERE>`
- Md. Bashir Al Lazim — ID: `<ADD_ID_HERE>`
