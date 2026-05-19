# Commands

This file lists the main commands used in the project.

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Train Detection

```bash
python scripts/train_detection.py --data configs/cattle_det.yaml --model yolov8s.pt --imgsz 640 --epochs 100 --patience 20 --batch 32 --seed 42 --project cattle_runs --name det_yolov8s_640 --device 0
```

## Train Segmentation

```bash
python scripts/train_segmentation.py --data configs/cattle_seg.yaml --model yolov8s-seg.pt --imgsz 640 --epochs 100 --patience 20 --batch 16 --seed 42 --project cattle_runs --name seg_yolov8s_640 --device 0
```

## Predict Detection

```bash
python scripts/predict_detection.py --weights cattle_runs/det_yolov8s_640/weights/best.pt --source dataset/annotation/detect/images/val/01.mp4 --imgsz 640 --conf 0.25 --device 0 --project cattle_project_outputs --name det_yolov8s_01_labels
```

## Predict Segmentation

```bash
python scripts/predict_segmentation.py --weights cattle_runs/seg_yolov8s_640/weights/best.pt --source dataset/annotation/segment/images/val/01.mp4 --imgsz 640 --conf 0.25 --device 0 --project cattle_project_outputs --name seg_yolov8s_01_labels
```

## Evaluate

```bash
python scripts/evaluate_common_gt.py --det-img-dir dataset/annotation/detect/images/val/01.mp4 --seg-img-dir dataset/annotation/segment/images/val/01.mp4 --det-gt-dir dataset/annotation/detect/labels/val/01.mp4 --seg-gt-dir dataset/annotation/segment/labels/val/01.mp4 --det-pred-dir cattle_project_outputs/det_yolov8s_01_labels/labels --seg-pred-dir cattle_project_outputs/seg_yolov8s_01_labels/labels --out-dir cattle_project_outputs/analysis_01
```

## Plot

```bash
python scripts/make_plots.py --frame-csv cattle_project_outputs/analysis_01/frame_level_commonGT_01.csv --summary-csv cattle_project_outputs/analysis_01/summary_commonGT_01.csv --out-dir cattle_project_outputs/analysis_01
```

## Monitoring System

```bash
python scripts/cattle_monitoring_system.py --frame-csv cattle_project_outputs/analysis_01/frame_level_commonGT_01.csv --raw-img-dir dataset/annotation/detect/images/val/01.mp4 --det-pred-dir cattle_project_outputs/det_yolov8s_01_labels/labels --seg-pred-dir cattle_project_outputs/seg_yolov8s_01_labels/labels --out-dir cattle_project_outputs/monitoring_system_01 --save-annotated-frames
```

## Dashboard

```bash
python scripts/build_monitoring_dashboard.py --monitoring-csv cattle_project_outputs/monitoring_system_01/monitoring_summary.csv --event-log cattle_project_outputs/monitoring_system_01/event_log.csv --plots-dir cattle_project_outputs/monitoring_system_01/plots --out-html cattle_project_outputs/monitoring_system_01/dashboard.html
```

## Video

```bash
python scripts/make_monitoring_video.py --frames-dir cattle_project_outputs/monitoring_system_01/annotated_frames --out-video cattle_project_outputs/monitoring_system_01/cattle_monitoring_demo.mp4 --fps 10
```
