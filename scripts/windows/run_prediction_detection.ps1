python scripts/predict_detection.py `
  --weights cattle_runs/det_yolov8s_640/weights/best.pt `
  --source dataset/annotation/detect/images/val/01.mp4 `
  --imgsz 640 `
  --conf 0.25 `
  --device 0 `
  --project cattle_project_outputs `
  --name det_yolov8s_01_labels
