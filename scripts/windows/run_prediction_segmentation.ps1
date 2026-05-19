python scripts/predict_segmentation.py `
  --weights cattle_runs/seg_yolov8s_640/weights/best.pt `
  --source dataset/annotation/segment/images/val/01.mp4 `
  --imgsz 640 `
  --conf 0.25 `
  --device 0 `
  --project cattle_project_outputs `
  --name seg_yolov8s_01_labels
