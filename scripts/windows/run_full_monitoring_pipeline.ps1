$OUT="cattle_project_outputs\monitoring_system_01"

python scripts/cattle_monitoring_system.py `
  --frame-csv "cattle_project_outputs\analysis_01\frame_level_commonGT_01.csv" `
  --raw-img-dir "dataset\annotation\detect\images\val\01.mp4" `
  --det-pred-dir "cattle_project_outputs\det_yolov8s_01_labels\labels" `
  --seg-pred-dir "cattle_project_outputs\seg_yolov8s_01_labels\labels" `
  --out-dir $OUT `
  --save-annotated-frames

python scripts/build_monitoring_dashboard.py `
  --monitoring-csv "$OUT\monitoring_summary.csv" `
  --event-log "$OUT\event_log.csv" `
  --plots-dir "$OUT\plots" `
  --out-html "$OUT\dashboard.html"

python scripts/make_monitoring_video.py `
  --frames-dir "$OUT\annotated_frames" `
  --out-video "$OUT\cattle_monitoring_demo.mp4" `
  --fps 10

explorer $OUT
