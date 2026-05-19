import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 detection prediction on cattle frames.")
    parser.add_argument("--weights", required=True, help="Path to trained detection best.pt.")
    parser.add_argument("--source", required=True, help="Image folder or video source.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--device", default="0", help="CUDA device index or cpu.")
    parser.add_argument("--project", default="cattle_project_outputs", help="Output project directory.")
    parser.add_argument("--name", default="det_yolov8s_01_labels", help="Output run name.")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        save_txt=True,
        save_conf=True,
        project=args.project,
        name=args.name,
    )

    print("Detection prediction complete.")
    print(f"Processed {len(results)} images.")


if __name__ == "__main__":
    main()
