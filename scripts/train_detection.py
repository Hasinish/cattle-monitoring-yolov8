import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8s detection model for cattle detection.")
    parser.add_argument("--data", required=True, help="Path to detection dataset YAML file.")
    parser.add_argument("--model", default="yolov8s.pt", help="Base YOLO detection model.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--project", default="cattle_runs", help="Output project directory.")
    parser.add_argument("--name", default="det_yolov8s_640", help="Run name.")
    parser.add_argument("--device", default="0", help="CUDA device index or cpu.")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        seed=args.seed,
        project=args.project,
        name=args.name,
        device=args.device,
    )

    print("Detection training complete.")
    print(results)


if __name__ == "__main__":
    main()
