import argparse
import csv
import math
from pathlib import Path


def read_lines(path: Path):
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def parse_detection_label(path: Path):
    """
    YOLO detection format:
    class x_center y_center width height [confidence]
    Coordinates are normalized.
    """
    count = 0
    total_area = 0.0

    for line in read_lines(path):
        parts = line.split()
        if len(parts) < 5:
            continue

        try:
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            continue

        count += 1
        total_area += max(0.0, width) * max(0.0, height)

    return count, total_area


def polygon_area(points):
    """Compute normalized polygon area using the shoelace formula."""
    if len(points) < 3:
        return 0.0

    area_sum = 0.0
    n = len(points)

    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area_sum += x1 * y2 - x2 * y1

    return abs(area_sum) / 2.0


def parse_segmentation_label(path: Path):
    """
    YOLO segmentation format:
    class x1 y1 x2 y2 x3 y3 ... [confidence]
    Coordinates are normalized.
    """
    count = 0
    total_area = 0.0

    for line in read_lines(path):
        parts = line.split()
        if len(parts) < 7:
            continue

        try:
            values = list(map(float, parts[1:]))
        except ValueError:
            continue

        # save_conf=True appends confidence as the last value.
        if len(values) % 2 == 1:
            values = values[:-1]

        if len(values) < 6:
            continue

        points = [(values[i], values[i + 1]) for i in range(0, len(values), 2)]

        count += 1
        total_area += polygon_area(points)

    return count, total_area


def mae(errors):
    return sum(abs(e) for e in errors) / len(errors) if errors else 0.0


def rmse(errors):
    return math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 detection vs segmentation for cattle count and occupancy.")
    parser.add_argument("--det-img-dir", required=True, type=Path)
    parser.add_argument("--seg-img-dir", required=True, type=Path)
    parser.add_argument("--det-gt-dir", required=True, type=Path)
    parser.add_argument("--seg-gt-dir", required=True, type=Path)
    parser.add_argument("--det-pred-dir", required=True, type=Path)
    parser.add_argument("--seg-pred-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    det_frames = sorted(p.stem for p in args.det_img_dir.glob("*.jpg"))
    seg_frames = sorted(p.stem for p in args.seg_img_dir.glob("*.jpg"))

    if det_frames != seg_frames:
        raise RuntimeError("Detection and segmentation frame names do not match.")

    rows = []

    for frame in det_frames:
        det_gt_count, det_gt_box_occ = parse_detection_label(args.det_gt_dir / f"{frame}.txt")
        seg_gt_count, seg_gt_mask_occ = parse_segmentation_label(args.seg_gt_dir / f"{frame}.txt")

        det_pred_count, det_pred_box_occ = parse_detection_label(args.det_pred_dir / f"{frame}.txt")
        seg_pred_count, seg_pred_mask_occ = parse_segmentation_label(args.seg_pred_dir / f"{frame}.txt")

        common_gt_count = seg_gt_count

        rows.append({
            "frame": frame,
            "common_gt_count": common_gt_count,
            "det_gt_count": det_gt_count,
            "seg_gt_count": seg_gt_count,
            "det_pred_count": det_pred_count,
            "seg_pred_count": seg_pred_count,
            "det_count_error_commonGT": det_pred_count - common_gt_count,
            "seg_count_error_commonGT": seg_pred_count - common_gt_count,
            "det_abs_count_error_commonGT": abs(det_pred_count - common_gt_count),
            "seg_abs_count_error_commonGT": abs(seg_pred_count - common_gt_count),
            "det_gt_box_occupancy": det_gt_box_occ,
            "det_pred_box_occupancy": det_pred_box_occ,
            "det_abs_box_occupancy_error": abs(det_pred_box_occ - det_gt_box_occ),
            "seg_gt_mask_occupancy": seg_gt_mask_occ,
            "seg_pred_mask_occupancy": seg_pred_mask_occ,
            "seg_abs_mask_occupancy_error": abs(seg_pred_mask_occ - seg_gt_mask_occ),
        })

    frame_csv = args.out_dir / "frame_level_commonGT_01.csv"
    summary_csv = args.out_dir / "summary_commonGT_01.csv"

    with frame_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    det_count_errors = [r["det_count_error_commonGT"] for r in rows]
    seg_count_errors = [r["seg_count_error_commonGT"] for r in rows]
    det_occ_errors = [r["det_abs_box_occupancy_error"] for r in rows]
    seg_occ_errors = [r["seg_abs_mask_occupancy_error"] for r in rows]

    summary = {
        "frames_analyzed": len(rows),
        "common_ground_truth": "segmentation_GT_count",
        "total_common_gt_count": sum(r["common_gt_count"] for r in rows),
        "total_det_pred_count": sum(r["det_pred_count"] for r in rows),
        "total_seg_pred_count": sum(r["seg_pred_count"] for r in rows),
        "det_count_MAE_commonGT": mae(det_count_errors),
        "det_count_RMSE_commonGT": rmse(det_count_errors),
        "seg_count_MAE_commonGT": mae(seg_count_errors),
        "seg_count_RMSE_commonGT": rmse(seg_count_errors),
        "det_box_occupancy_MAE": mae(det_occ_errors),
        "det_box_occupancy_RMSE": rmse(det_occ_errors),
        "seg_mask_occupancy_MAE": mae(seg_occ_errors),
        "seg_mask_occupancy_RMSE": rmse(seg_occ_errors),
    }

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("Evaluation complete.")
    print(f"Frame-level CSV: {frame_csv}")
    print(f"Summary CSV: {summary_csv}")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
