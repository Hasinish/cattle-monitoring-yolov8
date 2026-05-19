import argparse
import csv
from pathlib import Path
from collections import Counter

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


def read_csv_rows(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_lines(path: Path):
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def get_det_boxes(label_path: Path, width: int, height: int):
    boxes = []

    for line in read_lines(label_path):
        parts = line.split()
        if len(parts) < 5:
            continue

        try:
            x_center, y_center, box_width, box_height = map(float, parts[1:5])
        except ValueError:
            continue

        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)
        boxes.append((x1, y1, x2, y2))

    return boxes


def get_seg_polygons(label_path: Path, width: int, height: int):
    polygons = []

    for line in read_lines(label_path):
        parts = line.split()
        if len(parts) < 7:
            continue

        try:
            values = list(map(float, parts[1:]))
        except ValueError:
            continue

        if len(values) % 2 == 1:
            values = values[:-1]

        if len(values) < 6:
            continue

        points = []
        for i in range(0, len(values), 2):
            points.append((int(values[i] * width), int(values[i + 1] * height)))

        if len(points) >= 3:
            polygons.append(points)

    return polygons


def classify_density(count: int, high_count_threshold: int):
    if count == 0:
        return "EMPTY"
    if count >= high_count_threshold:
        return "HIGH_DENSITY"
    if count >= max(1, high_count_threshold // 2):
        return "MEDIUM_DENSITY"
    return "LOW_DENSITY"


def classify_occupancy(occupancy: float, high_occupancy_threshold: float):
    if occupancy == 0:
        return "NO_OCCUPANCY"
    if occupancy >= high_occupancy_threshold:
        return "HIGH_OCCUPANCY"
    if occupancy >= high_occupancy_threshold / 2:
        return "MEDIUM_OCCUPANCY"
    return "LOW_OCCUPANCY"


def build_events(seg_count, seg_occupancy, count_delta, high_count_threshold, high_occupancy_threshold, sudden_change_threshold):
    events = []

    if seg_count == 0:
        events.append("EMPTY_SCENE")

    if seg_count >= high_count_threshold:
        events.append("HIGH_DENSITY")

    if seg_occupancy >= high_occupancy_threshold:
        events.append("HIGH_OCCUPANCY")

    if abs(count_delta) >= sudden_change_threshold:
        if count_delta > 0:
            events.append("SUDDEN_COUNT_INCREASE")
        else:
            events.append("SUDDEN_COUNT_DECREASE")

    return events


def draw_monitoring_frame(raw_image, det_boxes, seg_polygons, row):
    image = raw_image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    for polygon in seg_polygons:
        draw_overlay.polygon(polygon, fill=(0, 90, 255, 95), outline=(0, 35, 220, 255))

    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)

    for box in det_boxes:
        draw.rectangle(box, outline=(0, 70, 255, 255), width=4)

    panel_height = 110
    draw.rectangle([0, 0, image.width, panel_height], fill=(0, 0, 0, 180))

    try:
        font_big = ImageFont.truetype("arial.ttf", 28)
        font_small = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    line1 = (
        f"Frame: {row['frame']} | "
        f"Seg Count: {row['seg_pred_count']} | "
        f"Det Count: {row['det_pred_count']} | "
        f"Seg Occupancy: {float(row['seg_mask_occupancy']):.3f}"
    )
    line2 = f"Status: {row['monitoring_status']} | Events: {row['events'] if row['events'] else 'None'}"

    draw.text((18, 16), line1, fill=(255, 255, 255, 255), font=font_big)
    draw.text((18, 62), line2, fill=(255, 255, 255, 255), font=font_small)

    return image.convert("RGB")


def create_plots(rows, out_dir: Path):
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    x = list(range(len(rows)))
    seg_counts = [int(r["seg_pred_count"]) for r in rows]
    det_counts = [int(r["det_pred_count"]) for r in rows]
    gt_counts = [int(r["common_gt_count"]) for r in rows]
    seg_occ = [float(r["seg_mask_occupancy"]) for r in rows]
    det_occ = [float(r["det_box_occupancy"]) for r in rows]
    count_delta = [int(r["count_delta"]) for r in rows]

    plt.figure(figsize=(14, 5))
    plt.plot(x, gt_counts, label="Common GT count")
    plt.plot(x, det_counts, label="Detection count")
    plt.plot(x, seg_counts, label="Segmentation count")
    plt.xlabel("Frame index")
    plt.ylabel("Cattle count")
    plt.title("Monitoring Count Timeline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "monitoring_count_timeline.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(x, det_occ, label="Detection box occupancy")
    plt.plot(x, seg_occ, label="Segmentation mask occupancy")
    plt.xlabel("Frame index")
    plt.ylabel("Normalized occupancy")
    plt.title("Monitoring Occupancy Timeline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "monitoring_occupancy_timeline.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(x, count_delta, label="Frame-to-frame segmentation count change")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Frame index")
    plt.ylabel("Count change")
    plt.title("Entry/Exit Change Signal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "monitoring_count_change_signal.png", dpi=200)
    plt.close()

    status_counts = Counter(r["density_status"] for r in rows)
    plt.figure(figsize=(8, 5))
    plt.bar(list(status_counts.keys()), list(status_counts.values()))
    plt.xlabel("Density status")
    plt.ylabel("Frame count")
    plt.title("Density Status Distribution")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(plots_dir / "monitoring_density_distribution.png", dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end cattle monitoring system using YOLOv8 outputs.")
    parser.add_argument("--frame-csv", required=True, type=Path)
    parser.add_argument("--raw-img-dir", required=True, type=Path)
    parser.add_argument("--det-pred-dir", required=True, type=Path)
    parser.add_argument("--seg-pred-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--high-count-threshold", type=int, default=9)
    parser.add_argument("--high-occupancy-threshold", type=float, default=0.18)
    parser.add_argument("--sudden-change-threshold", type=int, default=4)
    parser.add_argument("--save-annotated-frames", action="store_true")
    parser.add_argument("--max-annotated-frames", type=int, default=0, help="0 means save all frames.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    input_rows = read_csv_rows(args.frame_csv)

    monitoring_rows = []
    event_rows = []
    previous_seg_count = None

    for idx, row in enumerate(input_rows):
        frame = row["frame"]

        common_gt = int(float(row["common_gt_count"]))
        det_count = int(float(row["det_pred_count"]))
        seg_count = int(float(row["seg_pred_count"]))

        det_occupancy = float(row["det_pred_box_occupancy"])
        seg_occupancy = float(row["seg_pred_mask_occupancy"])

        count_delta = 0 if previous_seg_count is None else seg_count - previous_seg_count
        previous_seg_count = seg_count

        density_status = classify_density(seg_count, args.high_count_threshold)
        occupancy_status = classify_occupancy(seg_occupancy, args.high_occupancy_threshold)

        events = build_events(
            seg_count,
            seg_occupancy,
            count_delta,
            args.high_count_threshold,
            args.high_occupancy_threshold,
            args.sudden_change_threshold,
        )

        monitoring_status = "ALERT" if events else "NORMAL"

        monitoring_row = {
            "frame_index": idx,
            "frame": frame,
            "common_gt_count": common_gt,
            "det_pred_count": det_count,
            "seg_pred_count": seg_count,
            "det_box_occupancy": det_occupancy,
            "seg_mask_occupancy": seg_occupancy,
            "count_delta": count_delta,
            "density_status": density_status,
            "occupancy_status": occupancy_status,
            "monitoring_status": monitoring_status,
            "events": ";".join(events),
        }

        monitoring_rows.append(monitoring_row)

        for event in events:
            event_rows.append({
                "frame_index": idx,
                "frame": frame,
                "event_type": event,
                "seg_pred_count": seg_count,
                "seg_mask_occupancy": seg_occupancy,
                "count_delta": count_delta,
            })

    write_csv(args.out_dir / "monitoring_summary.csv", monitoring_rows, list(monitoring_rows[0].keys()))
    write_csv(
        args.out_dir / "event_log.csv",
        event_rows,
        ["frame_index", "frame", "event_type", "seg_pred_count", "seg_mask_occupancy", "count_delta"],
    )

    create_plots(monitoring_rows, args.out_dir)

    avg_seg_count = sum(int(r["seg_pred_count"]) for r in monitoring_rows) / len(monitoring_rows)
    max_seg_count = max(int(r["seg_pred_count"]) for r in monitoring_rows)
    avg_seg_occ = sum(float(r["seg_mask_occupancy"]) for r in monitoring_rows) / len(monitoring_rows)
    max_seg_occ = max(float(r["seg_mask_occupancy"]) for r in monitoring_rows)

    stats_text = f"""Cattle Monitoring System Summary
================================

Frames analyzed: {len(monitoring_rows)}
Average segmentation count: {avg_seg_count:.3f}
Maximum segmentation count: {max_seg_count}
Average segmentation occupancy: {avg_seg_occ:.4f}
Maximum segmentation occupancy: {max_seg_occ:.4f}
Total events logged: {len(event_rows)}

Thresholds
----------
High density threshold: {args.high_count_threshold}
High occupancy threshold: {args.high_occupancy_threshold}
Sudden count change threshold: {args.sudden_change_threshold}

Outputs
-------
monitoring_summary.csv
event_log.csv
plots/
annotated_frames/ optional
"""

    (args.out_dir / "monitoring_stats.txt").write_text(stats_text, encoding="utf-8")

    if args.save_annotated_frames:
        annotated_dir = args.out_dir / "annotated_frames"
        annotated_dir.mkdir(parents=True, exist_ok=True)

        limit = args.max_annotated_frames if args.max_annotated_frames > 0 else len(monitoring_rows)

        for row in monitoring_rows[:limit]:
            frame = row["frame"]
            raw_path = args.raw_img_dir / f"{frame}.jpg"

            if not raw_path.exists():
                continue

            raw = Image.open(raw_path).convert("RGB")
            width, height = raw.size

            det_boxes = get_det_boxes(args.det_pred_dir / f"{frame}.txt", width, height)
            seg_polygons = get_seg_polygons(args.seg_pred_dir / f"{frame}.txt", width, height)

            annotated = draw_monitoring_frame(raw, det_boxes, seg_polygons, row)
            annotated.save(annotated_dir / f"{frame}.jpg", quality=95)

    print("Monitoring system complete.")
    print(f"Output folder: {args.out_dir}")
    print(f"Monitoring CSV: {args.out_dir / 'monitoring_summary.csv'}")
    print(f"Event log: {args.out_dir / 'event_log.csv'}")
    print(f"Stats: {args.out_dir / 'monitoring_stats.txt'}")
    print(f"Plots: {args.out_dir / 'plots'}")


if __name__ == "__main__":
    main()
