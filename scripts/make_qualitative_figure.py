import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_FRAMES = ["00938", "01008", "00985", "00943", "01005"]

NOTES = {
    "00938": "Segmentation exactly matches GT count",
    "01008": "Segmentation much closer than detection",
    "00985": "Crowded scene where both models struggle",
    "00943": "Segmentation much closer than detection",
    "01005": "Segmentation much closer than detection",
}


def read_lines(path):
    if not path.exists():
        return []
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def get_det_boxes(label_path, width, height):
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


def get_seg_polygons(label_path, width, height):
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


def crop_from_labels(raw, boxes, polygons):
    width, height = raw.size
    y_values = []

    for x1, y1, x2, y2 in boxes:
        y_values.extend([y1, y2])

    for polygon in polygons:
        for _, y in polygon:
            y_values.append(y)

    if not y_values:
        return (0, int(height * 0.35), width, int(height * 0.75))

    y_min = max(0, min(y_values) - 80)
    y_max = min(height, max(y_values) + 80)

    return (0, y_min, width, y_max)


def draw_detection(raw, boxes):
    image = raw.convert("RGBA")
    draw = ImageDraw.Draw(image)

    for box in boxes:
        draw.rectangle(box, outline=(0, 70, 255, 255), width=7)

    return image.convert("RGB")


def draw_segmentation(raw, polygons):
    base = raw.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for points in polygons:
        draw.polygon(points, fill=(0, 90, 255, 120), outline=(0, 30, 220, 255))

    return Image.alpha_composite(base, overlay).convert("RGB")


def load_frame_data(frame_csv):
    with frame_csv.open("r", newline="") as f:
        return {row["frame"]: row for row in csv.DictReader(f)}


def parse_args():
    parser = argparse.ArgumentParser(description="Create qualitative detection-vs-segmentation comparison figure.")
    parser.add_argument("--frame-csv", required=True, type=Path)
    parser.add_argument("--raw-img-dir", required=True, type=Path)
    parser.add_argument("--det-pred-dir", required=True, type=Path)
    parser.add_argument("--seg-pred-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frames", nargs="*", default=DEFAULT_FRAMES)
    return parser.parse_args()


def main():
    args = parse_args()
    frame_data = load_frame_data(args.frame_csv)

    image_width, image_height = 760, 320
    header_height = 56
    subheader_height = 34
    row_height = header_height + subheader_height + image_height
    gap = 14

    canvas_width = image_width * 2
    canvas_height = row_height * len(args.frames) + gap * (len(args.frames) - 1)

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font_header = ImageFont.truetype("arial.ttf", 24)
        font_sub = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font_header = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    for index, frame in enumerate(args.frames):
        y0 = index * (row_height + gap)
        row = frame_data[frame]

        gt = int(float(row["common_gt_count"]))
        detection_count = int(float(row["det_pred_count"]))
        segmentation_count = int(float(row["seg_pred_count"]))

        raw = Image.open(args.raw_img_dir / f"{frame}.jpg").convert("RGB")
        width, height = raw.size

        boxes = get_det_boxes(args.det_pred_dir / f"{frame}.txt", width, height)
        polygons = get_seg_polygons(args.seg_pred_dir / f"{frame}.txt", width, height)

        crop_box = crop_from_labels(raw, boxes, polygons)

        det_image = draw_detection(raw.copy(), boxes).crop(crop_box).resize((image_width, image_height))
        seg_image = draw_segmentation(raw.copy(), polygons).crop(crop_box).resize((image_width, image_height))

        draw.rectangle([0, y0, canvas_width, y0 + header_height], fill=(242, 242, 242))

        note = NOTES.get(frame, "Qualitative comparison")
        title = f"Frame {frame} | GT={gt} | Detection={detection_count} | Segmentation={segmentation_count} | {note}"
        draw.text((10, y0 + 14), title, fill="black", font=font_header)

        y_sub = y0 + header_height
        draw.rectangle([0, y_sub, canvas_width, y_sub + subheader_height], fill=(250, 250, 250))
        draw.text((10, y_sub + 7), "YOLOv8s Detection: bounding boxes", fill="black", font=font_sub)
        draw.text((image_width + 10, y_sub + 7), "YOLOv8s Segmentation: instance masks", fill="black", font=font_sub)

        y_img = y0 + header_height + subheader_height
        canvas.paste(det_image, (0, y_img))
        canvas.paste(seg_image, (image_width, y_img))

        draw.line([(image_width, y_sub), (image_width, y0 + row_height)], fill=(0, 0, 0), width=2)
        draw.rectangle([0, y0, canvas_width - 1, y0 + row_height - 1], outline=(0, 0, 0), width=2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"Qualitative figure saved to: {args.out}")


if __name__ == "__main__":
    main()
