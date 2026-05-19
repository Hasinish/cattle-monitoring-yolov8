import argparse
import base64
import csv
from pathlib import Path
from html import escape


def read_csv_rows(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def img_to_base64(path: Path):
    if not path.exists():
        return ""
    data = path.read_bytes()
    suffix = path.suffix.lower().replace(".", "")
    if suffix == "jpg":
        suffix = "jpeg"
    return f"data:image/{suffix};base64," + base64.b64encode(data).decode("utf-8")


def make_table(rows, max_rows=30):
    if not rows:
        return "<p>No rows available.</p>"

    fieldnames = list(rows[0].keys())

    html = ["<table>", "<thead><tr>"]
    for field in fieldnames:
        html.append(f"<th>{escape(field)}</th>")
    html.append("</tr></thead><tbody>")

    for row in rows[:max_rows]:
        html.append("<tr>")
        for field in fieldnames:
            html.append(f"<td>{escape(str(row[field]))}</td>")
        html.append("</tr>")

    html.append("</tbody></table>")
    return "\n".join(html)


def parse_args():
    parser = argparse.ArgumentParser(description="Build standalone HTML dashboard for cattle monitoring results.")
    parser.add_argument("--monitoring-csv", required=True, type=Path)
    parser.add_argument("--event-log", required=True, type=Path)
    parser.add_argument("--plots-dir", required=True, type=Path)
    parser.add_argument("--out-html", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()

    monitoring_rows = read_csv_rows(args.monitoring_csv)
    event_rows = read_csv_rows(args.event_log)

    total_frames = len(monitoring_rows)
    total_events = len(event_rows)

    seg_counts = [int(float(r["seg_pred_count"])) for r in monitoring_rows]
    seg_occ = [float(r["seg_mask_occupancy"]) for r in monitoring_rows]

    avg_count = sum(seg_counts) / len(seg_counts) if seg_counts else 0
    max_count = max(seg_counts) if seg_counts else 0
    avg_occ = sum(seg_occ) / len(seg_occ) if seg_occ else 0
    max_occ = max(seg_occ) if seg_occ else 0

    plots = [
        ("Monitoring Count Timeline", args.plots_dir / "monitoring_count_timeline.png"),
        ("Monitoring Occupancy Timeline", args.plots_dir / "monitoring_occupancy_timeline.png"),
        ("Count Change Signal", args.plots_dir / "monitoring_count_change_signal.png"),
        ("Density Status Distribution", args.plots_dir / "monitoring_density_distribution.png"),
    ]

    plot_html = []
    for title, path in plots:
        encoded = img_to_base64(path)
        if encoded:
            plot_html.append(f"<section><h2>{escape(title)}</h2><img src='{encoded}' /></section>")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Cattle Monitoring Dashboard</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 32px;
    background: #f5f7fa;
    color: #222;
}}
h1 {{
    margin-bottom: 4px;
}}
.subtitle {{
    color: #555;
    margin-bottom: 24px;
}}
.cards {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}}
.card {{
    background: white;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}
.card .value {{
    font-size: 28px;
    font-weight: bold;
    margin-top: 8px;
}}
section {{
    background: white;
    padding: 18px;
    margin-bottom: 24px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}
img {{
    max-width: 100%;
    border: 1px solid #ddd;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}
th, td {{
    border: 1px solid #ddd;
    padding: 6px 8px;
    text-align: left;
}}
th {{
    background: #eef2f7;
}}
</style>
</head>
<body>

<h1>Cattle Monitoring Dashboard</h1>
<p class="subtitle">YOLOv8s Detection vs YOLOv8s Segmentation Monitoring Outputs</p>

<div class="cards">
    <div class="card"><div>Total Frames</div><div class="value">{total_frames}</div></div>
    <div class="card"><div>Total Events</div><div class="value">{total_events}</div></div>
    <div class="card"><div>Average Count</div><div class="value">{avg_count:.2f}</div></div>
    <div class="card"><div>Max Count</div><div class="value">{max_count}</div></div>
    <div class="card"><div>Avg Occupancy</div><div class="value">{avg_occ:.3f}</div></div>
</div>

{''.join(plot_html)}

<section>
<h2>Recent / First Events</h2>
{make_table(event_rows, max_rows=40)}
</section>

<section>
<h2>Monitoring Summary Sample</h2>
{make_table(monitoring_rows, max_rows=25)}
</section>

</body>
</html>
"""

    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(html, encoding="utf-8")

    print(f"Dashboard saved to: {args.out_html}")


if __name__ == "__main__":
    main()
