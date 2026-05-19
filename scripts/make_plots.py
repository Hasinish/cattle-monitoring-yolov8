import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(frame_csv):
    with frame_csv.open("r", newline="") as f:
        return list(csv.DictReader(f))


def read_summary(summary_csv):
    with summary_csv.open("r", newline="") as f:
        return next(csv.DictReader(f))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots for cattle detection vs segmentation evaluation.")
    parser.add_argument("--frame-csv", required=True, type=Path)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.frame_csv)
    summary = read_summary(args.summary_csv)

    x = list(range(len(rows)))

    gt = [int(float(r["common_gt_count"])) for r in rows]
    det_pred = [int(float(r["det_pred_count"])) for r in rows]
    seg_pred = [int(float(r["seg_pred_count"])) for r in rows]

    det_abs_count_error = [float(r["det_abs_count_error_commonGT"]) for r in rows]
    seg_abs_count_error = [float(r["seg_abs_count_error_commonGT"]) for r in rows]

    det_occ_error = [float(r["det_abs_box_occupancy_error"]) for r in rows]
    seg_occ_error = [float(r["seg_abs_mask_occupancy_error"]) for r in rows]

    plt.figure(figsize=(14, 5))
    plt.plot(x, gt, label="Common GT count")
    plt.plot(x, det_pred, label="Detection predicted count")
    plt.plot(x, seg_pred, label="Segmentation predicted count")
    plt.xlabel("Frame index")
    plt.ylabel("Cattle count")
    plt.title("Cattle Count Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "plot_count_over_time_commonGT_01.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(x, det_abs_count_error, label="Detection absolute count error")
    plt.plot(x, seg_abs_count_error, label="Segmentation absolute count error")
    plt.xlabel("Frame index")
    plt.ylabel("Absolute count error")
    plt.title("Count Error Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "plot_count_abs_error_commonGT_01.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(x, det_occ_error, label="Detection box occupancy absolute error")
    plt.plot(x, seg_occ_error, label="Segmentation mask occupancy absolute error")
    plt.xlabel("Frame index")
    plt.ylabel("Absolute occupancy error")
    plt.title("Occupancy Error Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "plot_occupancy_abs_error_01.png", dpi=200)
    plt.close()

    det_count_mae = float(summary["det_count_MAE_commonGT"])
    seg_count_mae = float(summary["seg_count_MAE_commonGT"])
    det_count_rmse = float(summary["det_count_RMSE_commonGT"])
    seg_count_rmse = float(summary["seg_count_RMSE_commonGT"])

    det_occ_mae = float(summary["det_box_occupancy_MAE"])
    seg_occ_mae = float(summary["seg_mask_occupancy_MAE"])
    det_occ_rmse = float(summary["det_box_occupancy_RMSE"])
    seg_occ_rmse = float(summary["seg_mask_occupancy_RMSE"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(
        ["Det MAE", "Seg MAE", "Det RMSE", "Seg RMSE"],
        [det_count_mae, seg_count_mae, det_count_rmse, seg_count_rmse],
    )
    axes[0].set_title("Count Error Comparison")
    axes[0].set_ylabel("Cattle count error")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(
        ["Det MAE", "Seg MAE", "Det RMSE", "Seg RMSE"],
        [det_occ_mae, seg_occ_mae, det_occ_rmse, seg_occ_rmse],
    )
    axes[1].set_title("Occupancy Error Comparison")
    axes[1].set_ylabel("Normalized occupancy error")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("YOLOv8s Detection vs YOLOv8s Segmentation")
    plt.tight_layout()
    plt.savefig(args.out_dir / "plot_metric_summary_REPORT_READY.png", dpi=200)
    plt.close()

    print(f"Plots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
