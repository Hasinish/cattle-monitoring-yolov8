import tempfile
from pathlib import Path
import time
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_DIR / "models" / "segmentation_best.pt"
DEFAULT_OUTPUT_DIR = APP_DIR / "outputs"

DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="Cattle Monitoring System",
    page_icon="🐄",
    layout="wide",
)


def polygon_area(points):
    if len(points) < 3:
        return 0.0

    s = 0.0
    n = len(points)

    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1

    return abs(s) / 2.0


def classify_status(count, occupancy, high_count_threshold, high_occupancy_threshold):
    events = []

    if count == 0:
        events.append("EMPTY_SCENE")

    if count >= high_count_threshold:
        events.append("HIGH_DENSITY")

    if occupancy >= high_occupancy_threshold:
        events.append("HIGH_OCCUPANCY")

    if events:
        return "ALERT", events

    return "NORMAL", []


def draw_panel(frame, frame_idx, count, occupancy, status, events):
    height, width = frame.shape[:2]

    panel_height = 105
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    event_text = ", ".join(events) if events else "None"

    line1 = f"Frame: {frame_idx} | Cattle Count: {count} | Mask Occupancy: {occupancy:.3f}"
    line2 = f"Status: {status} | Events: {event_text}"

    cv2.putText(
        frame,
        line1,
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    status_color = (0, 255, 0) if status == "NORMAL" else (0, 165, 255)

    cv2.putText(
        frame,
        line2,
        (18, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
        cv2.LINE_AA,
    )

    return frame


def convert_to_browser_mp4(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_exe,
        "-y",
        "-i", str(input_path),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError("FFmpeg conversion failed:\n" + result.stderr[-2500:])


def process_video(
    video_path,
    model_path,
    output_dir,
    conf_threshold,
    imgsz,
    high_count_threshold,
    high_occupancy_threshold,
    sudden_change_threshold,
    max_frames,
    device,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 10

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid video dimensions.")

    raw_video_path = output_dir / "cattle_monitoring_segmented_output_raw.mp4"
    browser_video_path = output_dir / "cattle_monitoring_segmented_output.mp4"
    out_csv_path = output_dir / "monitoring_frame_results.csv"
    event_csv_path = output_dir / "monitoring_event_log.csv"

    writer = cv2.VideoWriter(
        str(raw_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create output video writer.")

    rows = []
    event_rows = []

    previous_count = None
    frame_idx = 0

    progress_bar = st.progress(0)
    status_box = st.empty()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_to_process = min(total_frames, max_frames) if max_frames > 0 else total_frames

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if max_frames > 0 and frame_idx >= max_frames:
            break

        results = model.predict(
            source=frame,
            conf=conf_threshold,
            imgsz=imgsz,
            verbose=False,
            device=device,
        )

        result = results[0]

        count = 0
        occupancy = 0.0
        annotated = frame.copy()

        if result.masks is not None and result.masks.xy is not None:
            masks = result.masks.xy
            count = len(masks)

            mask_overlay = annotated.copy()

            for polygon in masks:
                pts = np.array(polygon, dtype=np.int32)

                if len(pts) < 3:
                    continue

                cv2.fillPoly(mask_overlay, [pts], color=(255, 90, 0))
                cv2.polylines(annotated, [pts], isClosed=True, color=(255, 40, 0), thickness=2)

                normalized_points = [(float(x) / width, float(y) / height) for x, y in polygon]
                occupancy += polygon_area(normalized_points)

            annotated = cv2.addWeighted(mask_overlay, 0.35, annotated, 0.65, 0)

        if previous_count is None:
            count_delta = 0
        else:
            count_delta = count - previous_count

        previous_count = count

        status, events = classify_status(
            count,
            occupancy,
            high_count_threshold,
            high_occupancy_threshold,
        )

        if abs(count_delta) >= sudden_change_threshold:
            if count_delta > 0:
                events.append("SUDDEN_COUNT_INCREASE")
            else:
                events.append("SUDDEN_COUNT_DECREASE")
            status = "ALERT"

        annotated = draw_panel(
            annotated,
            frame_idx,
            count,
            occupancy,
            status,
            events,
        )

        writer.write(annotated)

        row = {
            "frame_index": frame_idx,
            "count": count,
            "mask_occupancy": occupancy,
            "count_delta": count_delta,
            "status": status,
            "events": ";".join(events),
        }
        rows.append(row)

        for event in events:
            event_rows.append({
                "frame_index": frame_idx,
                "event_type": event,
                "count": count,
                "mask_occupancy": occupancy,
                "count_delta": count_delta,
            })

        frame_idx += 1

        if total_to_process > 0:
            progress_bar.progress(min(frame_idx / total_to_process, 1.0))

        if frame_idx % 10 == 0:
            status_box.info(f"Processed {frame_idx} frames...")

    cap.release()
    writer.release()

    if frame_idx == 0:
        raise RuntimeError("No frames were processed from the video.")

    status_box.info("Converting video to browser-compatible H.264 MP4...")
    convert_to_browser_mp4(raw_video_path, browser_video_path)

    try:
        raw_video_path.unlink()
    except OSError:
        pass

    df = pd.DataFrame(rows)
    event_df = pd.DataFrame(event_rows)

    df.to_csv(out_csv_path, index=False)
    event_df.to_csv(event_csv_path, index=False)

    progress_bar.progress(1.0)
    status_box.success(f"Done. Processed {frame_idx} frames.")

    return browser_video_path, out_csv_path, event_csv_path, df, event_df


def main():
    st.title("🐄 Cattle Monitoring System")
    st.caption("YOLOv8 Segmentation-Based Video Monitoring Demo")

    st.markdown(
        """
        Upload a cattle video, select your trained YOLOv8 segmentation model, and generate an annotated monitoring video with cattle count, mask occupancy, and event labels.
        """
    )

    with st.sidebar:
        st.header("Settings")

        model_path = st.text_input(
            "Segmentation model path",
            value=str(DEFAULT_MODEL_PATH),
            help="Place your model at streamlit_app/models/segmentation_best.pt or paste another path.",
        )

        device = st.text_input(
            "Device",
            value="0",
            help="Use 0 for GPU. Use cpu if CUDA/GPU is unavailable.",
        )

        conf_threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        imgsz = st.selectbox("Image size", [512, 640, 768], index=1)

        high_count_threshold = st.number_input("High density count threshold", min_value=1, value=9)
        high_occupancy_threshold = st.number_input("High occupancy threshold", min_value=0.01, value=0.18, step=0.01)
        sudden_change_threshold = st.number_input("Sudden count-change threshold", min_value=1, value=4)

        max_frames = st.number_input(
            "Max frames to process (0 = full video)",
            min_value=0,
            value=100,
            help="Use 100 for quick testing. Use 0 for full video.",
        )

    uploaded_video = st.file_uploader(
        "Upload cattle video",
        type=["mp4", "avi", "mov", "mkv"],
    )

    output_dir = st.text_input(
        "Output folder",
        value=str(DEFAULT_OUTPUT_DIR),
        help="Generated videos, CSV files, and event logs will be saved here.",
    )

    if uploaded_video is not None:
        st.subheader("Uploaded Input Video")
        st.video(uploaded_video)

    run_btn = st.button("🚀 Run Segmentation Monitoring", type="primary")

    if run_btn:
        if uploaded_video is None:
            st.error("Upload a video first.")
            return

        if not Path(model_path).exists():
            st.error(
                f"Model file not found:\n\n{model_path}\n\n"
                "Download the trained segmentation best.pt and place it at:\n\n"
                "streamlit_app/models/segmentation_best.pt"
            )
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            temp_video_path = Path(tmp.name)

        try:
            start = time.time()

            out_video, out_csv, event_csv, df, event_df = process_video(
                video_path=temp_video_path,
                model_path=model_path,
                output_dir=output_dir,
                conf_threshold=conf_threshold,
                imgsz=imgsz,
                high_count_threshold=int(high_count_threshold),
                high_occupancy_threshold=float(high_occupancy_threshold),
                sudden_change_threshold=int(sudden_change_threshold),
                max_frames=int(max_frames),
                device=device,
            )

            elapsed = time.time() - start

            st.success(f"Monitoring complete in {elapsed:.1f} seconds.")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Frames processed", len(df))

            with col2:
                st.metric("Average count", f"{df['count'].mean():.2f}" if not df.empty else "0")

            with col3:
                st.metric("Max count", int(df["count"].max()) if not df.empty else 0)

            with col4:
                st.metric("Events", len(event_df))

            st.subheader("Annotated Output Video")
            st.video(out_video.read_bytes())

            st.subheader("Monitoring Timeline")

            if not df.empty:
                chart_df = df[["frame_index", "count", "mask_occupancy"]].set_index("frame_index")
                st.line_chart(chart_df)

            st.subheader("Event Log")
            if event_df.empty:
                st.info("No events detected.")
            else:
                st.dataframe(event_df, use_container_width=True)

            st.subheader("Frame-Level Results")
            st.dataframe(df, use_container_width=True)

            with open(out_video, "rb") as f:
                st.download_button(
                    "Download annotated video",
                    data=f,
                    file_name="cattle_monitoring_segmented_output.mp4",
                    mime="video/mp4",
                )

            with open(out_csv, "rb") as f:
                st.download_button(
                    "Download frame results CSV",
                    data=f,
                    file_name="monitoring_frame_results.csv",
                    mime="text/csv",
                )

            with open(event_csv, "rb") as f:
                st.download_button(
                    "Download event log CSV",
                    data=f,
                    file_name="monitoring_event_log.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.exception(e)

        finally:
            try:
                temp_video_path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()