# Cattle Monitoring Streamlit App — Patched Version

This patched version fixes the browser video playback problem.

## What Was Fixed

OpenCV often writes `.mp4` videos using the `mp4v` codec. That file may exist and play in VLC, but browser-based players like Streamlit may show a black/unplayable video.

This patched app now:

1. Writes a temporary raw MP4 using OpenCV.
2. Converts it to browser-safe H.264 MP4 using `imageio-ffmpeg`.
3. Uses `st.video(out_video.read_bytes())` for reliable Streamlit playback.

## Features

- Upload cattle video.
- Run YOLOv8 segmentation.
- Generate annotated labelled video.
- Show cattle count per frame.
- Show mask occupancy per frame.
- Detect monitoring events:
  - HIGH_DENSITY
  - HIGH_OCCUPANCY
  - SUDDEN_COUNT_INCREASE
  - SUDDEN_COUNT_DECREASE
  - EMPTY_SCENE
- Download:
  - annotated MP4 video
  - frame results CSV
  - event log CSV

## Install

```powershell
pip install -r requirements.txt
```

## Run

Because Windows may not have `streamlit.exe` on PATH, use:

```powershell
python -m streamlit run app.py
```

## Recommended Settings

For quick testing:

```text
Max frames to process = 100
```

For full video:

```text
Max frames to process = 0
```

## Default Model Path

The sidebar defaults to:

```text
C:\Users\T25301094\Downloads\cattle_runs\seg_yolov8s_640\weights\best.pt
```

Change it if your model is somewhere else.
