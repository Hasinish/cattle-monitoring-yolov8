import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Create an annotated cattle monitoring video from annotated frames.")
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--out-video", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    frame_paths = sorted(args.frames_dir.glob("*.jpg"))

    if not frame_paths:
        raise RuntimeError(f"No .jpg frames found in {args.frames_dir}")

    first = cv2.imread(str(frame_paths[0]))

    if first is None:
        raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")

    height, width = first.shape[:2]

    args.out_video.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out_video), fourcc, args.fps, (width, height))

    for path in frame_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue

        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        writer.write(frame)

    writer.release()

    print(f"Video saved to: {args.out_video}")


if __name__ == "__main__":
    main()
