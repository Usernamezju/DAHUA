"""Render an RTMPose COCO-17 feature into a browser-safe skeleton video."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

from dahua_cup.pipeline.video_encoding import browser_video_args


COCO17_EDGES = ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7),
                (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
                (11, 13), (13, 15), (12, 14), (14, 16))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--bitrate", default="2M")
    parser.add_argument("--fps", type=float, default=10.0)
    return parser


def _load(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as value:
        schema = str(np.asarray(value["schema_version"]).item())
        keypoint = np.asarray(value["keypoint"], dtype=np.float32)
        score = np.asarray(value["keypoint_score"], dtype=np.float32) if "keypoint_score" in value else None
    if schema != "rtmpose_coco17_2d.v1" or keypoint.ndim != 4 or keypoint.shape[2:] != (17, 2):
        raise ValueError(f"expected rtmpose_coco17_2d.v1 [M,T,17,2], got {schema} {keypoint.shape}")
    return keypoint, score


def render(args: argparse.Namespace) -> None:
    keypoint, score = _load(Path(args.feature))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.mp4")
    width, height = 960, 540
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("unable to open temporary pose-video writer")
    try:
        for frame_index in range(keypoint.shape[1]):
            canvas = np.full((height, width, 3), 250, dtype=np.uint8)
            for person_index in range(keypoint.shape[0]):
                joints = keypoint[person_index, frame_index]
                confidence = score[person_index, frame_index] if score is not None else np.ones(17, dtype=np.float32)
                points = [(int(np.clip(x, 0, 1) * (width - 1)), int(np.clip(y, 0, 1) * (height - 1))) for x, y in joints]
                color = ((255, 96, 32), (32, 128, 255))[person_index % 2]
                for start, end in COCO17_EDGES:
                    if confidence[start] > 0 and confidence[end] > 0:
                        cv2.line(canvas, points[start], points[end], color, 2, cv2.LINE_AA)
                for index, point in enumerate(points):
                    if confidence[index] > 0:
                        cv2.circle(canvas, point, 3, color, -1, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()
    command = [
        args.ffmpeg,
        "-y",
        "-i",
        str(temporary),
        *browser_video_args(
            args.codec,
            preset=args.preset,
            bitrate=args.bitrate,
        ),
        str(output),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr[-2000:]) from exc
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None) -> None:
    render(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
