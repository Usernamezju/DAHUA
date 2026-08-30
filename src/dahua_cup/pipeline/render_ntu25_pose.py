"""Render a MediaPipe-to-NTU25 feature artifact as a pose-only MP4."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from dahua_cup.pipeline.common import log_event, require_file
from dahua_cup.pipeline.video_encoding import browser_video_args


NTU25_EDGES = (
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
)
COLORS = ((53, 211, 153), (251, 113, 133))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--bitrate", default="2M")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    return parser


def _world_projection(keypoint, valid_mask, width, height):
    """Lay out older world-coordinate artifacts that have no image joints."""
    projected = np.zeros(keypoint.shape[:-1] + (2,), dtype=np.float32)
    people = keypoint.shape[0]
    for person in range(people):
        valid = valid_mask[person]
        points = keypoint[person, :, :, :2]
        values = points[valid]
        if not values.size:
            continue
        low = np.percentile(values, 2, axis=0)
        high = np.percentile(values, 98, axis=0)
        span = np.maximum(high - low, 1e-4)
        normalized = (points - low) / span
        lane_width = width / people
        projected[person, :, :, 0] = (
            normalized[:, :, 0] * lane_width * 0.72
            + person * lane_width + lane_width * 0.14
        )
        projected[person, :, :, 1] = (
            (1.0 - normalized[:, :, 1]) * height * 0.72 + height * 0.14
        )
    return projected


def load_render_data(path, width, height):
    with np.load(path, allow_pickle=False) as artifact:
        keypoint = np.asarray(artifact["keypoint"], dtype=np.float32)
        if "valid_mask" in artifact:
            valid_mask = np.asarray(artifact["valid_mask"], dtype=bool)
        else:
            valid_mask = np.linalg.norm(keypoint, axis=-1) > 0
        fps = float(np.asarray(artifact["fps"]).item()) if "fps" in artifact else 25.0
        source_width = int(np.asarray(artifact["width"]).item()) if "width" in artifact else width
        source_height = int(np.asarray(artifact["height"]).item()) if "height" in artifact else height
        if "image_keypoint" in artifact:
            points = np.asarray(artifact["image_keypoint"], dtype=np.float32)
            points[..., 0] *= width / max(source_width, 1)
            points[..., 1] *= height / max(source_height, 1)
        else:
            points = _world_projection(keypoint, valid_mask, width, height)
    if points.shape[:-1] != valid_mask.shape or points.shape[-1] != 2:
        raise ValueError("pose artifact has incompatible render arrays")
    return points, valid_mask, fps


def render(args) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("pose rendering requires opencv-python") from exc

    feature = require_file(args.feature, "pose feature")
    points, validity, fps = load_render_data(feature, args.width, args.height)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{args.width}x{args.height}",
        "-r", f"{fps:.6f}", "-i", "-", "-an",
    ]
    command.extend(browser_video_args(
        args.codec, args.preset, quality=23, bitrate=args.bitrate
    ))
    command.append(str(output))
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame_index in range(points.shape[1]):
            canvas = np.full((args.height, args.width, 3), (19, 25, 36), dtype=np.uint8)
            for person in range(points.shape[0]):
                color = COLORS[person % len(COLORS)]
                frame_points = points[person, frame_index]
                frame_valid = validity[person, frame_index]
                for first, second in NTU25_EDGES:
                    if frame_valid[first] and frame_valid[second]:
                        cv2.line(
                            canvas,
                            tuple(np.rint(frame_points[first]).astype(int)),
                            tuple(np.rint(frame_points[second]).astype(int)),
                            color, 3, cv2.LINE_AA,
                        )
                for joint, valid in zip(frame_points, frame_valid):
                    if valid:
                        cv2.circle(canvas, tuple(np.rint(joint).astype(int)), 4, color, -1, cv2.LINE_AA)
                valid_indices = np.flatnonzero(frame_valid)
                if valid_indices.size:
                    anchor = tuple(np.rint(frame_points[valid_indices[0]]).astype(int))
                    cv2.putText(canvas, f"Person {person + 1}", anchor,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            cv2.putText(canvas, f"Frame {frame_index + 1}/{points.shape[1]}", (24, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 218, 230), 2, cv2.LINE_AA)
            process.stdin.write(canvas.tobytes())
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg pose renderer exited with code {return_code}")
    log_event("pose_render_complete", feature=str(feature), output=str(output), frames=points.shape[1])


def main(argv=None):
    render(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
