"""Detect up to two people with NanoDet and emit a group-cropped video."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np

from dahua_cup.person_localization.grouping import (
    build_group_crop,
    crop_and_letterbox,
    select_people,
)
from dahua_cup.person_localization.nanodet_backend import (
    NanoDetPersonDetector,
)
from dahua_cup.pipeline.common import log_event, require_file
from dahua_cup.pipeline.video_encoding import browser_video_args


SCHEMA_VERSION = "nanodet_group_crop.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument(
        "--config",
        default=os.environ.get("DAHUA_NANODET_CONFIG"),
        help="Official NanoDet YAML config (or DAHUA_NANODET_CONFIG)",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("DAHUA_NANODET_CHECKPOINT"),
        help="NanoDet checkpoint (or DAHUA_NANODET_CHECKPOINT)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DAHUA_NANODET_DEVICE", "cpu"),
    )
    parser.add_argument("--person-class-id", type=int)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--minimum-area-fraction", type=float, default=0.0001)
    parser.add_argument("--context-factor", type=float, default=1.20)
    parser.add_argument("--minimum-crop-fraction", type=float, default=0.20)
    parser.add_argument("--output-size", type=int, default=640)
    parser.add_argument(
        "--crowd-policy",
        choices=("top2", "full-frame"),
        default="full-frame",
        help="Fallback when more than two valid people are detected",
    )
    parser.add_argument(
        "--ffmpeg", default=os.environ.get("DAHUA_FFMPEG", "ffmpeg")
    )
    parser.add_argument(
        "--codec",
        default=os.environ.get("DAHUA_PREVIEW_CODEC", "libopenh264"),
    )
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--bitrate", default="3M")
    return parser


def _writer(
    ffmpeg: str,
    destination: Path,
    size: int,
    fps: float,
    codec: str,
    preset: str,
    bitrate: str,
) -> subprocess.Popen:
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{size}x{size}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
    ]
    command.extend(
        browser_video_args(codec, preset, quality=23, bitrate=bitrate)
    )
    command.append(str(destination))
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def process_video(args, detector=None) -> dict:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("NanoDet video cropping requires OpenCV") from exc
    video_path = require_file(args.video, "input video")
    if detector is None:
        if not args.config or not args.checkpoint:
            raise ValueError(
                "--config/--checkpoint or DAHUA_NANODET_* are required"
            )
        detector = NanoDetPersonDetector(
            args.config,
            args.checkpoint,
            device=args.device,
            person_class_id=args.person_class_id,
        )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open NanoDet input video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError(f"invalid video metadata: {video_path}")

    output = Path(args.output).expanduser().resolve()
    metadata_path = Path(args.metadata).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.stem + ".tmp.mp4")
    writer = _writer(
        args.ffmpeg,
        temporary_output,
        args.output_size,
        fps,
        args.codec,
        args.preset,
        args.bitrate,
    )
    if writer.stdin is None:
        capture.release()
        raise RuntimeError("FFmpeg did not expose a video input pipe")

    frames = []
    person_histogram = {"0": 0, "1": 0, "2": 0}
    crowded_frames = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            candidates = detector.detect(frame)
            selected, overflow = select_people(
                candidates,
                width=width,
                height=height,
                confidence_threshold=args.confidence,
                minimum_area_fraction=args.minimum_area_fraction,
                maximum_people=2,
            )
            if overflow:
                crowded_frames += 1
            use_people = selected
            mode = f"{len(selected)}_people"
            if overflow and args.crowd_policy == "full-frame":
                use_people = []
                mode = "crowd_full_frame"
            group = build_group_crop(
                use_people,
                width=width,
                height=height,
                context_factor=args.context_factor,
                minimum_crop_fraction=args.minimum_crop_fraction,
            )
            cropped, transform = crop_and_letterbox(
                frame, group, args.output_size
            )
            writer.stdin.write(cropped.tobytes())
            person_histogram[str(len(selected))] += 1
            frames.append(
                {
                    "frame_index": frame_index,
                    "mode": mode,
                    "candidate_people": len(selected) + overflow,
                    "selected_people": [item.as_dict() for item in selected],
                    "overflow_people": overflow,
                    "group_bbox_xyxy": [float(value) for value in group],
                    "transform": transform,
                }
            )
            frame_index += 1
    except BaseException:
        try:
            writer.stdin.close()
        finally:
            writer.wait()
            if temporary_output.is_file():
                temporary_output.unlink()
        raise
    finally:
        capture.release()
    writer.stdin.close()
    code = writer.wait()
    if code:
        raise RuntimeError(f"FFmpeg NanoDet crop writer failed: {code}")
    if not frames:
        raise RuntimeError(f"video contains no decodable frames: {video_path}")
    temporary_output.replace(output)

    result = {
        "schema_version": SCHEMA_VERSION,
        "video": str(video_path),
        "output": str(output),
        "device": str(args.device),
        "source_width": width,
        "source_height": height,
        "output_size": int(args.output_size),
        "fps": fps,
        "frame_count": frame_index,
        "maximum_people": 2,
        "confidence_threshold": float(args.confidence),
        "context_factor": float(args.context_factor),
        "temporal_smoothing": False,
        "crowd_policy": str(args.crowd_policy),
        "crowded_frames": crowded_frames,
        "crowd_frame_ratio": crowded_frames / frame_index,
        "selected_people_histogram": person_histogram,
        "frames": frames,
    }
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_metadata.replace(metadata_path)
    return result


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be in [0, 1]")
    if not 0.0 <= args.minimum_area_fraction <= 1.0:
        raise ValueError("--minimum-area-fraction must be in [0, 1]")
    if args.context_factor < 1.0:
        raise ValueError("--context-factor must be at least one")
    if not 0.0 < args.minimum_crop_fraction <= 1.0:
        raise ValueError("--minimum-crop-fraction must be in (0, 1]")
    if args.output_size < 32:
        raise ValueError("--output-size must be at least 32")
    result = process_video(args)
    log_event(
        "nanodet_group_crop_complete",
        video=result["video"],
        output=result["output"],
        frames=result["frame_count"],
        crowded_frames=result["crowded_frames"],
        temporal_smoothing=False,
    )


if __name__ == "__main__":
    main()
