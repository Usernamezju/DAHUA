"""Create timestamped temporal contact sheets without using a GPU."""

from __future__ import annotations

import math
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


class ContactSheetGenerationError(RuntimeError):
    """A video could not be decoded into reliable contact sheets."""


def _vision_modules():
    """Import native image libraries only in the process that decodes video."""

    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    return cv2, np, Image, ImageDraw, ImageFont


def _read_frame(capture, frame_index: int):
    cv2, _, _, _, _ = _vision_modules()
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    return frame if ok else None


def _uniform_indices(frame_count: int, count: int) -> List[int]:
    _, np, _, _, _ = _vision_modules()
    if frame_count <= 1:
        return [0]
    return sorted({int(round(value)) for value in np.linspace(0, frame_count - 1, count)})


def select_frame_indices(
    video_path: Path,
    total_frames: int,
    uniform_frames: int,
    motion_frames: int,
    scan_frames: int,
) -> Tuple[List[int], float, int]:
    cv2, _, _, _, _ = _vision_modules()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0

    selected = set(_uniform_indices(frame_count, uniform_frames))
    scan_indices = _uniform_indices(frame_count, min(scan_frames, frame_count))
    motion_scores: List[Tuple[float, int]] = []
    previous = None
    for index in scan_indices:
        frame = _read_frame(capture, index)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        if previous is not None:
            score = float(cv2.absdiff(previous, gray).mean())
            motion_scores.append((score, index))
        previous = gray

    minimum_gap = max(1, frame_count // max(total_frames * 3, 1))
    for _, index in sorted(motion_scores, reverse=True):
        if all(abs(index - existing) >= minimum_gap for existing in selected):
            selected.add(index)
        if len(selected) >= uniform_frames + motion_frames:
            break

    if len(selected) < total_frames:
        for index in _uniform_indices(frame_count, total_frames * 2):
            selected.add(index)
            if len(selected) >= total_frames:
                break

    capture.release()
    return sorted(selected)[:total_frames], fps, frame_count


def _letterbox(frame, width: int, height: int) -> Image.Image:
    cv2, _, Image, _, _ = _vision_modules()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    available_height = max(1, height - 28)
    image.thumbnail((width, available_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (16, 18, 24))
    left = (width - image.width) // 2
    top = (available_height - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def create_contact_sheets(video_path: Path, output_dir: Path, config: Dict[str, Any]):
    cv2, _, Image, ImageDraw, ImageFont = _vision_modules()
    output_dir.mkdir(parents=True, exist_ok=True)
    indices, fps, frame_count = select_frame_indices(
        video_path,
        int(config["total_frames"]),
        int(config["uniform_frames"]),
        int(config["motion_frames"]),
        int(config["scan_frames"]),
    )
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    for index in indices:
        frame = _read_frame(capture, index)
        if frame is not None:
            frames.append((index, frame))
    capture.release()
    if not frames:
        raise ValueError(f"video contains no readable frame: {video_path}")

    cell_width = int(config["cell_width"])
    cell_height = int(config["cell_height"])
    per_sheet = int(config["frames_per_sheet"])
    quality = int(config.get("jpeg_quality", 90))
    font = ImageFont.load_default()
    paths: List[Path] = []
    for sheet_index in range(0, len(frames), per_sheet):
        chunk = frames[sheet_index : sheet_index + per_sheet]
        sheet = Image.new("RGB", (cell_width * len(chunk), cell_height), (16, 18, 24))
        draw = ImageDraw.Draw(sheet)
        for column, (frame_index, frame) in enumerate(chunk):
            cell = _letterbox(frame, cell_width, cell_height)
            sheet.paste(cell, (column * cell_width, 0))
            timestamp = frame_index / fps
            caption = f"t={timestamp:06.2f}s  frame={frame_index}"
            x = column * cell_width + 8
            draw.rectangle((column * cell_width, cell_height - 28, (column + 1) * cell_width, cell_height), fill=(0, 0, 0))
            draw.text((x, cell_height - 22), caption, fill=(255, 255, 255), font=font)
        path = output_dir / f"sheet_{len(paths):02d}.jpg"
        sheet.save(path, format="JPEG", quality=quality, optimize=True)
        paths.append(path)

    metadata = {
        "video": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "selected_frames": indices,
        "timestamps_seconds": [round(index / fps, 3) for index in indices],
        "contact_sheets": [str(path) for path in paths],
    }
    return paths, metadata


def create_contact_sheets_isolated(
    video_path: Path,
    output_dir: Path,
    config: Dict[str, Any],
) -> Tuple[List[Path], Dict[str, Any]]:
    """Generate RGB contact sheets in a disposable child process.

    OpenCV decoder faults such as ``corrupted double-linked list`` terminate the
    interpreter at the native-code level, so they cannot be handled with a
    Python ``try/except`` around :func:`create_contact_sheets`.  This wrapper
    lets the child fail while the screening process records the bad candidate
    and continues with the next one.  Outputs are committed only after the
    child exits cleanly and has produced a complete metadata file.
    """

    output_dir = output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    timeout_seconds = int(config.get("worker_timeout_seconds", 180))
    command = [
        sys.executable,
        "-m",
        "dahua_cup.dataset_construction.contact_sheet_worker",
        "--video",
        str(video_path.expanduser().resolve()),
        "--output-dir",
        str(staging_dir),
        "--config-json",
        json.dumps(config, ensure_ascii=False, sort_keys=True),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise ContactSheetGenerationError(
            f"contact-sheet worker timed out after {timeout_seconds}s: {video_path}"
        ) from exc

    if result.returncode != 0:
        shutil.rmtree(staging_dir, ignore_errors=True)
        diagnostic = (result.stderr or result.stdout).strip().replace("\n", " ")
        raise ContactSheetGenerationError(
            f"contact-sheet worker exited {result.returncode} for {video_path.name}"
            + (f": {diagnostic[-1000:]}" if diagnostic else "")
        )

    metadata_path = staging_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        sheets = [Path(path) for path in metadata["contact_sheets"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise ContactSheetGenerationError(
            f"contact-sheet worker returned incomplete output for {video_path.name}"
        ) from exc
    if not sheets or not all(path.is_file() and path.parent == staging_dir for path in sheets):
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise ContactSheetGenerationError(
            f"contact-sheet worker returned missing sheets for {video_path.name}"
        )

    committed_sheets = [output_dir / path.name for path in sheets]
    metadata["contact_sheets"] = [str(path) for path in committed_sheets]
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_metadata, metadata_path)

    previous_dir = None
    if output_dir.exists() or output_dir.is_symlink():
        previous_dir = output_dir.parent / f".{output_dir.name}.previous-{uuid.uuid4().hex}"
        os.replace(output_dir, previous_dir)
    try:
        os.replace(staging_dir, output_dir)
    except OSError:
        if previous_dir is not None and previous_dir.exists():
            os.replace(previous_dir, output_dir)
        raise
    if previous_dir is not None:
        shutil.rmtree(previous_dir, ignore_errors=True)
    return committed_sheets, metadata
