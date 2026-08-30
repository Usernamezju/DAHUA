"""Render NTU RGB+D skeleton streams as frontal/top-down temporal contact sheets."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


NTU_BONES = (
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
)
OPENPOSE18_BONES = (
    (1, 0), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (0, 14), (14, 16), (0, 15), (15, 17),
)
BODY_COLORS = ((39, 174, 255), (255, 190, 49))


def load_ntu_skeleton(path: Path) -> np.ndarray:
    """Return [T, M, 25, 3] xyz; absent joints are NaN."""

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        line = stream.readline()
        if not line:
            raise ValueError(f"empty NTU skeleton: {path}")
        frame_count = int(line.strip())
        frames: List[List[np.ndarray]] = []
        for _ in range(frame_count):
            body_count = int(stream.readline().strip())
            bodies = []
            for _ in range(body_count):
                stream.readline()  # body metadata
                joint_count = int(stream.readline().strip())
                joints = np.full((25, 3), np.nan, dtype=np.float32)
                for joint_index in range(joint_count):
                    values = stream.readline().split()
                    if joint_index < 25 and len(values) >= 3:
                        joints[joint_index] = [float(values[0]), float(values[1]), float(values[2])]
                bodies.append(joints)
            frames.append(bodies)
    max_bodies = min(2, max((len(frame) for frame in frames), default=0))
    if max_bodies == 0:
        raise ValueError(f"NTU skeleton contains no body: {path}")
    result = np.full((frame_count, max_bodies, 25, 3), np.nan, dtype=np.float32)
    for frame_index, bodies in enumerate(frames):
        ranked = sorted(
            bodies,
            key=lambda body: int(np.isfinite(body[:, 0]).sum()),
            reverse=True,
        )[:max_bodies]
        for body_index, body in enumerate(ranked):
            result[frame_index, body_index] = body
    return result


def load_kinetics_skeleton(path: Path) -> np.ndarray:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("data", [])
    if not frames:
        raise ValueError(f"Kinetics skeleton contains no frame: {path}")
    max_bodies = min(2, max((len(frame.get("skeleton", [])) for frame in frames), default=0))
    if max_bodies == 0:
        raise ValueError(f"Kinetics skeleton contains no body: {path}")
    result = np.full((len(frames), max_bodies, 18, 3), np.nan, dtype=np.float32)
    for frame_index, frame in enumerate(frames):
        bodies = sorted(
            frame.get("skeleton", []),
            key=lambda body: float(np.mean(body.get("score", [0.0]))),
            reverse=True,
        )[:max_bodies]
        for body_index, body in enumerate(bodies):
            pose = np.asarray(body.get("pose", []), dtype=np.float32)
            score = np.asarray(body.get("score", []), dtype=np.float32)
            if pose.size != 36 or score.size != 18:
                continue
            xy = pose.reshape(18, 2)
            valid = (score > 0.05) & np.isfinite(xy).all(axis=1) & (np.abs(xy).sum(axis=1) > 0)
            result[frame_index, body_index, valid, :2] = xy[valid]
    return result


def load_skeleton(path: Path):
    if path.suffix.lower() == ".skeleton":
        return load_ntu_skeleton(path), NTU_BONES, True, "ntu25_3d"
    if path.suffix.lower() == ".json":
        return load_kinetics_skeleton(path), OPENPOSE18_BONES, False, "openpose18_2d"
    raise ValueError(f"unsupported skeleton format: {path}")


def _uniform_indices(length: int, count: int) -> List[int]:
    if length <= 1:
        return [0]
    return sorted({int(round(value)) for value in np.linspace(0, length - 1, count)})


def select_indices(points: np.ndarray, total: int, uniform: int, motion: int) -> List[int]:
    frame_count = points.shape[0]
    selected = set(_uniform_indices(frame_count, uniform))
    centers = np.nanmean(points[:, :, 0, :], axis=1)
    scores = []
    for index in range(1, frame_count):
        delta = centers[index] - centers[index - 1]
        score = float(np.linalg.norm(np.nan_to_num(delta, nan=0.0)))
        scores.append((score, index))
    minimum_gap = max(1, frame_count // max(total * 3, 1))
    for _, index in sorted(scores, reverse=True):
        if all(abs(index - old) >= minimum_gap for old in selected):
            selected.add(index)
        if len(selected) >= uniform + motion:
            break
    for index in _uniform_indices(frame_count, total * 2):
        selected.add(index)
        if len(selected) >= total:
            break
    return sorted(selected)[:total]


def _bounds(values: np.ndarray, axes: Tuple[int, int]) -> Tuple[float, float, float, float]:
    x = values[..., axes[0]]
    y = values[..., axes[1]]
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return -1.0, 1.0, -1.0, 1.0
    x0, x1 = float(x[finite].min()), float(x[finite].max())
    y0, y1 = float(y[finite].min()), float(y[finite].max())
    x_pad = max(0.15, (x1 - x0) * 0.12)
    y_pad = max(0.15, (y1 - y0) * 0.12)
    return x0 - x_pad, x1 + x_pad, y0 - y_pad, y1 + y_pad


def _project(
    point: np.ndarray,
    axes: Tuple[int, int],
    bounds: Tuple[float, float, float, float],
    box: Tuple[int, int, int, int],
) -> Tuple[int, int]:
    x0, x1, y0, y1 = bounds
    left, top, right, bottom = box
    x = (float(point[axes[0]]) - x0) / max(x1 - x0, 1e-6)
    y = (float(point[axes[1]]) - y0) / max(y1 - y0, 1e-6)
    return int(left + x * (right - left)), int(bottom - y * (bottom - top))


def _draw_projection(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    frame_index: int,
    axes: Tuple[int, int],
    bounds: Tuple[float, float, float, float],
    box: Tuple[int, int, int, int],
    trail: bool,
    bones: Sequence[Tuple[int, int]],
):
    for body_index in range(points.shape[1]):
        color = BODY_COLORS[body_index % len(BODY_COLORS)]
        body = points[frame_index, body_index]
        for first, second in bones:
            if np.isfinite(body[[first, second]][:, list(axes)]).all():
                draw.line(
                    [_project(body[first], axes, bounds, box), _project(body[second], axes, bounds, box)],
                    fill=color,
                    width=3,
                )
        for joint in body:
            if np.isfinite(joint[list(axes)]).all():
                x, y = _project(joint, axes, bounds, box)
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        if trail:
            history = []
            for old_index in range(max(0, frame_index - 8), frame_index + 1):
                root = points[old_index, body_index, 0]
                if np.isfinite(root[list(axes)]).all():
                    history.append(_project(root, axes, bounds, box))
            if len(history) > 1:
                draw.line(history, fill=color, width=2)


def create_skeleton_contact_sheets(path: Path, output_dir: Path, config: Dict[str, Any]):
    output_dir.mkdir(parents=True, exist_ok=True)
    points, bones, has_depth, skeleton_format = load_skeleton(path)
    total = int(config["total_frames"])
    indices = select_indices(
        points,
        total,
        int(config["uniform_frames"]),
        int(config["motion_frames"]),
    )
    width = int(config["cell_width"])
    height = int(config["cell_height"])
    per_sheet = int(config["frames_per_sheet"])
    quality = int(config.get("jpeg_quality", 90))
    front_bounds = _bounds(points, (0, 1))
    top_bounds = _bounds(points, (0, 2)) if has_depth else None
    font = ImageFont.load_default()
    cells: List[Image.Image] = []
    for frame_index in indices:
        image = Image.new("RGB", (width, height), (15, 18, 25))
        draw = ImageDraw.Draw(image)
        midpoint = width // 2
        front_box = (8, 24, midpoint - 5, height - 30) if has_depth else (8, 24, width - 8, height - 30)
        top_box = (midpoint + 5, 24, width - 8, height - 30)
        draw.text((8, 6), "front: x/y + root trail", fill=(210, 215, 225), font=font)
        draw.rectangle(front_box, outline=(60, 65, 78))
        _draw_projection(draw, points, frame_index, (0, 1), front_bounds, front_box, True, bones)
        if has_depth:
            draw.text((midpoint + 5, 6), "top: x/z + root trail", fill=(210, 215, 225), font=font)
            draw.rectangle(top_box, outline=(60, 65, 78))
            _draw_projection(draw, points, frame_index, (0, 2), top_bounds, top_box, True, bones)
        draw.rectangle((0, height - 27, width, height), fill=(0, 0, 0))
        draw.text(
            (8, height - 20),
            f"pose-only | frame={frame_index}/{points.shape[0] - 1}",
            fill=(255, 255, 255),
            font=font,
        )
        cells.append(image)
    paths = []
    for offset in range(0, len(cells), per_sheet):
        chunk = cells[offset : offset + per_sheet]
        sheet = Image.new("RGB", (width * len(chunk), height), (15, 18, 25))
        for column, cell in enumerate(chunk):
            sheet.paste(cell, (column * width, 0))
        sheet_path = output_dir / f"sheet_{len(paths):02d}.jpg"
        sheet.save(sheet_path, "JPEG", quality=quality, optimize=True)
        paths.append(sheet_path)
    metadata = {
        "source": str(path),
        "modality": "skeleton",
        "frame_count": int(points.shape[0]),
        "body_slots": int(points.shape[1]),
        "selected_frames": indices,
        "contact_sheets": [str(item) for item in paths],
        "projections": ["front_xy", "top_xz_with_root_trail"] if has_depth else ["front_xy"],
        "skeleton_format": skeleton_format,
    }
    return paths, metadata
