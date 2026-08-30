"""Pure helpers for building YOLO track distillation records."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence


SCHEMA_VERSION = "stn_yolo_tracks.v1"


def normalize_xyxy(
    box: Sequence[float], width: int, height: int
) -> list[float]:
    if width <= 0 or height <= 0 or len(box) != 4:
        raise ValueError("box normalization requires xyxy and positive size")
    left, top, right, bottom = map(float, box)
    left = max(0.0, min(float(width), left)) / width
    right = max(0.0, min(float(width), right)) / width
    top = max(0.0, min(float(height), top)) / height
    bottom = max(0.0, min(float(height), bottom)) / height
    if right <= left or bottom <= top:
        raise ValueError("YOLO returned an empty person box")
    return [left, top, right, bottom]


def select_track_ids(
    frames: Sequence[dict],
    *,
    maximum_people: int = 2,
    minimum_coverage: float = 0.20,
) -> list[int]:
    """Select persistent, confident tracks for a zero/one/two-person model."""
    if maximum_people < 1 or not 0 <= minimum_coverage <= 1:
        raise ValueError("invalid track-selection configuration")
    frame_count = max(1, len(frames))
    observations = defaultdict(list)
    for frame in frames:
        observed = set()
        for person in frame.get("persons") or []:
            track_id = int(person["track_id"])
            if track_id in observed:
                continue
            observed.add(track_id)
            observations[track_id].append(float(person["confidence"]))
    ranked = []
    for track_id, scores in observations.items():
        coverage = len(scores) / frame_count
        if coverage < minimum_coverage:
            continue
        mean_confidence = sum(scores) / len(scores)
        ranked.append(
            (track_id, coverage, mean_confidence, len(scores))
        )
    ranked.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return [item[0] for item in ranked[:maximum_people]]


def interpolate_track_gaps(
    frames: Sequence[dict],
    selected_track_ids: Sequence[int],
    *,
    maximum_gap: int = 3,
) -> list[dict]:
    """Linearly fill short YOLO tracking gaps without inventing long tracks."""
    if maximum_gap < 0:
        raise ValueError("maximum_gap must be non-negative")
    selected = {int(value) for value in selected_track_ids}
    output = [
        {
            "frame_index": int(frame["frame_index"]),
            "persons": [
                dict(person)
                for person in frame.get("persons") or []
                if int(person["track_id"]) in selected
            ],
        }
        for frame in frames
    ]
    by_track = defaultdict(dict)
    for frame_position, frame in enumerate(output):
        for person in frame["persons"]:
            by_track[int(person["track_id"])][frame_position] = person
    for track_id, observations in by_track.items():
        positions = sorted(observations)
        for left_position, right_position in zip(
            positions[:-1], positions[1:]
        ):
            gap = right_position - left_position - 1
            if gap <= 0 or gap > maximum_gap:
                continue
            left = observations[left_position]
            right = observations[right_position]
            for offset in range(1, gap + 1):
                ratio = offset / (gap + 1)
                box = [
                    (1.0 - ratio) * first + ratio * second
                    for first, second in zip(
                        left["bbox_xyxy"], right["bbox_xyxy"]
                    )
                ]
                output[left_position + offset]["persons"].append(
                    {
                        "track_id": track_id,
                        "bbox_xyxy": box,
                        "confidence": min(
                            float(left["confidence"]),
                            float(right["confidence"]),
                        ),
                        "interpolated": True,
                    }
                )
    for frame in output:
        frame["persons"].sort(key=lambda item: int(item["track_id"]))
    return output


def crowd_frame_ratio(frames: Sequence[dict], maximum_people: int = 2) -> float:
    if not frames:
        return 0.0
    crowded = sum(
        len(frame.get("persons") or []) > maximum_people for frame in frames
    )
    return crowded / len(frames)


def validate_teacher_record(record: dict) -> None:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported STN teacher schema")
    if not record.get("sample_id") or not record.get("video_path"):
        raise ValueError("teacher record requires sample_id and video_path")
    frames = record.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("teacher record contains no frames")
    previous = -1
    for frame in frames:
        frame_index = int(frame["frame_index"])
        if frame_index <= previous:
            raise ValueError("teacher frame indices must increase")
        previous = frame_index
        people = frame.get("persons") or []
        if len(people) > 2:
            raise ValueError("teacher record exceeds two selected people")
        for person in people:
            box = list(map(float, person["bbox_xyxy"]))
            if (
                len(box) != 4
                or not 0 <= box[0] < box[2] <= 1
                or not 0 <= box[1] < box[3] <= 1
            ):
                raise ValueError("teacher box must be normalized xyxy")
            confidence = float(person["confidence"])
            if not 0 <= confidence <= 1:
                raise ValueError("teacher confidence must be in [0,1]")
