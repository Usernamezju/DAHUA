"""Select a small, stratified video subset from the 358-sample Campus6 set.

The deep-compression annotation file stores a feature ``frame_dir`` rather
than an absolute RGB path.  This utility resolves each frame directory to the
corresponding probe video and writes a JSONL list without touching any
existing skeleton artifacts.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=10)
    return parser


def _resolve_videos(annotations: List[dict], video_root: Path) -> List[dict]:
    by_stem = {path.stem: path for path in video_root.rglob("*.mp4")}
    resolved = []
    missing = []
    for annotation in annotations:
        frame_dir = str(annotation["frame_dir"])
        stem = Path(frame_dir).name
        source_stem = stem.rsplit("__", 1)[0]
        path = by_stem.get(source_stem) or by_stem.get(stem)
        if path is None:
            missing.append(frame_dir)
            continue
        resolved.append(
            {
                "frame_dir": frame_dir,
                "label_id": int(annotation["label"]),
                "video": str(path.resolve()),
            }
        )
    if missing:
        raise FileNotFoundError(
            f"could not resolve {len(missing)} of {len(annotations)} samples; "
            f"first missing frame_dir={missing[0]}"
        )
    return resolved


def _stratified(items: List[dict], count: int) -> List[dict]:
    if count < 1:
        raise ValueError("count must be positive")
    groups: Dict[int, List[dict]] = defaultdict(list)
    for item in sorted(items, key=lambda value: value["frame_dir"]):
        groups[item["label_id"]].append(item)
    if count < len(groups):
        raise ValueError(f"count={count} is smaller than {len(groups)} classes")
    selected: list[dict] = []
    cursors = {label: 0 for label in sorted(groups)}
    while len(selected) < count:
        progressed = False
        for label in sorted(groups):
            index = cursors[label]
            if index >= len(groups[label]):
                continue
            selected.append(groups[label][index])
            cursors[label] += 1
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError(f"requested {count} samples but only {len(items)} resolved")
    return selected


def main(argv: Optional[List[str]] = None) -> None:
    args = _parser().parse_args(argv)
    payload = pickle.loads(Path(args.annotations).read_bytes())
    annotations = list(payload["annotations"])
    if len(annotations) != 358:
        raise ValueError(f"expected the 358-sample annotation set, got {len(annotations)}")
    items = _resolve_videos(annotations, Path(args.video_root))
    selected = _stratified(items, args.count)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in selected),
        encoding="utf-8",
    )
    print(json.dumps({"resolved": len(items), "selected": len(selected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
