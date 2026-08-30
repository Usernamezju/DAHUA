#!/usr/bin/env python3
"""Selectively extract UAV-Human action classes from a remote ZIP archive."""

import argparse
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from remotezip import RemoteZip


def emit(event: str, **fields: object) -> None:
    payload = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download selected UAV-Human action classes via HTTP Range requests."
    )
    parser.add_argument("--url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", default="A076,A133")
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    return parser.parse_args()


def open_archive(url: str) -> RemoteZip:
    return RemoteZip(url, timeout=120)


def main() -> None:
    args = parse_args()
    labels = tuple(item.strip() for item in args.labels.split(",") if item.strip())
    if not labels:
        raise SystemExit("at least one action label is required")

    action_pattern = re.compile(r"(" + "|".join(map(re.escape, labels)) + r")R\d")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    archive = open_archive(args.url)
    try:
        selected = []
        for info in archive.infolist():
            match = action_pattern.search(info.filename)
            if match and not info.is_dir():
                selected.append((info.filename, match.group(1), info.file_size))

        selected.sort(key=lambda item: item[0])
        emit(
            "uavhuman_subset_start",
            labels=list(labels),
            members=len(selected),
            expected_bytes=sum(item[2] for item in selected),
            output_dir=str(args.output_dir),
        )

        completed = 0
        skipped = 0
        failed = []
        for index, (member_name, label, expected_size) in enumerate(selected, start=1):
            target_dir = args.output_dir / label
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / Path(member_name).name
            partial = target.with_suffix(target.suffix + ".part")

            if target.is_file() and target.stat().st_size == expected_size:
                skipped += 1
                emit(
                    "uavhuman_subset_progress",
                    current=index,
                    total=len(selected),
                    label=label,
                    file=target.name,
                    status="skipped_existing",
                )
                continue

            success = False
            for attempt in range(1, args.max_attempts + 1):
                try:
                    with archive.open(member_name) as source, partial.open("wb") as destination:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
                    actual_size = partial.stat().st_size
                    if actual_size != expected_size:
                        raise IOError(
                            f"size mismatch for {member_name}: {actual_size} != {expected_size}"
                        )
                    os.replace(partial, target)
                    completed += 1
                    success = True
                    emit(
                        "uavhuman_subset_progress",
                        current=index,
                        total=len(selected),
                        label=label,
                        file=target.name,
                        bytes=actual_size,
                        status="downloaded",
                    )
                    break
                except Exception as exc:
                    partial.unlink(missing_ok=True)
                    emit(
                        "uavhuman_subset_retry",
                        current=index,
                        total=len(selected),
                        file=target.name,
                        attempt=attempt,
                        error=str(exc),
                    )
                    try:
                        archive.close()
                    except Exception:
                        pass
                    if attempt < args.max_attempts:
                        time.sleep(args.retry_delay * attempt)
                        archive = open_archive(args.url)

            if not success:
                failed.append(member_name)

        emit(
            "uavhuman_subset_complete",
            members=len(selected),
            downloaded=completed,
            skipped=skipped,
            failed=len(failed),
            failed_members=failed,
        )
        if failed:
            raise SystemExit(1)
    finally:
        archive.close()


if __name__ == "__main__":
    main()
