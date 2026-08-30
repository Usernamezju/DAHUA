"""Download/extract resumable RGB candidate sources for Campus6 screening."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "multimodal_candidates.json"


def emit(event: str, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False, sort_keys=True), flush=True)


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def executable(name: str) -> str:
    environment_name = {
        "yt-dlp": "DAHUA_YTDLP",
        "ffmpeg": "DAHUA_FFMPEG",
    }.get(name)
    if environment_name:
        configured = os.environ.get(environment_name, "").strip()
        if configured and Path(configured).is_file():
            return configured
    value = shutil.which(name)
    if not value:
        raise RuntimeError(f"required executable is missing: {name}")
    return value


def proxy_arguments() -> List[str]:
    """Return an explicit yt-dlp proxy so its ffmpeg downloader uses it too."""

    for name in (
        "DAHUA_DOWNLOAD_PROXY",
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
    ):
        value = os.environ.get(name, "").strip()
        if value:
            return ["--proxy", value]
    return []


def ytdlp_network_arguments() -> List[str]:
    """Use conservative retries and pacing for a tunneled residential proxy."""

    return [
        *proxy_arguments(),
        "--socket-timeout",
        "45",
        "--retries",
        "5",
        "--fragment-retries",
        "5",
        "--extractor-retries",
        "5",
        "--retry-sleep",
        "3",
        "--sleep-requests",
        "1",
    ]


def video_count(path: Path) -> int:
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in extensions)


def kinetics_skeleton_source_ids(config: Dict[str, Any]) -> Set[str]:
    root = Path(config["paths"]["kinetics_skeleton_root"])
    source_ids: Set[str] = set()
    for split in ("train", "val"):
        labels_path = root / f"kinetics_{split}_label.json"
        if labels_path.is_file():
            source_ids.update(json.loads(labels_path.read_text(encoding="utf-8")).keys())
    return source_ids


def prepare_ytdlp_exclusion_archive(config: Dict[str, Any]) -> Path:
    """Exclude skeleton source videos and already downloaded RGB across searches."""

    archive = Path(config["paths"]["candidate_root"]) / "manifests" / "yt_dlp_exclusions.txt"
    entries = {f"youtube {source_id}" for source_id in kinetics_skeleton_source_ids(config)}
    if archive.is_file():
        entries.update(line.strip() for line in archive.read_text(encoding="utf-8").splitlines() if line.strip())
    download_root = Path(config["paths"]["download_root"])
    for dataset in ("kinetics700", "web_cc"):
        dataset_root = download_root / dataset
        if not dataset_root.is_dir():
            continue
        for path in dataset_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".webm"}:
                entries.add(f"youtube {path.stem}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    temporary.write_text("\n".join(sorted(entries)) + "\n", encoding="utf-8")
    os.replace(temporary, archive)
    return archive


def extract_hmdb51(config: Dict[str, Any], dry_run: bool = False):
    data_root = Path(config["paths"]["data_root"])
    archive = data_root / "datasets" / "OpenDataLab___HMDB51" / "raw" / "VIDEO DATABASE" / "hmdb51_org.rar"
    destination = Path(config["paths"]["download_root"]) / "hmdb51"
    labels = ("walk", "run", "push")
    if not archive.is_file():
        emit("candidate_download_skip", source="hmdb51", reason=f"archive missing: {archive}")
        return
    if all(video_count(destination / label) >= 90 for label in labels):
        emit("candidate_download_resume", source="hmdb51", status="already_extracted")
        return
    if dry_run:
        emit("candidate_download_plan", source="hmdb51", archive=str(archive), labels=labels)
        return
    seven_zip = shutil.which("7z") or shutil.which("7zz")
    if not seven_zip:
        emit(
            "candidate_download_skip",
            source="hmdb51",
            reason="7z/7zz is missing; KTH already supplies the normal-class RGB floor",
        )
        return
    staging = destination.parent / ".hmdb51_archives"
    staging.mkdir(parents=True, exist_ok=True)
    subprocess.run([seven_zip, "x", "-y", f"-o{staging}", str(archive)], check=True)
    for label in labels:
        target = destination / label
        target.mkdir(parents=True, exist_ok=True)
        archives = sorted(staging.rglob(f"{label}.rar"))
        if not archives:
            raise RuntimeError(f"HMDB51 inner archive not found: {label}.rar")
        subprocess.run([seven_zip, "x", "-y", f"-o{target}", str(archives[0])], check=True)
        emit("candidate_download_progress", source="hmdb51", label=label, videos=video_count(target))


def download_file(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "DAHUA-Campus6/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    os.replace(temporary, destination)


def _kinetics_rows(
    annotation: Path,
    wanted: Iterable[str],
    excluded_video_ids: Iterable[str] = (),
) -> Dict[str, List[Dict[str, str]]]:
    groups = {label: [] for label in wanted}
    excluded = set(excluded_video_ids)
    with annotation.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            label = row.get("label", "")
            if label in groups and row.get("youtube_id") not in excluded:
                groups[label].append(row)
    for label in groups:
        groups[label].sort(
            key=lambda row: hashlib.sha1(row["youtube_id"].encode("ascii")).hexdigest()
        )
    return groups


def _download_kinetics_clip(yt_dlp: str, row: Dict[str, str], output: Path) -> bool:
    if any(output.with_suffix(suffix).is_file() for suffix in (".mp4", ".mkv", ".webm")):
        return True
    start = float(row["time_start"])
    end = float(row["time_end"])
    command = [
        yt_dlp,
        *ytdlp_network_arguments(),
        "--ffmpeg-location",
        executable("ffmpeg"),
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--download-sections",
        f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "--max-filesize",
        "120M",
        "-f",
        "bv*[height<=720]+ba/b[height<=720]",
        "--merge-output-format",
        "mp4",
        "-o",
        str(output.with_suffix(".%(ext)s")),
        f"https://www.youtube.com/watch?v={row['youtube_id']}",
    ]
    return subprocess.run(command, check=False).returncode == 0


def download_kinetics(config: Dict[str, Any], dry_run: bool = False):
    root = Path(config["paths"]["download_root"])
    destination = root / "kinetics700"
    annotation = root / "metadata" / "kinetics700_train.csv"
    labels = list(config["download"].get("kinetics_labels", []))
    if not labels:
        labels = sorted(
            key.split("|", 1)[1]
            for key in config["downloaded_rgb_mappings"]
            if key.startswith("kinetics700|")
        )
    target = int(config["download"]["kinetics_per_label"])
    request_interval = float(config["download"].get("request_interval_seconds", 1.0))
    if dry_run:
        emit("candidate_download_plan", source="kinetics700", labels=labels, target_per_label=target)
        return
    yt_dlp = executable("yt-dlp")
    if not annotation.is_file():
        download_file(config["download"]["kinetics_annotation_url"], annotation)
    excluded_video_ids = kinetics_skeleton_source_ids(config)
    groups = _kinetics_rows(annotation, labels, excluded_video_ids)
    emit(
        "candidate_cross_modality_exclusion",
        source="kinetics700",
        skeleton_source_ids=len(excluded_video_ids),
    )
    for label, rows in groups.items():
        class_root = destination / label
        class_root.mkdir(parents=True, exist_ok=True)
        accepted = video_count(class_root)
        for row in rows:
            if accepted >= target:
                break
            output = class_root / row["youtube_id"]
            if _download_kinetics_clip(yt_dlp, row, output):
                accepted = video_count(class_root)
                emit(
                    "candidate_download_progress",
                    source="kinetics700",
                    label=label,
                    videos=accepted,
                    target=target,
                )
            if request_interval > 0:
                time.sleep(request_interval)
        if accepted < target:
            emit(
                "candidate_download_shortfall",
                source="kinetics700",
                label=label,
                videos=accepted,
                target=target,
                reason="some source videos are unavailable",
            )


def download_web_cc(config: Dict[str, Any], dry_run: bool = False):
    root = Path(config["paths"]["download_root"]) / "web_cc"
    target = int(config["download"]["web_cc_per_class"])
    queries = config["download"]["web_cc_queries"]
    if dry_run:
        emit("candidate_download_plan", source="web_cc", target_per_class=target, queries=queries)
        return
    yt_dlp = executable("yt-dlp")
    exclusion_archive = prepare_ytdlp_exclusion_archive(config)
    for label, class_queries in queries.items():
        class_root = root / label
        class_root.mkdir(parents=True, exist_ok=True)
        current = video_count(class_root)
        for query_index, query in enumerate(class_queries):
            if current >= target:
                break
            remaining = target - current
            command = [
                yt_dlp,
                *ytdlp_network_arguments(),
                "--ffmpeg-location",
                executable("ffmpeg"),
                "--ignore-errors",
                "--no-warnings",
                "--no-playlist",
                "--restrict-filenames",
                "--write-info-json",
                "--download-archive",
                str(exclusion_archive),
                "--match-filter",
                "license ~= (?i)creative commons & duration >= 3 & duration <= 240",
                "--download-sections",
                "*0-15",
                "--force-keyframes-at-cuts",
                "--max-downloads",
                str(remaining),
                "--max-filesize",
                "120M",
                "-f",
                "bv*[height<=720]+ba/b[height<=720]",
                "--merge-output-format",
                "mp4",
                "-o",
                str(class_root / "%(id)s.%(ext)s"),
                f"ytsearch{max(remaining * 5, 100)}:{query}",
            ]
            return_code = subprocess.run(command, check=False).returncode
            current = video_count(class_root)
            emit(
                "candidate_download_progress",
                source="web_cc",
                label=label,
                query_index=query_index,
                videos=current,
                target=target,
                return_code=return_code,
            )
        if current < target:
            emit(
                "candidate_download_shortfall",
                source="web_cc",
                label=label,
                videos=current,
                target=target,
                reason="insufficient Creative Commons search results",
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--sources", default="hmdb51,kinetics700,web_cc")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(Path(args.config))
    selected = {item.strip() for item in args.sources.split(",") if item.strip()}
    unknown = selected - {"hmdb51", "kinetics700", "web_cc"}
    if unknown:
        raise ValueError(f"unknown sources: {', '.join(sorted(unknown))}")
    emit(
        "candidate_download_start",
        sources=sorted(selected),
        dry_run=args.dry_run,
        proxy_enabled=bool(proxy_arguments()),
    )
    if "hmdb51" in selected:
        extract_hmdb51(config, args.dry_run)
    if "kinetics700" in selected:
        download_kinetics(config, args.dry_run)
    if "web_cc" in selected:
        download_web_cc(config, args.dry_run)
    emit("candidate_download_complete", sources=sorted(selected))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        emit("candidate_download_failed", error=str(exc))
        raise SystemExit(2)
