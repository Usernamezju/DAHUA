"""Build a deduplicated RGB+skeleton Campus6 candidate manifest and quota report."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .decision import CAMPUS6_LABELS
from .workflow import sample_id_for, write_jsonl


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "multimodal_candidates.json"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg", ".webm"}
NTU_ACTION_RE = re.compile(r"A(\d{3})", re.IGNORECASE)


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _record(
    path: Path,
    modality: str,
    source_dataset: str,
    source_label: str,
    candidate_labels: Sequence[str],
    license_note: str = "",
    source_video_id: str = "",
) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "sample_id": sample_id_for(resolved),
        "path": str(resolved),
        "modality": modality,
        "source_dataset": source_dataset,
        "source_label": source_label,
        "candidate_labels": sorted(set(candidate_labels)),
        "source_role": "retrieval_hint_only",
        "source_video_id": source_video_id,
        "license_note": license_note,
        "size_bytes": resolved.stat().st_size,
    }


def existing_rgb_records(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    path = Path(config["paths"]["existing_rgb_manifest"])
    if not path.is_file():
        return
    mappings = config["existing_rgb_mappings"]
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = f"{row.get('source_dataset', '')}|{row.get('source_label', '')}"
            labels = mappings.get(key)
            video = Path(row.get("path", ""))
            if not labels or not video.is_file():
                continue
            yield _record(
                video,
                "rgb",
                row.get("source_dataset", "unknown"),
                row.get("source_label", "unknown"),
                labels,
                "existing server dataset; retain original terms",
                f"{row.get('source_dataset', 'unknown').lower()}:{video.stem}",
            )


def downloaded_rgb_records(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    root = Path(config["paths"]["download_root"])
    mappings = config["downloaded_rgb_mappings"]
    for key, labels in mappings.items():
        dataset, source_label = key.split("|", 1)
        class_root = root / dataset / source_label
        if not class_root.is_dir():
            continue
        for path in sorted(class_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                yield _record(
                    path,
                    "rgb",
                    dataset,
                    source_label,
                    labels,
                    "downloaded candidate; verify source manifest before publication",
                    (
                        f"youtube:{path.stem}"
                        if dataset in {"kinetics700", "web_cc"}
                        else f"{dataset}:{path.stem}"
                    ),
                )


def uav_human_rgb_records(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Load the downloaded UAV-Human action subsets as broad RGB candidates."""

    root_value = config.get("paths", {}).get("uav_human_rgb_root")
    if not root_value:
        return
    root = Path(root_value)
    mappings = {
        key.upper(): value for key, value in config.get("uav_human_mappings", {}).items()
    }
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        match = NTU_ACTION_RE.search(path.parent.name) or NTU_ACTION_RE.search(path.stem)
        if not match:
            continue
        action = f"A{match.group(1)}".upper()
        labels = mappings.get(action)
        if not labels:
            continue
        yield _record(
            path,
            "rgb",
            "UAV-Human",
            action,
            labels,
            "UAV-Human academic-research terms apply",
            f"uav-human:{path.stem}",
        )


def hmdb51_push_records(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Load the extracted HMDB51 push class without treating its class as truth."""

    root_value = config.get("paths", {}).get("hmdb51_push_root")
    if not root_value:
        return
    root = Path(root_value)
    labels = config.get("hmdb51_push_mappings", {}).get("push", [])
    if not root.is_dir() or not labels:
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield _record(
                path,
                "rgb",
                "HMDB51",
                "push",
                labels,
                "HMDB51 academic-research terms apply",
                f"hmdb51:{path.stem}",
            )


def ntu_skeleton_records(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    root = Path(config["paths"]["ntu_skeleton_root"])
    mappings = {key.upper(): value for key, value in config["ntu_skeleton_mappings"].items()}
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*.skeleton")):
        match = NTU_ACTION_RE.search(path.stem)
        if not match:
            continue
        action = f"A{match.group(1)}".upper()
        labels = mappings.get(action)
        if labels:
            yield _record(
                path,
                "skeleton",
                "NTU120",
                action,
                labels,
                "NTU RGB+D 120 academic-research terms apply",
                f"ntu120:{path.stem}",
            )


def kinetics_skeleton_records(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    root = Path(config["paths"]["kinetics_skeleton_root"])
    mappings = config.get("kinetics_skeleton_mappings", {})
    if not root.is_dir():
        return
    for split in ("train", "val"):
        labels_path = root / f"kinetics_{split}_label.json"
        samples_root = root / f"kinetics_{split}"
        if not labels_path.is_file() or not samples_root.is_dir():
            continue
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        for sample_name, metadata in sorted(labels.items()):
            source_label = metadata.get("label", "")
            candidate_labels = mappings.get(source_label)
            sample_path = samples_root / f"{sample_name}.json"
            if not candidate_labels or not metadata.get("has_skeleton") or not sample_path.is_file():
                continue
            yield _record(
                sample_path,
                "skeleton",
                "Kinetics-Skeleton",
                source_label,
                candidate_labels,
                "Kinetics source-video terms apply; pose extraction only",
                f"youtube:{sample_name}",
            )


def merge_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for record in records:
        key = record["path"]
        if key in merged:
            labels = set(merged[key]["candidate_labels"]) | set(record["candidate_labels"])
            merged[key]["candidate_labels"] = sorted(labels)
        else:
            merged[key] = record
    return sorted(merged.values(), key=lambda item: (item["modality"], item["sample_id"]))


def enforce_cross_modality_disjoint(records: Sequence[Dict[str, Any]]):
    """Prefer skeleton records and exclude RGB derived from the same source video."""

    skeleton_ids = {
        record.get("source_video_id")
        for record in records
        if record.get("modality") == "skeleton" and record.get("source_video_id")
    }
    excluded = [
        record
        for record in records
        if record.get("modality") == "rgb"
        and record.get("source_video_id") in skeleton_ids
    ]
    excluded_paths = {record["path"] for record in excluded}
    kept = [record for record in records if record["path"] not in excluded_paths]
    kept_rgb_ids = {
        record.get("source_video_id")
        for record in kept
        if record.get("modality") == "rgb" and record.get("source_video_id")
    }
    kept_skeleton_ids = {
        record.get("source_video_id")
        for record in kept
        if record.get("modality") == "skeleton" and record.get("source_video_id")
    }
    overlap = sorted(kept_rgb_ids & kept_skeleton_ids)
    if overlap:
        raise RuntimeError(f"cross-modality source overlap remains: {overlap[:5]}")
    return kept, excluded


def quota_report(
    records: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    excluded_cross_modality: int = 0,
) -> Dict[str, Any]:
    rgb = Counter()
    skeleton = Counter()
    for record in records:
        target = rgb if record["modality"] == "rgb" else skeleton
        for label in record["candidate_labels"]:
            target[label] += 1
    rgb_min = int(config["quotas"]["minimum_rgb_per_class"])
    combined_min = int(config["quotas"]["minimum_combined_per_class"])
    classes = {}
    for label in CAMPUS6_LABELS:
        combined = rgb[label] + skeleton[label]
        classes[label] = {
            "rgb": rgb[label],
            "skeleton": skeleton[label],
            "combined": combined,
            "rgb_shortfall": max(0, rgb_min - rgb[label]),
            "combined_shortfall": max(0, combined_min - combined),
            "ready": rgb[label] >= rgb_min and combined >= combined_min,
        }
    return {
        "schema_version": config["schema_version"],
        "unique_candidates": len(records),
        "minimum_rgb_per_class": rgb_min,
        "minimum_combined_per_class": combined_min,
        "classes": classes,
        "cross_modality_disjoint": True,
        "cross_modality_rgb_excluded": excluded_cross_modality,
        "complete": all(value["ready"] for value in classes.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output")
    parser.add_argument("--summary")
    parser.add_argument("--require-quotas", action="store_true")
    parser.add_argument(
        "--only-source-dataset",
        action="append",
        default=[],
        help=(
            "Keep only this exact source_dataset value; repeat the option for an "
            "incremental multi-source manifest"
        ),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(Path(args.config))
    root = Path(config["paths"]["candidate_root"])
    output = Path(args.output or root / "manifests" / "multimodal_candidates.jsonl")
    summary_path = Path(args.summary or root / "manifests" / "candidate_summary.json")
    merged = merge_records(
        list(existing_rgb_records(config))
        + list(downloaded_rgb_records(config))
        + list(uav_human_rgb_records(config))
        + list(hmdb51_push_records(config))
        + list(ntu_skeleton_records(config))
        + list(kinetics_skeleton_records(config))
    )
    records, excluded = enforce_cross_modality_disjoint(merged)
    if args.only_source_dataset:
        selected_sources = set(args.only_source_dataset)
        records = [
            record for record in records if record["source_dataset"] in selected_sources
        ]
        excluded = [
            record for record in excluded if record["source_dataset"] in selected_sources
        ]
    report = quota_report(records, config, excluded_cross_modality=len(excluded))
    write_jsonl(output, records)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(output), "summary": str(summary_path), **report}, ensure_ascii=False))
    return 3 if args.require_quotas and not report["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
