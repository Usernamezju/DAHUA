"""Command-line entry point for hosted Campus6 dataset screening."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from .decision import CAMPUS6_LABELS
from .workflow import ScreeningWorkflow, discover_directory, load_manifest, write_jsonl


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "hosted_screening.json"


def load_config(path: Path) -> Dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = {"target_per_class", "video_extensions", "contact_sheets", "providers", "request"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"configuration is missing: {', '.join(missing)}")
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Screen Campus6 videos with two blind hosted multimodal reviewers and "
            "an optional Qwen adjudicator."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", help="Directory recursively containing candidate videos")
    source.add_argument("--manifest", help="CSV/JSONL manifest containing at least a path field")
    parser.add_argument("--output", help="Screening output root")
    parser.add_argument("--limit", type=int, help="Process at most N candidates in this run")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only generate temporal contact sheets; do not call hosted models",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover candidates and print the resolved configuration",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run successful reviews even when a compatible result exists",
    )
    parser.add_argument(
        "--stop-when-quotas-met",
        action="store_true",
        help="Stop once every class has >=100 RGB and >=500 combined accepted samples",
    )
    return parser


def balanced_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Round-robin retrieval hints, processing RGB before skeleton without duplication."""

    ordered = []
    used = set()
    for modality in ("rgb", "skeleton"):
        subset = [item for item in candidates if item.get("modality", "rgb") == modality]
        queues = {
            label: [item for item in subset if label in item.get("candidate_labels", [])]
            for label in CAMPUS6_LABELS
        }
        while True:
            added = False
            for label in CAMPUS6_LABELS:
                while queues[label] and queues[label][0]["sample_id"] in used:
                    queues[label].pop(0)
                if queues[label]:
                    item = queues[label].pop(0)
                    used.add(item["sample_id"])
                    ordered.append(item)
                    added = True
            if not added:
                break
        for item in subset:
            if item["sample_id"] not in used:
                used.add(item["sample_id"])
                ordered.append(item)
    return ordered


def quotas_met(records: List[Dict[str, Any]], config: Dict[str, Any]) -> bool:
    rgb_target = int(config.get("minimum_rgb_per_class", 100))
    combined_target = int(config.get("minimum_combined_per_class", 500))
    rgb = {label: 0 for label in CAMPUS6_LABELS}
    combined = {label: 0 for label in CAMPUS6_LABELS}
    for record in records:
        final = record.get("final", {})
        label = final.get("label")
        if final.get("status") != "accepted" or label not in combined:
            continue
        combined[label] += 1
        if record.get("sample", {}).get("modality", "rgb") == "rgb":
            rgb[label] += 1
    return all(rgb[label] >= rgb_target and combined[label] >= combined_target for label in CAMPUS6_LABELS)


def resolve_paths(args, config: Dict[str, Any]):
    paths = config.get("paths", {})
    source = Path(args.manifest or args.input or paths.get("input_root", "data/campus6_candidates"))
    output = Path(args.output or paths.get("output_root", "data/campus6_screened"))
    return source.expanduser().resolve(), output.expanduser().resolve()


def discover(args, config: Dict[str, Any], source: Path) -> List[Dict[str, Any]]:
    if args.manifest:
        if not source.is_file():
            raise FileNotFoundError(f"manifest does not exist: {source}")
        candidates = load_manifest(source)
    else:
        if not source.is_dir():
            source.mkdir(parents=True, exist_ok=True)
            raise FileNotFoundError(
                f"candidate directory was created but contains no videos: {source}"
            )
        candidates = discover_directory(source, config["video_extensions"])
    return candidates


def emit(event: str, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False, sort_keys=True), flush=True)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    source, output = resolve_paths(args, config)
    output.mkdir(parents=True, exist_ok=True)
    candidates = discover(args, config, source)
    candidates = balanced_candidates(candidates)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        candidates = candidates[: args.limit]
    write_jsonl(output / "manifests" / "candidates.jsonl", candidates)

    emit(
        "campus6_screening_start",
        config=str(config_path),
        source=str(source),
        output=str(output),
        candidates=len(candidates),
        target_per_class=int(config["target_per_class"]),
        prepare_only=bool(args.prepare_only),
        dry_run=bool(args.dry_run),
    )
    if not candidates:
        emit("campus6_screening_empty", source=str(source))
        return 2
    if args.dry_run:
        emit("campus6_screening_dry_run_complete", candidates=len(candidates))
        return 0

    workflow = ScreeningWorkflow(config, output)
    if not args.prepare_only:
        workflow.preflight()
        emit(
            "campus6_screening_providers",
            primary=config["providers"]["primary"]["model"],
            secondary=config["providers"]["secondary"]["model"],
            adjudicator=config["providers"]["adjudicator"]["model"],
            adjudicator_enabled=workflow.providers["adjudicator"].available,
        )

    records = []
    total = len(candidates)
    for index, candidate in enumerate(candidates, 1):
        record, resumed = workflow.process_candidate(
            candidate,
            prepare_only=args.prepare_only,
            force=args.force,
        )
        records.append(record)
        if not args.prepare_only:
            workflow.materialize(record)
        final = record.get("final", {})
        emit(
            "campus6_screening_progress",
            current=index,
            total=total,
            sample_id=candidate["sample_id"],
            status=final.get("status", "unknown"),
            label=final.get("label") or final.get("motion"),
            resumed=resumed,
        )
        if args.stop_when_quotas_met and quotas_met(records, config):
            emit("campus6_screening_quota_stop", processed=index, total=total)
            break

    summary = workflow.export(records, candidates)
    emit("campus6_screening_complete", **summary)
    return 0 if not any(
        record.get("final", {}).get("status") == "error" for record in records
    ) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            json.dumps(
                {"event": "campus6_screening_failed", "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)
