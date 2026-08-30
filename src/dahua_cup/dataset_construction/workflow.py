"""Resumable end-to-end Campus6 dataset screening workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .contact_sheets import create_contact_sheets_isolated
from .decision import (
    CAMPUS6_LABELS,
    ParsedReview,
    decide_reviews,
    parse_model_response,
    reviews_need_adjudication,
)
from .providers import HostedVisionClient


WORKFLOW_VERSION = "campus6.hosted_screening.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sample_id_for(path: Path) -> str:
    stat = path.stat()
    signature = f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}"
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    safe_stem = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in path.stem
    )
    return f"{safe_stem[:80]}_{digest}"


def _candidate(path: Path, **metadata) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    return {
        "sample_id": sample_id_for(path),
        "path": str(path),
        "modality": metadata.get("modality", "rgb"),
        "candidate_labels": metadata.get("candidate_labels", []),
        "source_dataset": metadata.get("source_dataset", path.parent.parent.name),
        "source_label": metadata.get("source_label", path.parent.name),
        "license_note": metadata.get("license_note", ""),
        "privacy_confirmed": metadata.get("privacy_confirmed"),
        "privacy_note": metadata.get("privacy_note", ""),
        "size_bytes": path.stat().st_size,
    }


def discover_directory(root: Path, extensions: Sequence[str]) -> List[Dict[str, Any]]:
    normalized = {extension.lower() for extension in extensions}
    paths = sorted(
        path for path in root.expanduser().resolve().rglob("*")
        if path.is_file() and path.suffix.lower() in normalized
    )
    return [_candidate(path) for path in paths]


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    path = path.expanduser().resolve()
    rows: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            source_rows = list(csv.DictReader(stream))
    else:
        with path.open("r", encoding="utf-8") as stream:
            source_rows = [json.loads(line) for line in stream if line.strip()]
    for row in source_rows:
        video = Path(str(row["path"]))
        if not video.is_absolute():
            video = path.parent / video
        if not video.is_file():
            continue
        rows.append(
            _candidate(
                video,
                modality=row.get("modality", "rgb"),
                candidate_labels=row.get("candidate_labels", []),
                source_dataset=row.get("source_dataset", ""),
                source_label=row.get("source_label", ""),
                license_note=row.get("license_note", ""),
                privacy_confirmed=row.get("privacy_confirmed"),
                privacy_note=row.get("privacy_note", ""),
            )
        )
    return rows


def load_prompt(prompt_dir: Path, stage_file: str) -> str:
    common = (prompt_dir / "common_rules_zh.txt").read_text(encoding="utf-8").strip()
    stage = (prompt_dir / stage_file).read_text(encoding="utf-8").strip()
    return f"{common}\n\n{stage}\n"


def _materialize(source: Path, destination: Path, mode: str):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            mode = "symlink"
    if mode == "symlink":
        destination.symlink_to(source)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"invalid link mode: {mode}")


def _serialize_review(parsed: ParsedReview) -> Dict[str, Any]:
    return {
        "valid": parsed.valid,
        "errors": list(parsed.errors),
        "parsed": parsed.raw,
    }


class ScreeningWorkflow:
    def __init__(self, config: Dict[str, Any], output_root: Path):
        self.config = config
        self.output_root = output_root.expanduser().resolve()
        self.package_root = Path(__file__).resolve().parent
        self.prompt_dir = self.package_root / "prompts"
        signature_value = {
            "workflow_version": WORKFLOW_VERSION,
            "config": config,
            "prompts": {
                path.name: path.read_text(encoding="utf-8")
                for path in sorted(self.prompt_dir.glob("*.txt"))
            },
        }
        self.workflow_signature = hashlib.sha256(
            json.dumps(signature_value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.providers = {
            name: HostedVisionClient(provider, config["request"])
            for name, provider in config["providers"].items()
        }

    def preflight(self):
        self.providers["primary"].require_key()
        self.providers["secondary"].require_key()

    def _review_one(
        self,
        stage: str,
        sheets: Sequence[Path],
    ) -> Dict[str, Any]:
        provider_config = self.config["providers"][stage]
        prompt = load_prompt(self.prompt_dir, provider_config["prompt"])
        response = self.providers[stage].invoke(sheets, prompt)
        parsed = parse_model_response(response["content"])
        return {"response": response, **_serialize_review(parsed)}

    @staticmethod
    def _parsed(record: Dict[str, Any]) -> ParsedReview:
        return ParsedReview(
            record.get("parsed", {}),
            bool(record.get("valid")),
            tuple(record.get("errors", [])),
        )

    def process_candidate(
        self,
        candidate: Dict[str, Any],
        prepare_only: bool = False,
        force: bool = False,
    ):
        sample_id = candidate["sample_id"]
        review_path = self.output_root / "reviews" / f"{sample_id}.json"
        if review_path.is_file() and not prepare_only and not force:
            existing = json.loads(review_path.read_text(encoding="utf-8"))
            if (
                existing.get("workflow_signature") == self.workflow_signature
                and existing.get("final", {}).get("status") != "error"
            ):
                return existing, True

        sheet_dir = self.output_root / "work" / "contact_sheets" / sample_id
        metadata_path = sheet_dir / "metadata.json"
        modality = candidate.get("modality", "rgb")
        sheet_config = self.config["contact_sheets"]
        if modality == "skeleton":
            from .skeleton_sheets import create_skeleton_contact_sheets
            sheet_creator = create_skeleton_contact_sheets
        else:
            sheet_creator = create_contact_sheets_isolated
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            sheets = [Path(path) for path in metadata["contact_sheets"]]
            if not all(path.is_file() for path in sheets):
                try:
                    sheets, metadata = sheet_creator(
                        Path(candidate["path"]), sheet_dir, sheet_config
                    )
                    atomic_json(metadata_path, metadata)
                except Exception as sheet_exc:
                    record = {
                        "workflow_version": WORKFLOW_VERSION,
                        "workflow_signature": self.workflow_signature,
                        "created_at": utc_now(),
                        "sample": candidate,
                        "reviews": {},
                        "final": {"status": "error", "reason": str(sheet_exc)},
                    }
                    atomic_json(review_path, record)
                    return record, False
        else:
            try:
                sheets, metadata = sheet_creator(
                    Path(candidate["path"]), sheet_dir, sheet_config
                )
                atomic_json(metadata_path, metadata)
            except Exception as sheet_exc:
                record = {
                    "workflow_version": WORKFLOW_VERSION,
                    "workflow_signature": self.workflow_signature,
                    "created_at": utc_now(),
                    "sample": candidate,
                    "reviews": {},
                    "final": {"status": "error", "reason": str(sheet_exc)},
                }
                atomic_json(review_path, record)
                return record, False

        if prepare_only:
            return {
                "workflow_version": WORKFLOW_VERSION,
                "workflow_signature": self.workflow_signature,
                "sample": candidate,
                "contact_sheet_metadata": metadata,
                "final": {"status": "prepared"},
            }, False

        record: Dict[str, Any] = {
            "workflow_version": WORKFLOW_VERSION,
            "workflow_signature": self.workflow_signature,
            "created_at": utc_now(),
            "sample": candidate,
            "contact_sheet_metadata": metadata,
            "reviews": {},
        }
        if not sheets:
            record["final"] = {"status": "error", "reason": "sheet creation produced no images"}
            atomic_json(review_path, record)
            return record, False
        try:
            record["reviews"]["primary"] = self._review_one("primary", sheets)
            record["reviews"]["secondary"] = self._review_one("secondary", sheets)
            independent = [
                self._parsed(record["reviews"]["primary"]),
                self._parsed(record["reviews"]["secondary"]),
            ]
            adjudicator = None
            if reviews_need_adjudication(independent):
                if self.providers["adjudicator"].available:
                    record["reviews"]["adjudicator"] = self._review_one(
                        "adjudicator", sheets
                    )
                    adjudicator = self._parsed(record["reviews"]["adjudicator"])
                else:
                    record["adjudicator_skipped"] = (
                        f"missing {self.providers['adjudicator'].api_key_env}"
                    )
            record["final"] = decide_reviews(independent, adjudicator)
        except Exception as exc:
            record["final"] = {"status": "error", "reason": str(exc)}
        atomic_json(review_path, record)
        return record, False

    def materialize(self, record: Dict[str, Any]):
        final = record["final"]
        source = Path(record["sample"]["path"])
        sample_id = record["sample"]["sample_id"]
        filename = f"{sample_id}{source.suffix.lower()}"
        status = final.get("status")
        if status == "accepted":
            modality = record["sample"].get("modality", "rgb")
            destination = (
                self.output_root / "labeled" / modality / final["label"] / filename
            )
        elif status == "partial":
            destination = self.output_root / "partial_labels" / final["motion"] / filename
        elif status == "manual_review":
            destination = self.output_root / "manual_review" / filename
        else:
            return
        _materialize(source, destination, self.config.get("link_mode", "hardlink"))

    def export(self, records: Sequence[Dict[str, Any]], candidates: Sequence[Dict[str, Any]]):
        manifests = self.output_root / "manifests"
        write_jsonl(manifests / "candidates.jsonl", candidates)
        write_jsonl(manifests / "final_dataset.jsonl", records)
        counts = Counter(
            record["final"].get("label")
            for record in records
            if record.get("final", {}).get("status") == "accepted"
        )
        statuses = Counter(record.get("final", {}).get("status", "unknown") for record in records)
        target = int(self.config.get("target_per_class", 500))
        summary = {
            "schema_version": WORKFLOW_VERSION,
            "updated_at": utc_now(),
            "candidate_count": len(candidates),
            "processed_count": len(records),
            "target_per_class": target,
            "accepted_per_class": {label: counts.get(label, 0) for label in CAMPUS6_LABELS},
            "shortfall_per_class": {
                label: max(0, target - counts.get(label, 0)) for label in CAMPUS6_LABELS
            },
            "status_counts": dict(statuses),
            "complete": all(counts.get(label, 0) >= target for label in CAMPUS6_LABELS),
        }
        atomic_json(manifests / "summary.json", summary)
        return summary
