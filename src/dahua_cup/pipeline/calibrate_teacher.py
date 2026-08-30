"""Fit a deterministic temperature calibrator from human-labeled teacher outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dahua_cup.pipeline.common import log_event, read_jsonl
from dahua_cup.semantic_teacher.pseudo_label.calibration import fit_temperature


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--version", default="campus6-temperature-v1")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    samples = []
    for row in read_jsonl(args.input_jsonl):
        teacher = row.get("result") or row
        distribution = teacher["distribution"]
        if isinstance(distribution, list):
            distribution = {
                item["label"]: item.get("probability", item.get("score", 0.0))
                for item in distribution
            }
        samples.append((distribution, str(row["true_label"])))
    calibrator = fit_temperature(samples, version=args.version)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(calibrator.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log_event(
        "teacher_calibration_complete",
        samples=len(samples),
        temperature=calibrator.temperature,
        output=str(destination),
    )


if __name__ == "__main__":
    main()
