"""Generate competition metrics and a Markdown test report from JSONL."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dahua_cup.evaluation.metrics import evaluate_records, render_markdown_report
from dahua_cup.pipeline.common import file_hash, read_jsonl
from dahua_cup.semantic_teacher.schemas import LABELS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output")
    labels = parser.add_mutually_exclusive_group()
    labels.add_argument(
        "--labels",
        default=",".join(LABELS),
        help="comma-separated closed-set labels; defaults to Campus6",
    )
    labels.add_argument(
        "--label-map", help="one closed-set label per line"
    )
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument(
        "--slice-field",
        action="append",
        dest="slice_fields",
        help="repeat for robustness fields; defaults to occlusion/viewpoint/lighting",
    )
    parser.add_argument(
        "--edge-model",
        action="append",
        default=[],
        help="repeat for every model file in the deployed edge bundle",
    )
    parser.add_argument("--model-size-limit-mib", type=float, default=50.0)
    return parser


def _labels(args) -> tuple[str, ...]:
    if args.label_map:
        values = tuple(
            line.strip()
            for line in Path(args.label_map).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    else:
        values = tuple(
            value.strip() for value in args.labels.split(",") if value.strip()
        )
    if not values or len(set(values)) != len(values):
        raise ValueError("evaluation labels must be non-empty and unique")
    return values


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.model_size_limit_mib <= 0:
        raise ValueError("model-size-limit-mib must be positive")
    input_path = Path(args.input_jsonl).expanduser().resolve()
    report = evaluate_records(
        read_jsonl(input_path),
        _labels(args),
        ece_bins=args.ece_bins,
        slice_fields=args.slice_fields
        or ("occlusion", "viewpoint", "lighting"),
        edge_models=args.edge_model,
        maximum_edge_bytes=int(args.model_size_limit_mib * 1024 * 1024),
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["provenance"] = {
        "input_jsonl": str(input_path),
        "input_sha256": file_hash(input_path),
    }
    output = Path(args.output).expanduser().resolve()
    _write_atomic(
        output,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = (
        Path(args.markdown_output).expanduser().resolve()
        if args.markdown_output
        else output.with_suffix(".md")
    )
    _write_atomic(markdown, render_markdown_report(report))
    print(
        json.dumps(
            {
                "samples": report["classification"]["sample_count"],
                "accuracy": report["classification"]["accuracy"],
                "macro_f1": report["classification"]["macro_f1"],
                "ece": report["calibration"].get("ece"),
                "edge_bundle_passed": report["edge_model_bundle"]["passed"],
                "output": str(output),
                "markdown_output": str(markdown),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
