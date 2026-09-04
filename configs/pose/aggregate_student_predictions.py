"""Summarize Campus6 predictions for the selected 358-sample subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in args.list.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    for index, row in enumerate(rows):
        path = args.predictions / "sample_{:02d}.json".format(index)
        payload = json.loads(path.read_text(encoding="utf-8"))
        top1 = (payload.get("topk") or [{}])[0]
        expected = int(row["label_id"])
        predicted = int(top1.get("class_index", -1))
        results.append({
            "index": index,
            "video": row["video"],
            "frame_dir": row.get("frame_dir", ""),
            "expected_label_id": expected,
            "expected_label": row.get("frame_dir", "").split("/", 1)[0],
            "predicted_label_id": predicted,
            "predicted_label": top1.get("label", ""),
            "score": top1.get("score"),
            "correct": predicted == expected,
        })
    correct = sum(int(item["correct"]) for item in results)
    report = {
        "videos": len(results),
        "correct": correct,
        "accuracy": float(correct / len(results)) if results else 0.0,
        "checkpoint_format": "quantized",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
