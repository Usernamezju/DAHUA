#!/usr/bin/env python3
"""Static QDQ INT8 quantization for a skeleton-classification ONNX model.

The calibration archive must contain training-only ``data`` with shape
``N,C,T,V,M``.  Optional ``label`` values are used only for a post-quantization
sanity report; they do not influence calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def softmax(values: np.ndarray) -> np.ndarray:
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def metrics(scores: np.ndarray, labels: np.ndarray, num_classes: int) -> dict:
    prediction = scores.argmax(axis=1)
    recalls, f1s = [], []
    for label in range(num_classes):
        truth = labels == label
        predicted = prediction == label
        tp = int((truth & predicted).sum())
        recall = tp / max(int(truth.sum()), 1)
        precision = tp / max(int(predicted.sum()), 1)
        recalls.append(recall)
        f1s.append(2 * precision * recall / max(precision + recall, 1e-12))
    return {
        "top1": float((prediction == labels).mean()), "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)), "per_class_recall": recalls,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-model", type=Path, required=True)
    parser.add_argument("--calibration-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-name", default="")
    parser.add_argument("--max-calibration-samples", type=int, default=256)
    parser.add_argument("--verify-npz", type=Path)
    parser.add_argument("--verify-samples", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--per-channel", action="store_true")
    args = parser.parse_args()
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    except ImportError as exc:
        raise SystemExit("Install onnxruntime and onnxruntime-tools in the quantization environment.") from exc

    archive = np.load(args.calibration_npz, allow_pickle=False)
    data = np.asarray(archive["data"], dtype=np.float32)[:args.max_calibration_samples]
    if data.ndim != 5 or not len(data):
        raise SystemExit("calibration data must be nonempty N,C,T,V,M float32")
    fp32_session = ort.InferenceSession(str(args.fp32_model), providers=["CPUExecutionProvider"])
    input_name = args.input_name or fp32_session.get_inputs()[0].name
    output_name = fp32_session.get_outputs()[0].name

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.index = 0

        def get_next(self):
            if self.index >= len(data):
                return None
            value = data[self.index:self.index + 1]
            self.index += 1
            return {input_name: np.ascontiguousarray(value)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(args.fp32_model), str(args.output), Reader(), quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        per_channel=args.per_channel,
    )
    int8_session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    report = {
        "fp32_model": str(args.fp32_model), "int8_model": str(args.output),
        "calibration_samples": int(len(data)), "per_channel": args.per_channel,
        "fp32_bytes": args.fp32_model.stat().st_size, "int8_bytes": args.output.stat().st_size,
    }
    if args.verify_npz:
        verify = np.load(args.verify_npz, allow_pickle=False)
        verify_data = np.asarray(verify["data"], dtype=np.float32)[:args.verify_samples]
        fp32 = np.concatenate([fp32_session.run([output_name], {input_name: item[None]})[0] for item in verify_data])
        int8 = np.concatenate([int8_session.run(None, {input_name: item[None]})[0] for item in verify_data])
        report.update({
            "verify_samples": int(len(verify_data)),
            "top1_agreement": float((fp32.argmax(1) == int8.argmax(1)).mean()),
            "mean_abs_logit_delta": float(np.abs(fp32 - int8).mean()),
        })
        if "label" in verify:
            labels = np.asarray(verify["label"], dtype=np.int64)[:len(verify_data)]
            report["fp32_metrics"] = metrics(fp32, labels, args.num_classes)
            report["int8_metrics"] = metrics(int8, labels, args.num_classes)
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
