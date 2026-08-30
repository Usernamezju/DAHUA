#!/usr/bin/env python3
"""Evaluate Campus6 skeletons with GAP's frozen CLIP-aligned action encoder.

This is a strict text-prototype zero-shot evaluation: the six semantic prompts
are fixed before inference, and Campus6 labels are read only to report metrics.
No Campus6 sample is used to train, calibrate, select, or adapt the model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


LABELS = (
    "normal_walk",
    "normal_run",
    "playful_chase",
    "playful_push",
    "conflict_chase",
    "conflict_push",
)
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def read_manifest(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def feature_path(features: Path, video: Path, label: str) -> Path:
    digest = hashlib.sha1(str(video.resolve()).encode("utf-8")).hexdigest()[:12]
    return features / label / (video.stem + "__" + digest + ".npz")


def prenormalize_ntu25(keypoint: np.ndarray) -> np.ndarray:
    """Match the center-only NTU preprocessing used for Campus6 ProtoGCN."""
    skeleton = np.asarray(keypoint, dtype=np.float32).copy()
    if skeleton.ndim != 4 or skeleton.shape[0] not in (1, 2) or skeleton.shape[2:] != (25, 3):
        raise ValueError("expected (M,T,25,3), got %s" % (skeleton.shape,))
    valid0 = np.flatnonzero(~np.all(np.isclose(skeleton[0], 0.0), axis=(1, 2)))
    if skeleton.shape[0] == 2:
        valid1 = np.flatnonzero(~np.all(np.isclose(skeleton[1], 0.0), axis=(1, 2)))
        if len(valid1) > len(valid0):
            skeleton = skeleton[[1, 0]]
            valid0 = valid1
    if not len(valid0):
        raise ValueError("no pose slot has a valid frame")
    skeleton = skeleton[:, valid0]
    center = skeleton[0, 0, 1].copy()
    mask = (np.abs(skeleton).sum(axis=-1) > 0)[..., None]
    return (skeleton - center) * mask


def crop_resize(data: np.ndarray, window: int = 64, p: float = 0.95) -> np.ndarray:
    """Deterministic implementation of GAP feeder's valid_crop_resize."""
    # data is M,T,V,C; GAP feeder expects C,T,V,M.
    data = data.transpose(3, 1, 2, 0)
    _, total_frames, _, _ = data.shape
    bias = int((1.0 - p) * total_frames / 2)
    data = data[:, bias: total_frames - bias]
    length = data.shape[1]
    if length < 1:
        raise ValueError("no frames remain after crop")
    tensor = torch.from_numpy(data).float().permute(0, 2, 3, 1).contiguous()
    tensor = tensor.view(1, 1, -1, length)
    tensor = torch.nn.functional.interpolate(
        tensor, size=(tensor.shape[2], window), mode="bilinear", align_corners=False
    )
    tensor = tensor.squeeze(0).squeeze(0).view(3, 25, 2, window)
    return tensor.permute(0, 3, 1, 2).contiguous().numpy()


def load_samples(manifest: Path, features: Path):
    samples, missing, invalid = [], [], []
    for row in read_manifest(manifest):
        label = row["label"]
        if label not in LABEL_TO_ID:
            continue
        video = Path(row["video"])
        feature = feature_path(features, video, label)
        if not feature.is_file():
            missing.append(str(feature))
            continue
        try:
            with np.load(feature) as arrays:
                skeleton = prenormalize_ntu25(arrays["keypoint"])
        except (KeyError, ValueError) as error:
            invalid.append({"video": str(video), "feature": str(feature), "reason": str(error)})
            continue
        samples.append({
            "video": str(video), "feature": str(feature), "label": label,
            "label_id": LABEL_TO_ID[label], "data": crop_resize(skeleton),
        })
    if missing:
        raise RuntimeError("%d feature files missing; example: %s" % (len(missing), missing[0]))
    if not samples:
        raise RuntimeError("no valid Campus6 samples found")
    return samples, invalid


def load_gap_model(gap_root: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(gap_root))
    import clip
    from model.baseline import TextCLIP

    model_cls = importlib.import_module("model.ctrgcn").Model_lst_4part
    model = model_cls(
        num_class=120, num_point=25, num_person=2,
        graph="graph.ntu_rgb_d.Graph", k=8, head=["ViT-B/32"],
        graph_args={"labeling_mode": "spatial"},
    )
    weights = torch.load(checkpoint, map_location="cpu")
    if "model_state_dict" in weights:
        weights = weights["model_state_dict"]
    weights = {key.split("module.")[-1]: value for key, value in weights.items()}
    incompatible = model.load_state_dict(weights, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("unexpected GAP checkpoint mismatch: %s" % incompatible)
    model = model.to(device).eval()
    clip_model, _ = clip.load("ViT-B/32", device=device, download_root="/root/.cache/clip")
    del clip_model.visual
    text_model = TextCLIP(clip_model).to(device).eval()
    return model, text_model, clip


@torch.no_grad()
def text_prototypes(text_model, clip_module, prompts: dict, device: torch.device) -> torch.Tensor:
    prototypes = []
    for label in LABELS:
        variants = prompts.get(label)
        if not variants:
            raise ValueError("prompt variants missing for %s" % label)
        feature = text_model(clip_module.tokenize(variants).to(device)).float()
        feature = torch.nn.functional.normalize(feature, dim=-1)
        prototypes.append(torch.nn.functional.normalize(feature.mean(dim=0), dim=-1))
    return torch.stack(prototypes)


@torch.no_grad()
def predict(model, samples, prototypes, device, batch_size):
    all_scores = []
    for start in range(0, len(samples), batch_size):
        batch = np.stack([row["data"] for row in samples[start: start + batch_size]])
        tensor = torch.from_numpy(batch).to(device)
        _, feature_dict, _, _ = model(tensor)
        visual = torch.nn.functional.normalize(feature_dict["ViT-B/32"].float(), dim=-1)
        all_scores.append((visual @ prototypes.T).cpu().numpy())
    return np.concatenate(all_scores, axis=0)


def metrics(samples, scores):
    truth = np.asarray([row["label_id"] for row in samples], dtype=np.int64)
    prediction = scores.argmax(axis=1)
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for actual, predicted in zip(truth, prediction):
        matrix[actual, predicted] += 1
    per_class = {}
    for index, label in enumerate(LABELS):
        tp = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted_total = int(matrix[:, index].sum())
        precision = tp / predicted_total if predicted_total else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    return truth, prediction, matrix, {
        "protocol": "strict_text_prototype_zero_shot",
        "samples": int(len(samples)),
        "top1_accuracy": float((truth == prediction).mean()),
        "macro_f1": float(np.mean([value["f1"] for value in per_class.values()])),
        "mean_class_accuracy": float(np.mean([value["recall"] for value in per_class.values()])),
        "class_counts": dict(Counter(row["label"] for row in samples)),
        "per_class": per_class,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for this evaluation")
    args.output.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    samples, invalid = load_samples(args.manifest, args.features)
    device = torch.device(args.device)
    model, text_model, clip_module = load_gap_model(args.gap_root, args.checkpoint, device)
    prototypes = text_prototypes(text_model, clip_module, prompts, device)
    scores = predict(model, samples, prototypes, device, args.batch_size)
    truth, prediction, matrix, summary = metrics(samples, scores)
    summary.update({
        "checkpoint": str(args.checkpoint), "prompts": str(args.prompts),
        "manifest": str(args.manifest), "features": str(args.features),
        "device": str(device), "text_variants_per_class": {label: len(prompts[label]) for label in LABELS},
        "skipped_invalid_skeletons": len(invalid),
    })
    (args.output / "prompts_en.json").write_text(json.dumps(prompts, indent=2) + "\n", encoding="utf-8")
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output / "skipped_invalid_skeletons.jsonl").open("w", encoding="utf-8") as stream:
        for row in invalid:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual\\predicted", *LABELS])
        for label, row in zip(LABELS, matrix):
            writer.writerow([label, *row.tolist()])
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row, actual, predicted, score in zip(samples, truth, prediction, scores):
            result = {key: value for key, value in row.items() if key != "data"}
            result.update({
                "prediction": LABELS[int(predicted)], "prediction_id": int(predicted),
                "correct": bool(actual == predicted),
                "cosine_scores": {label: float(value) for label, value in zip(LABELS, score)},
            })
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
