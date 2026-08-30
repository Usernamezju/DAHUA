#!/usr/bin/env python3
"""Strict grouped few-shot prototype evaluation using frozen GAP embeddings.

The visual GAP encoder is frozen.  Only a small residual metric adapter is
episodically fitted on the training sources; it begins as the identity and is
regularised towards the original GAP geometry.  Validation labels are never
used for prompt selection, prototype construction, or adapter training.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluate_campus6_gap_zeroshot import (
    LABELS, load_gap_model, load_samples, text_prototypes,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--shot", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--geometry-weight", type=float, default=0.10)
    parser.add_argument("--text-prior-count", type=float, default=3.0)
    parser.add_argument("--transfer-adapter", type=Path, action="append", default=[],
                        help="existing NTU episodic adapter to transfer without fitting Campus6")
    parser.add_argument("--skip-train-adapter", action="store_true")
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


class ResidualAdapter(nn.Module):
    def __init__(self, dimension: int, hidden: int = 256):
        super().__init__()
        self.refinement = nn.Sequential(
            nn.Linear(dimension, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, dimension)
        )
        nn.init.zeros_(self.refinement[-1].weight)
        nn.init.zeros_(self.refinement[-1].bias)

    def forward(self, x):
        base = F.normalize(x, dim=-1)
        return F.normalize(base + self.refinement(base), dim=-1)


class LinearAdapter(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.projector = nn.Linear(dimension, dimension, bias=False)

    def forward(self, x):
        return F.normalize(self.projector(x), dim=-1)


def load_transfer_adapter(path: Path, dimension: int, device: torch.device) -> nn.Module:
    artifact = torch.load(path, map_location="cpu")
    state = artifact.get("metric_adapter", artifact)
    adapter = ResidualAdapter(dimension) if "refinement.0.weight" in state else LinearAdapter(dimension)
    adapter.load_state_dict(state, strict=True)
    return adapter.to(device).eval()


def frame_dir(sample):
    return "%s/%s" % (sample["label"], Path(sample["feature"]).stem)


@torch.no_grad()
def embeddings(model, samples, device, batch_size):
    result = []
    for offset in range(0, len(samples), batch_size):
        data = np.stack([row["data"] for row in samples[offset:offset + batch_size]])
        _, features, _, _ = model(torch.from_numpy(data).to(device))
        result.append(features["ViT-B/32"].float().cpu())
    return torch.cat(result)


def build_split(samples, annotation_path):
    with annotation_path.open("rb") as stream:
        annotations = pickle.load(stream)
    train_ids, test_ids = set(annotations["split"]["train"]), set(annotations["split"]["test"])
    train, test = [], []
    for index, sample in enumerate(samples):
        identifier = frame_dir(sample)
        if identifier in train_ids:
            train.append(index)
        elif identifier in test_ids:
            test.append(index)
        else:
            raise RuntimeError("sample not in grouped annotation split: %s" % identifier)
    return np.asarray(train), np.asarray(test)


def per_class_metrics(truth, prediction):
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for actual, predicted in zip(truth, prediction):
        matrix[actual, predicted] += 1
    return matrix, {
        label: {
            "support": int(matrix[i].sum()),
            "accuracy": float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else 0.0,
        }
        for i, label in enumerate(LABELS)
    }


def evaluate(name, support_embedding, support_labels, test_embedding, test_labels,
             text_prior=None, prior_count=0.0):
    prototypes = []
    for class_id in range(len(LABELS)):
        empirical = support_embedding[support_labels == class_id].mean(dim=0)
        if text_prior is not None and prior_count:
            count = int((support_labels == class_id).sum())
            empirical = (count * empirical + prior_count * text_prior[class_id]) / (count + prior_count)
        prototypes.append(F.normalize(empirical, dim=-1))
    prototypes = torch.stack(prototypes)
    scores = (test_embedding @ prototypes.T).cpu().numpy()
    prediction = scores.argmax(axis=1)
    matrix, per_class = per_class_metrics(test_labels, prediction)
    return {
        "name": name,
        "top1_accuracy": float((prediction == test_labels).mean()),
        "mean_class_accuracy": float(np.mean([item["accuracy"] for item in per_class.values()])),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "prediction": prediction.tolist(),
        "scores": scores.tolist(),
    }


def train_adapter(train_features, train_labels, adapter, args, device):
    rng = np.random.RandomState(args.seed)
    features = train_features.to(device)
    labels = np.asarray(train_labels)
    by_class = [np.flatnonzero(labels == class_id) for class_id in range(len(LABELS))]
    if min(len(indices) for indices in by_class) <= args.shot:
        raise ValueError("each class needs more support samples than --shot")
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    for step in range(args.steps):
        query_class = int(rng.randint(len(LABELS)))
        query_index = int(rng.choice(by_class[query_class]))
        supports = []
        for class_id, indices in enumerate(by_class):
            candidates = indices if class_id != query_class else indices[indices != query_index]
            supports.extend(rng.choice(candidates, size=args.shot, replace=False).tolist())
        support_index = torch.as_tensor(supports, device=device)
        support = adapter(features[support_index]).reshape(len(LABELS), args.shot, -1).mean(dim=1)
        query = adapter(features[query_index:query_index + 1])
        logits = -((query[:, None, :] - support[None, :, :]) ** 2).sum(dim=-1)
        proto_loss = F.cross_entropy(logits, torch.tensor([query_class], device=device))
        anchor = torch.cat((features[support_index], features[query_index:query_index + 1]), dim=0)
        geometry = F.mse_loss(adapter(anchor), F.normalize(anchor, dim=-1))
        loss = proto_loss + args.geometry_weight * geometry
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 500 == 0:
            print("adapter step %d/%d loss %.4f" % (step + 1, args.steps, loss.item()), flush=True)


def serialise(result):
    # Scores are written separately; metrics stay human-readable.
    return {key: value for key, value in result.items() if key not in {"scores", "prediction"}}


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    samples, invalid = load_samples(args.manifest, args.features)
    train_index, test_index = build_split(samples, args.annotations)
    model, text_model, clip_module = load_gap_model(args.gap_root, args.checkpoint, device)
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    text_prior = text_prototypes(text_model, clip_module, prompts, device)
    feature = F.normalize(embeddings(model, samples, device, args.batch_size), dim=-1)
    labels = np.asarray([sample["label_id"] for sample in samples], dtype=np.int64)
    train_feature, test_feature = feature[train_index].to(device), feature[test_index].to(device)
    train_label, test_label = labels[train_index], labels[test_index]
    direct = evaluate("frozen_gap_empirical_prototype", train_feature, train_label, test_feature, test_label)
    shrunk = evaluate("frozen_gap_text_shrunk_prototype", train_feature, train_label, test_feature, test_label,
                      text_prior=text_prior, prior_count=args.text_prior_count)
    results = [direct, shrunk]
    for path in args.transfer_adapter:
        adapter = load_transfer_adapter(path, feature.shape[1], device)
        with torch.no_grad():
            transferred_train, transferred_test = adapter(train_feature), adapter(test_feature)
        stem = path.parent.name
        results.append(evaluate("transfer_%s" % stem, transferred_train, train_label,
                                transferred_test, test_label))
        results.append(evaluate("transfer_text_shrunk_%s" % stem, transferred_train, train_label,
                                transferred_test, test_label, text_prior=text_prior,
                                prior_count=args.text_prior_count))
    if not args.skip_train_adapter:
        adapter = ResidualAdapter(feature.shape[1]).to(device)
        train_adapter(train_feature, train_label, adapter, args, device)
        adapter.eval()
        with torch.no_grad():
            adapted_train, adapted_test = adapter(train_feature), adapter(test_feature)
        results.append(evaluate("episodic_residual_protonet", adapted_train, train_label, adapted_test, test_label))
        results.append(evaluate("episodic_residual_text_shrunk_prototype", adapted_train, train_label, adapted_test,
                                test_label, text_prior=text_prior, prior_count=args.text_prior_count))
        torch.save({"adapter": adapter.state_dict(), "seed": args.seed, "steps": args.steps,
                    "shot": args.shot, "geometry_weight": args.geometry_weight}, args.output / "metric_adapter.pt")
    arrays = {result["name"] + "_scores": np.asarray(result["scores"], dtype=np.float32)
              for result in results}
    arrays.update({"labels": test_label, "test_indices": test_index, "train_indices": train_index})
    np.savez(args.output / "scores.npz", **arrays)
    report = {
        "protocol": "grouped_few_shot_support_only",
        "support_samples": int(len(train_index)), "query_samples": int(len(test_index)),
        "skipped_invalid_skeletons": len(invalid), "seed": args.seed,
        "adapter": {"steps": args.steps, "shot": args.shot, "geometry_weight": args.geometry_weight},
        "text_prior_count": args.text_prior_count,
        "results": [serialise(item) for item in results],
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
