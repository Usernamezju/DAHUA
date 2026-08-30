#!/usr/bin/env python3
"""Create an isolated, blind human-reannotation pool for Campus6 videos.

The pool uses the existing AVA/Kinetics review-page implementation, but has
its own manifests, review server and hard-linked media.  Existing human labels
are retained only in a non-web provenance CSV so they cannot bias the new
annotation pass.  The visible AI suggestion is the current ProtoGCN prediction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np


SOURCE_ROOT = Path("/workspace/data/xzz_data/AVA_Kinetics_competition_audit_v1")
POOL_ROOT = SOURCE_ROOT / "campus6_protogcn_reaudit_v1"
LABELS = (
    "normal_walk", "normal_run", "playful_chase", "playful_push",
    "conflict_chase", "conflict_push",
)
REVIEW_LABEL = {
    "normal_walk": "normal_walking",
    "normal_run": "normal_running",
    "playful_chase": "playful_chasing",
    "playful_push": "playful_pushing",
    "conflict_chase": "aggressive_chasing",
    "conflict_push": "aggressive_pushing",
}
MANIFEST_FIELDS = [
    "video_id", "source", "candidate_group", "kinetics_label", "ava_actions",
    "mapping_status", "shard", "extract_status", "decode_status", "local_path",
    "duration", "fps", "width", "height", "frame_count", "file_size",
    "technical_quality", "contact_sheet_path", "extract_error", "candidate_reason",
    "same_scope", "timestamp", "bbox_person", "source_split", "split_status",
    "parent_video_id", "segment_start", "segment_end", "split_at",
]
VLM_FIELDS = [
    "video_id", "primary_motion", "push_present", "chase_present",
    "interaction_direction", "role_switching", "escape_behavior",
    "defensive_retreat", "strong_contact", "continued_aggression", "scene_type",
    "pose_visibility", "vlm_proposed_label", "vlm_confidence", "vlm_rationale",
    "vlm_model",
]
MANUAL_FIELDS = [
    "video_id", "candidate_group", "manual_label", "interaction_direction",
    "trajectory", "contact", "after_contact", "aggression_evidence",
    "playful_evidence", "scene", "pose_quality", "annotator_note", "vlm_decision",
    "updated_at",
]


def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--pool-root", type=Path, default=POOL_ROOT)
    parser.add_argument(
        "--annotations", type=Path,
        default=SOURCE_ROOT / "rtmpose26_coco17/annotations_protogcn_2d.pkl",
    )
    parser.add_argument(
        "--predictions", type=Path,
        default=SOURCE_ROOT / "rtmpose26_training_results/campus6_rtmpose26_k400_2d_full_all_predictions.pkl",
    )
    parser.add_argument(
        "--source-manifest", type=Path,
        default=SOURCE_ROOT / "candidate_manifests/rtmpose26_campus6_manifest_v1.jsonl",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing empty/new pool")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def source_key(row: dict) -> str:
    video = Path(row["video"])
    digest = hashlib.sha1(str(video).encode("utf-8")).hexdigest()[:12]
    return f"{row['label']}/{video.stem}__{digest}"


def probe(video: Path) -> dict[str, str]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
         "-show_entries", "format=duration,size", "-of", "json", str(video)],
        text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"ffprobe failed for {video}: {result.stderr[-300:]}")
    payload = json.loads(result.stdout)
    stream, fmt = (payload.get("streams") or [{}])[0], payload.get("format") or {}
    try:
        numerator, denominator = str(stream.get("r_frame_rate", "0/1")).split("/", 1)
        fps = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    duration = float(stream.get("duration") or fmt.get("duration") or 0)
    width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
    if duration <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"invalid video stream: {video}")
    quality = "good" if duration >= 5 and fps >= 5 and width >= 320 and height >= 240 else "usable"
    return {
        "duration": f"{duration:.3f}", "fps": f"{fps:.3f}", "width": str(width),
        "height": str(height), "frame_count": str(stream.get("nb_frames") or ""),
        "file_size": str(fmt.get("size") or video.stat().st_size), "technical_quality": quality,
    }


def load_rows(args) -> tuple[list[dict], list[dict]]:
    with args.annotations.open("rb") as handle:
        annotations = pickle.load(handle)
    with args.predictions.open("rb") as handle:
        scores = np.asarray(pickle.load(handle))
    samples = annotations["annotations"]
    if scores.ndim != 2 or len(samples) != len(scores) or scores.shape[1] != len(LABELS):
        raise RuntimeError(f"annotation/prediction mismatch: {len(samples)} annotations, {scores.shape} predictions")
    train_ids = set(annotations["split"]["train"])
    val_test_ids = set(annotations["split"]["val"]) | set(annotations["split"]["test"])
    if len(train_ids | val_test_ids) != len(samples):
        raise RuntimeError("annotations contain an unsupported split")
    manifest = {
        source_key(row): row
        for row in (json.loads(line) for line in args.source_manifest.read_text(encoding="utf-8").splitlines() if line.strip())
    }
    if len(manifest) != len({source_key(row) for row in manifest.values()}):
        raise RuntimeError("source manifest has duplicate feature identities")
    prior_labels = {
        row.get("video_id", ""): row.get("manual_label", "")
        for row in read_csv(args.source_root / "candidate_manifests/manual_labels_v3.csv")
    }
    candidates, provenance = [], []
    for annotation, score in zip(samples, scores):
        source = manifest.get(annotation["frame_dir"])
        if source is None:
            raise RuntimeError(f"no source-manifest row for {annotation['frame_dir']}")
        split = "train" if annotation["frame_dir"] in train_ids else "eval_holdout"
        if split == "eval_holdout" and annotation["frame_dir"] not in val_test_ids:
            raise RuntimeError(f"unexpected split for {annotation['frame_dir']}")
        prediction = LABELS[int(np.argmax(score))]
        shifted = score - np.max(score)
        confidence = float(np.exp(shifted).max() / np.exp(shifted).sum())
        video = Path(source["video"])
        if not video.is_file():
            raise FileNotFoundError(video)
        digest = hashlib.sha1(str(video).encode("utf-8")).hexdigest()[:12]
        identifier = f"campus6_protogcn_{split}_{digest}"
        candidates.append({
            "video_id": identifier, "split": split, "video": video,
            "prediction": prediction, "review_label": REVIEW_LABEL[prediction],
            "confidence": confidence, "source_video_id": source.get("video_id", ""),
            "prior_label": prior_labels.get(source.get("video_id", ""), ""),
        })
    if len({row["video_id"] for row in candidates}) != len(candidates):
        raise RuntimeError("duplicate new candidate IDs")
    return candidates, provenance


def make_pool(args, candidates: list[dict]) -> dict:
    root = args.pool_root.resolve()
    if root.exists() and any(root.iterdir()) and not args.force:
        raise FileExistsError(f"pool already exists and is nonempty: {root}; use --force only after review")
    if args.dry_run:
        return {
            "pool_root": str(root), "samples": len(candidates),
            "groups": dict(Counter(row["split"] for row in candidates)),
            "predictions": dict(Counter(row["prediction"] for row in candidates)),
            "action": "dry_run",
        }
    root.mkdir(parents=True, exist_ok=True)
    manifests, videos, scripts = root / "candidate_manifests", root / "videos_probe", root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    # The legacy page builder writes directly to these paths instead of
    # creating them itself.
    (root / "audit_gallery").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    # The legacy static server exposes audit_gallery as its document root,
    # while the generated page addresses sibling media as ../videos_probe/.
    # Make that URL resolvable without broadening the server's exposed root.
    gallery_media = root / "audit_gallery/videos_probe"
    if not gallery_media.exists():
        gallery_media.symlink_to("../videos_probe")
    source_builder = args.source_root / "scripts/probe_v3_completion_pipeline.py"
    copied_builder = scripts / "probe_v3_completion_pipeline.py"
    builder_text = source_builder.read_text(encoding="utf-8")
    # This field is intentionally visible in this comparison pass: reviewers
    # need to inspect both the legacy manual label and the new model label.
    builder_text = builder_text.replace("<dt>Kinetics label</dt>", "<dt>原始人工标注</dt>")
    builder_text = builder_text.replace(
        "data-vlm-conf='{esc(v.get('vlm_confidence',''))}'>",
        "data-vlm-conf='{esc(v.get('vlm_confidence',''))}' "
        "data-original-label='{esc(r.get('kinetics_label',''))}'>",
    )
    builder_text = builder_text.replace(
        "let g=groupFilter.value, ai=aiLabelFilter.value, u=onlyUnlabeled.checked",
        "let g=groupFilter.value, original=originalLabelFilter.value, u=onlyUnlabeled.checked",
    ).replace(
        "(g&&s.dataset.group!==g)||(ai&&s.dataset.vlmLabel!==ai)",
        "(g&&s.dataset.group!==g)||(original&&s.dataset.originalLabel!==original)",
    )
    builder_text = builder_text.replace(
        "ai_labels=sorted({str(v.get('vlm_proposed_label','')) for v in vmap.values() if str(v.get('vlm_proposed_label',''))})",
        "ai_labels=sorted({str(r.get('kinetics_label','')) for r in rows if str(r.get('kinetics_label',''))})",
    ).replace("id='aiLabelFilter'", "id='originalLabelFilter'").replace(
        "全部 AI 标签", "全部原始人工标注"
    ).replace("只看 AI:", "只看原始人工标注:")
    copied_builder.write_text(builder_text.replace(str(args.source_root.resolve()), str(root)), encoding="utf-8")

    manifest_rows, vlm_rows, provenance = [], [], []
    for index, item in enumerate(candidates, 1):
        group = f"campus6_protogcn_v1_{item['split']}"
        destination = videos / item["split"] / f"{item['video_id']}.mp4"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            os.link(item["video"], destination)
        measured = probe(destination)
        manifest_rows.append({
            "video_id": item["video_id"], "source": "Campus6_ProtoGCN",
            "candidate_group": group,
            "kinetics_label": item["prior_label"] or "unlabeled",
            "ava_actions": f"AI prediction: {item['review_label']}",
            "mapping_status": "PROTOGCN_REANNOTATION_CANDIDATE", "shard": "",
            "extract_status": "imported", "decode_status": "ok", "local_path": str(destination),
            **measured, "contact_sheet_path": "", "extract_error": "",
            "candidate_reason": "Reannotation candidate. Review the video independently; the AI suggestion is advisory only.",
            "same_scope": "campus6_protogcn_reaudit_v1", "timestamp": "", "bbox_person": "",
            "source_split": item["split"], "split_status": "", "parent_video_id": "",
            "segment_start": "", "segment_end": "", "split_at": "",
        })
        vlm_rows.append({
            "video_id": item["video_id"], "primary_motion": "", "push_present": "", "chase_present": "",
            "interaction_direction": "", "role_switching": "", "escape_behavior": "",
            "defensive_retreat": "", "strong_contact": "", "continued_aggression": "",
            "scene_type": "", "pose_visibility": "", "vlm_proposed_label": item["review_label"],
            "vlm_confidence": f"{item['confidence']:.4f}",
            "vlm_rationale": "Prediction from current RTMPose-26 + full-finetuned ProtoGCN; not a human label.",
            "vlm_model": "campus6_rtmpose26_k400_2d_full_epoch40",
        })
        provenance.append({
            "new_video_id": item["video_id"], "source_video_id": item["source_video_id"],
            "split": item["split"], "model_prediction": item["prediction"],
            "ai_display_label": item["review_label"], "prior_human_label_hidden_from_web": item["prior_label"],
            "original_video_path": str(item["video"]),
        })
        if index % 50 == 0 or index == len(candidates):
            print(f"prepared {index}/{len(candidates)}", flush=True)
    write_csv(manifests / "probe_extraction_manifest_v3.csv", manifest_rows, MANIFEST_FIELDS)
    write_csv(manifests / "vlm_soft_review_v3.csv", vlm_rows, VLM_FIELDS)
    manual_path = manifests / "manual_labels_v3.csv"
    existing_reviews = read_csv(manual_path) if manual_path.exists() else []
    # --force rebuilds manifest/page assets, never discards reviews already
    # saved through the page.
    write_csv(manual_path, existing_reviews, MANUAL_FIELDS)
    write_csv(manifests / "hidden_provenance.csv", provenance, list(provenance[0]))
    write_csv(manifests / "review_order_v3.csv", [
        {"video_id": row["video_id"], "candidate_group": f"campus6_protogcn_v1_{row['split']}", "display_rank": index}
        for index, row in enumerate(sorted(candidates, key=lambda row: (row["split"], row["video_id"])), 1)
    ], ["video_id", "candidate_group", "display_rank"])

    spec = importlib.util.spec_from_file_location("campus6_reaudit_builder", copied_builder)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load copied review-page builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.build_web(manifest_rows, vlm_rows)
    summary = {
        "pool_root": str(root), "samples": len(candidates),
        "groups": dict(Counter(row["split"] for row in candidates)),
        "predictions": dict(Counter(row["prediction"] for row in candidates)),
        "manifest": str(manifests / "probe_extraction_manifest_v3.csv"),
        "labels": str(manifests / "manual_labels_v3.csv"),
        "web": str(root / "audit_gallery/index.html"),
        "server": str(scripts / "serve_audit_gallery_v3.py"),
    }
    (root / "STATUS.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = args_parser().parse_args()
    candidates, _ = load_rows(args)
    print(json.dumps(make_pool(args, candidates), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
