#!/usr/bin/env python3
"""Strictly re-audit previously accepted Campus6 clips without overwriting them.

The first pass is a high-recall seven-way classifier.  This utility is the
high-precision second pass: each request has one proposed class and can only
return that class or ``irrelevant``.  Accepted clips are copied into a separate
``curated_screened`` tree, so model training can use a frozen audit manifest.
"""

from __future__ import annotations

import argparse
import json
import shutil
import threading
from collections import Counter
from pathlib import Path

from . import config
from .classify import RateLimitExceeded, RateLimiter, classify_clip


LABELS = tuple(config.CAMPUS6_CLASSES[:-1])
AUDIT_VERSION = "binary_evidence_v1"


def audit_prompt(label: str) -> str:
    descriptions = {
        "normal_walk": "一名或多名完整可见的真人持续正常行走；没有持续奔跑、追逐、推搡或冲突。",
        "normal_run": "完整可见的真人持续自主奔跑；不是被另一人追赶，也不是追逐游戏。",
        "playful_chase": "恰好两名完整可见真人以玩耍方式追跑；有轮流、等待或游戏性互动，且没有恐惧逃离。",
        "playful_push": "恰好两名完整可见真人在玩耍中轻推或打闹；接触后两人稳定且继续友好互动。",
        "conflict_chase": "恰好两名完整可见真人发生单向冲突追赶；被追者明显在逃离/躲避，追者持续逼近。",
        "conflict_push": "恰好两名完整可见真人有争执性单向推搡；被推者明显后退、失衡或防御。",
    }
    return f"""你是严格的数据集质检员。以下 8 帧来自同一段短视频，候选标签是 `{label}`。

只有在画面直接、充分证明以下定义时才能验收：{descriptions[label]}

硬性拒绝规则：
- 必须是现实拍摄的真人；动画、游戏、动物、车辆、玩具、新闻演播室/静态截图一律拒绝。
- 关键动作帧至少一名主体头到脚或近乎全身清晰可见；四个交互类必须恰好两名主要真人，且两人均近乎全身可见。人物过小、严重裁切、只有局部、镜头外互动、遮挡严重均拒绝。
- 不可根据文件名、字幕或新闻文字推断；必须看见该动作本身。普通同向跑步不是追逐；普通接触不是推搡。
- 证据不充分时必须拒绝，宁可错杀，不可误收。

只输出 JSON：
{{"campus6_label":"{label} 或 irrelevant", "confidence":"high|medium|low", "reason":"不超过50字的可见证据"}}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=config.SCREENED_DIR)
    parser.add_argument("--output", type=Path,
                        default=config.DATA_ROOT / "curated_screened")
    parser.add_argument("--manifest", type=Path,
                        default=config.DATA_ROOT / "curated_audit_manifest.jsonl")
    parser.add_argument("--labels", nargs="+", choices=LABELS, default=list(LABELS))
    parser.add_argument("--limit", type=int, default=0,
                        help="Audit at most this many pending clips (0 means all).")
    parser.add_argument("--workers", type=int, default=3,
                        help="Concurrent API requests sharing one RPM limiter.")
    return parser.parse_args()


def clip_key(source: Path, root: Path) -> str:
    return source.resolve().relative_to(root.resolve()).as_posix()


def load_existing(manifest: Path) -> dict[str, dict]:
    completed: dict[str, dict] = {}
    if not manifest.is_file():
        return completed
    for line in manifest.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if row.get("audit_version") == AUDIT_VERSION and "clip" in row:
                completed[row["clip"]] = row
        except (TypeError, ValueError):
            pass
    return completed


def write_manifest(manifest: Path, rows: dict[str, dict]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for key in sorted(rows):
            handle.write(json.dumps(rows[key], ensure_ascii=False) + "\n")
    temporary.replace(manifest)


def main() -> None:
    args = parse_args()
    api_keys, rpm, model = {
        "dashscope": (config.DASHSCOPE_API_KEYS, config.DASHSCOPE_RPM, config.DASHSCOPE_MODEL),
        "modelscope": (config.MODELSCOPE_API_KEYS, config.MODELSCOPE_RPM, config.MODELSCOPE_MODEL),
        "gemini": (config.GEMINI_API_KEYS, config.GEMINI_RPM, config.GEMINI_MODEL),
    }[config.CLASSIFIER_BACKEND]
    if not api_keys:
        raise RuntimeError(f"No API key for {config.CLASSIFIER_BACKEND}")
    if len(api_keys) != 1:
        raise RuntimeError("Pass exactly one API key; --workers safely shares its RPM budget.")

    completed = load_existing(args.manifest)
    tasks = []
    for label in args.labels:
        source_dir = args.source / label
        tasks.extend((label, path) for path in sorted(source_dir.rglob("*.mp4")))
    pending = [(label, path) for label, path in tasks
               if clip_key(path, args.source) not in completed]
    if args.limit:
        pending = pending[:args.limit]
    print(f"Audit backend: {config.CLASSIFIER_BACKEND} ({model}); completed={len(completed)}, pending={len(pending)}")

    lock = threading.Lock()
    task_lock = threading.Lock()
    limiter = RateLimiter(rpm)
    next_task = 0
    quota_reached = threading.Event()

    def worker(worker_id: int) -> None:
        nonlocal next_task
        while not quota_reached.is_set():
            with task_lock:
                if next_task >= len(pending):
                    return
                index = next_task + 1
                expected, clip = pending[next_task]
                next_task += 1
            key = clip_key(clip, args.source)
            try:
                limiter.wait()
                result = classify_clip(clip, api_keys[0], prompt=audit_prompt(expected))
            except RateLimitExceeded as error:
                print(f"Quota reached after scheduling {index} clips: {error}")
                quota_reached.set()
                return
            # A transport/API failure is not a review decision.  Leave the
            # clip pending so a later invocation can retry it.
            if result is None:
                print(f"{index}/{len(pending)} API failure; leaving pending: {key}")
                continue
            accepted = result["campus6_label"] == expected and result["confidence"] in {"high", "medium"}
            row = {
                "clip": key, "expected_label": expected, "accepted": accepted,
                "audit_version": AUDIT_VERSION, **result,
            }
            with lock:
                completed[key] = row
                if accepted:
                    destination = args.output / expected / key
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(clip, destination)
                write_manifest(args.manifest, completed)
                done = len(completed)
                if done % 10 == 0 or index == len(pending):
                    counts = Counter(item["expected_label"] for item in completed.values() if item.get("accepted"))
                    print(f"{done} completed; accepted=" + ", ".join(f"{k}:{counts[k]}" for k in LABELS))

    threads = [threading.Thread(target=worker, args=(worker_id,), daemon=False)
               for worker_id in range(1, max(1, args.workers) + 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
