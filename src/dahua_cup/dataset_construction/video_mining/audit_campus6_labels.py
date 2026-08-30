#!/usr/bin/env python3
"""Blindly audit Campus6 video labels with the configured multimodal API.

This command never deletes videos or rewrites annotation pickles.  It records a
conservative accept/reject/manual-review decision for every completed API call,
allowing any later dataset rebuild to use only an explicitly reviewed manifest.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from . import config
from .classify import RateLimitExceeded, RateLimiter, classify_clip


LABELS = tuple(config.CAMPUS6_CLASSES[:-1])
PROMPT_VERSION = "campus6_blind_label_audit_v1"
REVIEW_PROMPT_VERSION = "campus6_blind_label_review_v1"
AUDIT_PROMPT = """你是 Campus6 人类行为视频数据集的独立质检员。请观看短视频提供的视觉信息，进行**盲态**七分类。你不知道该视频的原始标签，不能使用文件名、字幕、标题、配音、新闻文字或任何元数据推断答案，只能依据可见的人体动作和互动关系判断。

可选标签：
1. normal_walk：真人正常持续行走，无追逐、奔跑、推搡或冲突。
2. normal_run：真人自主持续奔跑，无另一人追赶、逃离或推搡。
3. playful_chase：两名真人以游戏方式追跑，存在轮流、等待、互动或轻松游戏性；无明确恐惧、躲避或攻击。
4. playful_push：两名真人轻度、互相或友好地推搡/打闹；接触后身体稳定，继续友好互动。
5. conflict_chase：一名真人持续单向追赶另一名真人；被追者有明确逃离、躲避、防御性加速或恐惧迹象。
6. conflict_push：一名真人争执性、攻击性地单向推/搡另一名真人；被推者后退、失衡、防御或冲突升级。
7. irrelevant：不满足任何上述定义，或证据不足以可靠判定。

严格判定规则：
- 只接受真实拍摄的人类行为；动画、游戏、动物、车辆、玩具、静态图片一律为 irrelevant。
- normal_walk / normal_run 的关键帧中至少一名主体须头到脚或近乎全身可见，且动作连续可辨。
- 四个双人互动类必须有恰好两名主要真人，两人都在关键帧中头到脚或近乎全身可见；人物过小、严重裁切、遮挡、多人混杂、第一视角只见局部、动作未发生或难以看清，一律为 irrelevant。
- 无法从可见画面区分 playful 与 conflict 时，一律为 irrelevant，不能猜测。

只输出一个 JSON 对象，不要 Markdown：
{"campus6_label":"normal_walk|normal_run|playful_chase|playful_push|conflict_chase|conflict_push|irrelevant", "confidence":"high|medium|low", "reason":"不超过50字，仅描述可见证据"}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        help="pose_quality.jsonl or a compatible label/video JSONL manifest.")
    parser.add_argument("--output", type=Path,
                        help="Append-only audit result JSONL.")
    parser.add_argument("--summary", type=Path,
                        help="Summary JSON regenerated from completed results.")
    parser.add_argument("--api-key-file", type=Path,
                        help="Optional local secret file. Otherwise use the configured backend environment key.")
    parser.add_argument("--source-video-root", type=Path,
                        help="Original video root when auditing copied local videos.")
    parser.add_argument("--video-root", type=Path,
                        help="Local root preserving paths relative to --source-video-root.")
    parser.add_argument("--labels", nargs="+", choices=LABELS, default=list(LABELS))
    parser.add_argument("--input-action", choices=("accept", "reject", "manual_review"),
                        help="Only use rows with this action from an earlier audit JSONL.")
    parser.add_argument("--stage", choices=("primary", "review"), default="primary",
                        help="review records independent second-pass provenance.")
    parser.add_argument("--model",
                        help="DashScope model override, e.g. qwen3-vl-plus.")
    parser.add_argument("--input-mode", choices=("sampled_frames", "full_video"),
                        default="sampled_frames",
                        help="Use fixed local frames or the complete local video as Base64.")
    parser.add_argument("--video-fps", type=float, default=2.0,
                        help="Server-side sampling rate for full_video mode (0.1 to 10).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Audit at most N pending videos; 0 means all selected videos.")
    parser.add_argument("--print-prompt", action="store_true",
                        help="Print the exact prompt and exit without reading videos or calling an API.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count selected/pending videos without calling an API.")
    return parser.parse_args()


def backend_key() -> Tuple[Optional[str], int, str]:
    settings = {
        "dashscope": (config.DASHSCOPE_API_KEYS, config.DASHSCOPE_RPM, config.DASHSCOPE_MODEL),
        "modelscope": (config.MODELSCOPE_API_KEYS, config.MODELSCOPE_RPM, config.MODELSCOPE_MODEL),
        "gemini": (config.GEMINI_API_KEYS, config.GEMINI_RPM, config.GEMINI_MODEL),
    }
    if config.CLASSIFIER_BACKEND not in settings:
        raise ValueError("Unsupported CAMPUS6_CLASSIFIER_BACKEND: %s" % config.CLASSIFIER_BACKEND)
    keys, rpm, model = settings[config.CLASSIFIER_BACKEND]
    if len(keys) > 1:
        raise RuntimeError("Use exactly one API key for a resumable, rate-limited audit.")
    return (keys[0] if keys else None), rpm, model


def read_key_file(path: Path) -> str:
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            return value.rsplit(":", 1)[-1].strip()
    raise RuntimeError("No usable API key in --api-key-file")


def iter_source(
    path: Path, labels: Iterable[str], required_action: Optional[str] = None
) -> Iterable[Dict]:
    wanted = set(labels)
    # Audit JSONL is append-only. Keep the newest row for a video so resumed
    # primary runs cannot make a second-pass review call the same clip twice.
    latest: Dict[str, Dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        label = row.get("label", row.get("expected_label"))
        video = row.get("video")
        if label in wanted and isinstance(video, str):
            latest[video] = {"label": label, "video": video, "source": row}
    for item in latest.values():
        if required_action is None or item["source"].get("action") == required_action:
            yield item


def load_completed(path: Path, prompt_version: str) -> Dict[str, Dict]:
    completed: Dict[str, Dict] = {}
    if not path.is_file():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if row.get("prompt_version") == prompt_version and row.get("video"):
                completed[str(row["video"])] = row
        except (TypeError, ValueError):
            continue
    return completed


def resolve_video(source: Path, source_root: Optional[Path], video_root: Optional[Path]) -> Path:
    if source_root is None and video_root is None:
        return source
    if source_root is None or video_root is None:
        raise ValueError("Use --source-video-root and --video-root together")
    try:
        return video_root / source.relative_to(source_root)
    except ValueError as error:
        raise ValueError("%s is outside --source-video-root %s" % (source, source_root)) from error


def decision(expected: str, result: Dict) -> str:
    """Return a conservative automation action; never authorize deletion."""
    predicted = result["campus6_label"]
    confidence = result.get("confidence", "low")
    if predicted == expected and confidence in {"high", "medium"}:
        return "accept"
    if predicted == "irrelevant" and confidence in {"high", "medium"}:
        return "reject"
    if predicted != expected and confidence == "high":
        return "reject"
    return "manual_review"


def write_summary(completed: Dict[str, Dict], path: Path, prompt_version: str) -> None:
    actions = Counter(row.get("action", "unknown") for row in completed.values())
    by_label = {
        label: dict(Counter(
            row.get("action", "unknown")
            for row in completed.values()
            if row.get("expected_label") == label
        ))
        for label in LABELS
    }
    summary = {
        "prompt_version": prompt_version,
        "completed": len(completed),
        "actions": dict(sorted(actions.items())),
        "by_expected_label": by_label,
        "safety": "This manifest never deletes source videos or modifies annotations.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.print_prompt:
        print(AUDIT_PROMPT)
        return
    if not args.input or not args.output or not args.summary:
        raise ValueError("--input, --output, and --summary are required unless --print-prompt is used")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if not 0.1 <= args.video_fps <= 10:
        raise ValueError("--video-fps must be between 0.1 and 10")
    if args.stage == "review" and args.input_action != "accept":
        raise ValueError("--stage review requires --input-action accept")
    if args.input_mode == "full_video" and config.CLASSIFIER_BACKEND != "dashscope":
        raise ValueError("--input-mode full_video is currently supported only with dashscope")
    if args.model:
        if config.CLASSIFIER_BACKEND != "dashscope":
            raise ValueError("--model is currently supported only with the dashscope backend")
        config.DASHSCOPE_MODEL = args.model
    prompt_version = REVIEW_PROMPT_VERSION if args.stage == "review" else PROMPT_VERSION
    selected = list(iter_source(args.input, args.labels, args.input_action))
    completed = load_completed(args.output, prompt_version)
    pending = [row for row in selected if row["video"] not in completed]
    if args.limit:
        pending = pending[:args.limit]
    print("Blind audit: selected=%d, completed=%d, pending=%d" % (
        len(selected), len(completed), len(pending)
    ))
    if args.dry_run:
        return
    api_key, rpm, model = backend_key()
    if args.api_key_file:
        api_key = read_key_file(args.api_key_file)
    if not api_key:
        raise RuntimeError("No API key. Set the backend key environment variable or pass --api-key-file.")
    print("Backend: %s (%s), RPM=%d" % (config.CLASSIFIER_BACKEND, model, rpm))
    limiter = RateLimiter(rpm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream:
        for index, item in enumerate(pending, 1):
            source_video = Path(item["video"])
            video = resolve_video(source_video, args.source_video_root, args.video_root)
            if not video.is_file():
                raise FileNotFoundError("Missing video: %s" % video)
            try:
                limiter.wait()
                result = classify_clip(
                    video,
                    api_key,
                    prompt=AUDIT_PROMPT,
                    input_mode=args.input_mode,
                    video_fps=args.video_fps,
                )
            except RateLimitExceeded as error:
                print("Quota reached after %d calls: %s" % (index - 1, error))
                break
            if result is None:
                print("%d/%d API failure, leaving pending: %s" % (index, len(pending), video.name))
                continue
            row = {
                "video": str(source_video),
                "local_video": str(video),
                "expected_label": item["label"],
                "prediction": result["campus6_label"],
                "confidence": result.get("confidence", "low"),
                "reason": result.get("reason", ""),
                "action": decision(item["label"], result),
                "backend": config.CLASSIFIER_BACKEND,
                "model": model,
                "input_mode": args.input_mode,
                "video_fps": args.video_fps if args.input_mode == "full_video" else None,
                "prompt_version": prompt_version,
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            completed[row["video"]] = row
            print("%d/%d %s -> %s (%s)" % (
                index, len(pending), item["label"], row["prediction"], row["action"]
            ))
    write_summary(completed, args.summary, prompt_version)


if __name__ == "__main__":
    main()
