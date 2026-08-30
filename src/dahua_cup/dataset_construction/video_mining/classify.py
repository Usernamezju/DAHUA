"""Resumable multimodal screening for Campus6 video clips.

ModelScope Qwen3-VL is the default backend.  A short video is represented by
uniformly sampled JPEG frames, then sent through ModelScope's OpenAI-compatible
chat endpoint.  Gemini 3.5 Flash remains available as an explicit fallback.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

from . import config


class FrameExtractionError(RuntimeError):
    """FFmpeg could not produce enough readable frames from a clip."""


class RateLimitExceeded(RuntimeError):
    """The provider rejected a request because its quota is exhausted."""


def _ffprobe_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise FrameExtractionError(result.stderr.strip() or "ffprobe failed")
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise FrameExtractionError("ffprobe returned no duration") from exc
    if duration <= 0:
        raise FrameExtractionError("clip has a non-positive duration")
    return duration


def _sample_jpeg_frames(video_path: Path) -> List[bytes]:
    """Extract evenly spaced, bounded-size JPEG frames with FFmpeg."""

    duration = _ffprobe_duration(video_path)
    count = max(1, config.VLM_FRAME_COUNT)
    timestamps = [duration * (index + 0.5) / count for index in range(count)]
    frames: List[bytes] = []
    with tempfile.TemporaryDirectory(prefix="campus6_vlm_frames_") as temporary:
        frame_dir = Path(temporary)
        for index, timestamp in enumerate(timestamps):
            destination = frame_dir / f"frame_{index:02d}.jpg"
            command = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{timestamp:.3f}", "-i", str(video_path),
                "-frames:v", "1",
                "-vf", f"scale='min({config.VLM_FRAME_MAX_EDGE},iw)':-2",
                "-q:v", "4", str(destination),
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
            if result.returncode == 0 and destination.is_file() and destination.stat().st_size > 100:
                frames.append(destination.read_bytes())
    if not frames:
        raise FrameExtractionError("FFmpeg extracted no frames")
    return frames


def _parse_response(text: str, video_name: str) -> Dict | None:
    """Extract and validate the required JSON response."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        print(f"  [no JSON] {video_name}: {text[:200]}", file=sys.stderr)
        return None
    try:
        result = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        print(f"  [JSON parse] {video_name}: {exc}", file=sys.stderr)
        return None
    label = str(result.get("campus6_label", "")).strip()
    if label not in config.CAMPUS6_CLASSES:
        print(f"  [bad label] {video_name}: {label}", file=sys.stderr)
        return None
    return {
        "campus6_label": label,
        "confidence": result.get("confidence", "low"),
        "reason": result.get("reason", ""),
    }


def _classify_with_modelscope(video_path: Path, api_key: str, prompt: str | None = None) -> Dict | None:
    try:
        from openai import OpenAI

        frames = _sample_jpeg_frames(video_path)
        content = [{"type": "text", "text": prompt or config.CLASSIFY_PROMPT}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,"
                    + base64.b64encode(frame).decode("ascii")
                },
            }
            for frame in frames
        )
        client = OpenAI(api_key=api_key, base_url=config.MODELSCOPE_BASE_URL)
        response = client.chat.completions.create(
            model=config.MODELSCOPE_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=256,
        )
        text = response.choices[0].message.content or ""
    except Exception as exc:
        message = str(exc)
        if "429" in message or "rate limit" in message.lower() or "quota" in message.lower():
            raise RateLimitExceeded(message) from exc
        print(f"  [modelscope error] {video_path.name}: {exc}", file=sys.stderr)
        return None
    return _parse_response(text, video_path.name)


def _classify_with_dashscope(
    video_path: Path,
    api_key: str,
    prompt: str | None = None,
    input_mode: str = "sampled_frames",
    video_fps: float = 2.0,
) -> Dict | None:
    """Classify sampled frames or a complete local video with DashScope."""

    if input_mode == "full_video" and video_path.stat().st_size > 7_500_000:
        # DashScope limits Base64 video input to 10 MB. Reject safely before
        # importing the client or attempting an upload.
        raise FrameExtractionError(
            "full-video Base64 input exceeds DashScope's 10 MB limit: "
            + video_path.name
        )
    try:
        from openai import OpenAI

        content = [{"type": "text", "text": prompt or config.CLASSIFY_PROMPT}]
        if input_mode == "sampled_frames":
            frames = _sample_jpeg_frames(video_path)
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,"
                        + base64.b64encode(frame).decode("ascii")
                    },
                }
                for frame in frames
            )
        elif input_mode == "full_video":
            suffix = video_path.suffix.lower().lstrip(".") or "mp4"
            encoded = base64.b64encode(video_path.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "data:video/%s;base64,%s" % (suffix, encoded)
                    },
                    "fps": video_fps,
                    "max_pixels": 655360,
                }
            )
        else:
            raise ValueError("Unsupported DashScope input mode: %s" % input_mode)
        client = OpenAI(api_key=api_key, base_url=config.DASHSCOPE_BASE_URL)
        response = client.chat.completions.create(
            model=config.DASHSCOPE_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=256,
            extra_body={"enable_thinking": False},
        )
        text = response.choices[0].message.content or ""
    except Exception as exc:
        message = str(exc)
        if "429" in message or "rate limit" in message.lower() or "quota" in message.lower():
            raise RateLimitExceeded(message) from exc
        print(f"  [dashscope error] {video_path.name}: {exc}", file=sys.stderr)
        return None
    return _parse_response(text, video_path.name)


def _classify_with_gemini(video_path: Path, api_key: str, prompt: str | None = None) -> Dict | None:
    """Legacy direct-video Gemini path, enabled with CAMPUS6_CLASSIFIER_BACKEND=gemini."""

    try:
        from google import genai

        client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
        video_part = genai.types.Part.from_bytes(
            data=video_path.read_bytes(), mime_type="video/mp4"
        )
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=[video_part, prompt or config.CLASSIFY_PROMPT],
        )
        text = response.text.strip() if response.text else ""
    except Exception as exc:
        print(f"  [gemini error] {video_path.name}: {exc}", file=sys.stderr)
        return None
    return _parse_response(text, video_path.name) if text else None


def classify_clip(
    video_path: Path,
    api_key: str,
    prompt: str | None = None,
    input_mode: str = "sampled_frames",
    video_fps: float = 2.0,
) -> Dict | None:
    if config.CLASSIFIER_BACKEND == "modelscope":
        return _classify_with_modelscope(video_path, api_key, prompt)
    if config.CLASSIFIER_BACKEND == "dashscope":
        return _classify_with_dashscope(
            video_path,
            api_key,
            prompt,
            input_mode=input_mode,
            video_fps=video_fps,
        )
    if config.CLASSIFIER_BACKEND == "gemini":
        return _classify_with_gemini(video_path, api_key, prompt)
    raise ValueError(
        f"unsupported CAMPUS6_CLASSIFIER_BACKEND: {config.CLASSIFIER_BACKEND}"
    )


class RateLimiter:
    def __init__(self, rpm: int):
        self.min_interval = 60.0 / max(1, rpm)
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_call = time.monotonic()


def _manifest_path() -> Path:
    return config.DATA_ROOT / "screening_manifest.jsonl"


def _clip_key(path: Path) -> str:
    """Stable manifest key that cannot collide across source subdirectories."""

    try:
        return path.resolve().relative_to(config.CLIPS_DIR.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_results() -> Dict[str, Dict]:
    path = _manifest_path()
    if not path.is_file():
        return {}
    results: Dict[str, Dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            clip, label = row["clip"], row["campus6_label"]
            # Existing valid labels are completed work.  Prompt refinements
            # govern new clips only unless an explicit audit is requested.
            if label in config.CAMPUS6_CLASSES:
                results[str(clip)] = {
                    "campus6_label": label,
                    "confidence": row.get("confidence", "low"),
                    "reason": row.get("reason", ""),
                    "prompt_version": row.get("prompt_version", "legacy"),
                }
        except (TypeError, ValueError, KeyError):
            continue
    return results


def _write_results(results: Dict[str, Dict]) -> None:
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for clip_name in sorted(results):
            stream.write(json.dumps({"clip": clip_name, **results[clip_name]}, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _classify_worker(
    tasks: List[Path],
    api_key: str,
    worker_id: int,
    results: Dict[str, Dict],
    results_lock: threading.Lock,
    limiter: RateLimiter,
    rate_limited: threading.Event,
) -> None:
    processed = 0
    for path in tasks:
        if rate_limited.is_set():
            break
        clip_name = _clip_key(path)
        with results_lock:
            if clip_name in results:
                continue
        limiter.wait()
        try:
            result = classify_clip(path, api_key)
        except RateLimitExceeded as exc:
            print(f"    [worker{worker_id}] provider quota reached: {exc}", file=sys.stderr)
            rate_limited.set()
            break
        if result is None:
            result = {
                "campus6_label": "error",
                "confidence": "low",
                "reason": "classifier call failed",
            }
        else:
            # A prompt revision may change a previous accepted class to
            # irrelevant.  Remove only this clip's old local label copies.
            for old_label in config.CAMPUS6_CLASSES:
                stale = config.SCREENED_DIR / old_label / path.name
                try:
                    stale.unlink()
                except FileNotFoundError:
                    pass
            destination = config.SCREENED_DIR / result["campus6_label"] / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(path, destination)
            except OSError as exc:
                result["copy_error"] = str(exc)
        result["prompt_version"] = config.CLASSIFY_PROMPT_VERSION
        with results_lock:
            results[clip_name] = result
            _write_results(results)
        processed += 1
        if processed % 10 == 0:
            print(f"    [worker{worker_id}] {processed}/{len(tasks)} classified")
    print(f"    [worker{worker_id}] done: {processed} clips classified")


def _backend_settings() -> tuple[List[str], int, str]:
    if config.CLASSIFIER_BACKEND == "modelscope":
        return config.MODELSCOPE_API_KEYS, config.MODELSCOPE_RPM, config.MODELSCOPE_MODEL
    if config.CLASSIFIER_BACKEND == "dashscope":
        return config.DASHSCOPE_API_KEYS, config.DASHSCOPE_RPM, config.DASHSCOPE_MODEL
    if config.CLASSIFIER_BACKEND == "gemini":
        return config.GEMINI_API_KEYS, config.GEMINI_RPM, config.GEMINI_MODEL
    raise ValueError(f"unsupported CAMPUS6_CLASSIFIER_BACKEND: {config.CLASSIFIER_BACKEND}")


def run_screening_phase(all_clips: dict[str, list[Path]]) -> dict[str, int]:
    all_paths = [path for paths in all_clips.values() for path in paths]
    api_keys, rpm, model = _backend_settings()
    print(f"\nTotal clips to screen: {len(all_paths)}")
    print(f"Backend: {config.CLASSIFIER_BACKEND} ({model}); keys: {len(api_keys)}")
    if not all_paths:
        return {}
    if not api_keys:
        variable = {
            "modelscope": "MODELSCOPE_API_KEY",
            "dashscope": "DASHSCOPE_API_KEY",
            "gemini": "GEMINI_API_KEYS",
        }.get(config.CLASSIFIER_BACKEND, "API key")
        raise RuntimeError(f"{variable} is not set; cannot start {config.CLASSIFIER_BACKEND} screening.")

    results = _load_results()
    pending = [path for path in all_paths if _clip_key(path) not in results]
    print(f"Resuming: {len(results)} completed, {len(pending)} pending")
    if not pending:
        return dict(Counter(result["campus6_label"] for result in results.values()))

    tasks_per_worker: list[list[Path]] = [[] for _ in api_keys]
    for index, path in enumerate(pending):
        tasks_per_worker[index % len(api_keys)].append(path)
    results_lock = threading.Lock()
    rate_limited = threading.Event()
    threads = []
    for index, api_key in enumerate(api_keys):
        thread = threading.Thread(
            target=_classify_worker,
            args=(
                tasks_per_worker[index], api_key, index + 1, results, results_lock,
                RateLimiter(rpm), rate_limited,
            ),
            daemon=True,
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    if rate_limited.is_set():
        raise RateLimitExceeded(f"{config.CLASSIFIER_BACKEND} quota reached")

    counts = Counter(result["campus6_label"] for result in results.values())
    print(f"\nScreening results ({len(results)} clips):")
    for label in config.CAMPUS6_CLASSES:
        print(f"  {label:20s}: {counts.get(label, 0)}")
    return dict(counts)
