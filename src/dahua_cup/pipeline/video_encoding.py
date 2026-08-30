"""Shared browser-compatible video encoder arguments."""

from __future__ import annotations

from typing import List


def browser_video_args(
    codec: str,
    preset: str = "veryfast",
    quality: int = 24,
    bitrate: str = "2M",
) -> List[str]:
    """Return codec-aware FFmpeg arguments for an H.264 MP4."""
    result = ["-c:v", codec]
    if codec == "libopenh264":
        result.extend(("-b:v", bitrate))
    elif codec.endswith("_nvenc"):
        if preset:
            result.extend(("-preset", preset))
        result.extend(("-cq", str(quality)))
    else:
        if preset:
            result.extend(("-preset", preset))
        result.extend(("-crf", str(quality)))
    result.extend(("-pix_fmt", "yuv420p", "-movflags", "+faststart"))
    return result
