"""Backward-compatible public API for the Qwen3-VL teacher.

The implementation is intentionally lazy: importing this module never selects a
GPU, imports Transformers, or accesses model weights.
"""

import os
from pathlib import Path

from .qwen3_backend import Qwen3Config, Qwen3Teacher, extract_json_object


class QwenTeacher(Qwen3Teacher):
    """Compatibility name with a lazy, environment-driven default config."""

    def __init__(self, config=None):
        if config is None:
            model_root = Path(os.environ.get(
                "DAHUA_MODEL_ROOT", "/workspace/data/xzz_data/DAHUA/models"
            ))
            config = Qwen3Config(model_dir=str(model_root / "Qwen3-VL-32B-Instruct"))
        super().__init__(config)


extract_json = extract_json_object

__all__ = ["Qwen3Config", "Qwen3Teacher", "QwenTeacher", "extract_json"]
