"""Incremental-learning primitives used by the closed-loop orchestrator."""

from .hard_mining import hard_sample_score
from .release_gate import ProductionPointer, ReleaseGates, evaluate_release
from .replay_buffer import build_replay_buffer

__all__ = [
    "ProductionPointer",
    "ReleaseGates",
    "build_replay_buffer",
    "evaluate_release",
    "hard_sample_score",
]
