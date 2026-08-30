"""Lightweight person localization before pose extraction."""

from .grouping import (
    PersonDetection,
    build_group_crop,
    select_people,
)

__all__ = [
    "PersonDetection",
    "build_group_crop",
    "select_people",
]
