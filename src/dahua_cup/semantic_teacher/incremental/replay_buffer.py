"""Deterministic, stratified replay manifest maintenance."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable, Mapping


def build_replay_buffer(samples: Iterable[Mapping], capacity: int, strata=("label", "scene", "camera")) -> list[dict]:
    if capacity < 1:
        raise ValueError("capacity must be positive")
    buckets = defaultdict(list)
    for sample in samples:
        item = dict(sample)
        if "sample_id" not in item:
            raise ValueError("every replay sample requires sample_id")
        key = tuple(item.get(field, "unknown") for field in strata)
        buckets[key].append(item)
    if not buckets:
        return []
    for values in buckets.values():
        values.sort(key=lambda item: hashlib.sha256(str(item["sample_id"]).encode()).hexdigest())
    selected = []
    keys = sorted(buckets, key=str)
    while len(selected) < capacity and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(selected) < capacity:
                selected.append(buckets[key].pop(0))
    return selected
