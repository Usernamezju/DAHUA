#!/usr/bin/env python3
"""Rebuild the AVA review page from its existing manifest without touching labels."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path


ROOT = Path('/workspace/data/xzz_data/AVA_Kinetics_competition_audit_v1')


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    manifest = ROOT / 'candidate_manifests' / 'probe_extraction_manifest_v3.csv'
    pipeline = ROOT / 'scripts' / 'probe_v3_completion_pipeline.py'
    rows = read_csv(manifest)
    vlm_path = ROOT / 'candidate_manifests' / 'vlm_soft_review_v3.csv'
    vlm_rows = read_csv(vlm_path) if vlm_path.exists() else []
    vlm_by_id = {row.get('video_id', ''): row for row in vlm_rows if row.get('video_id')}
    prefix = 'AI coarse suggestion: '
    for row in rows:
        video_id = row.get('video_id', '')
        suggestion = row.get('ava_actions', '')
        if video_id and not vlm_by_id.get(video_id, {}).get('vlm_proposed_label') and suggestion.startswith(prefix):
            vlm_by_id[video_id] = {
                'video_id': video_id,
                'vlm_proposed_label': suggestion[len(prefix):],
                'vlm_confidence': '',
                'vlm_rationale': 'Imported AI coarse-screened result; not human GT.',
            }
    spec = importlib.util.spec_from_file_location('ava_probe_v3_refresh', pipeline)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load page builder: {pipeline}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.build_web(rows, list(vlm_by_id.values()))
    print(f'rebuilt {ROOT / "audit_gallery" / "index.html"} with {len(rows)} manifest rows')


if __name__ == '__main__':
    main()
