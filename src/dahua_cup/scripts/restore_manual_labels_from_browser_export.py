#!/usr/bin/env python3
"""Validate and restore manual labels exported from an open review-browser tab."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path('/workspace/data/xzz_data/AVA_Kinetics_competition_audit_v1')
FIELDS = [
    'video_id', 'candidate_group', 'manual_label', 'interaction_direction',
    'trajectory', 'contact', 'after_contact', 'aggression_evidence',
    'playful_evidence', 'scene', 'pose_quality', 'annotator_note',
    'vlm_decision', 'updated_at',
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> str:
    return f"{row.get('video_id', '')}|{row.get('candidate_group', '')}"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--audit-root', type=Path, default=ROOT)
    result.add_argument('--recovery-json', type=Path, required=True)
    result.add_argument('--restore', action='store_true')
    return result


def main() -> int:
    args = parser().parse_args()
    root = args.audit_root.resolve()
    manifest_path = root / 'candidate_manifests' / 'probe_extraction_manifest_v3.csv'
    labels_path = root / 'candidate_manifests' / 'manual_labels_v3.csv'
    if not manifest_path.is_file() or not labels_path.is_file() or not args.recovery_json.is_file():
        raise FileNotFoundError('manifest, current labels, or recovery JSON is missing')

    exported = json.loads(args.recovery_json.read_text(encoding='utf-8'))
    if not isinstance(exported, dict):
        raise ValueError('recovery JSON must be an object keyed by video_id|candidate_group')
    recovered: dict[str, dict[str, str]] = {}
    for exported_key, value in exported.items():
        if not isinstance(value, dict):
            raise ValueError(f'non-object recovered row: {exported_key}')
        row = {field: str(value.get(field, '') or '') for field in FIELDS}
        row_key = key(row)
        if not row['video_id'] or not row['candidate_group'] or row_key != exported_key:
            raise ValueError(f'invalid recovered key: {exported_key}')
        recovered[row_key] = row

    manifest_keys = {key(row) for row in read_csv(manifest_path)}
    missing = sorted(set(recovered) - manifest_keys)
    if missing:
        raise ValueError(f'{len(missing)} recovered labels do not exist in the manifest; first: {missing[0]}')
    current = {key(row): {field: row.get(field, '') for field in FIELDS} for row in read_csv(labels_path)}
    current_only = {row_key: row for row_key, row in current.items() if row_key not in recovered}
    merged = {**recovered, **current_only}
    distribution = Counter((row.get('manual_label') or 'unlabeled') for row in merged.values())
    report = {
        'recovered_entries': len(recovered),
        'current_entries': len(current),
        'current_entries_preserved_outside_export': len(current_only),
        'merged_entries': len(merged),
        'completed_reviews': sum(count for label, count in distribution.items() if label != 'unlabeled'),
        'label_distribution': dict(sorted(distribution.items())),
        'restore': args.restore,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.restore:
        return 0

    recovery_dir = root / 'candidate_manifests' / 'recovery' / datetime.now().strftime('%Y%m%d_%H%M%S')
    recovery_dir.mkdir(parents=True, exist_ok=True)
    backup = recovery_dir / 'manual_labels_v3.before_browser_recovery.csv'
    backup.write_bytes(labels_path.read_bytes())
    write_csv(labels_path, [merged[row_key] for row_key in sorted(merged)])
    print(json.dumps({'status': 'restored', 'backup': str(backup)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
