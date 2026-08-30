#!/usr/bin/env python3
"""Import UT-Interaction and CMU Tagging review clips into the AVA candidate pool.

This appends only unlabelled candidate rows.  It never edits the manual-label
CSV, and it rebuilds the static review page only after every media file has
been validated as browser-playable H.264 MP4.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


AUDIT_ROOT = Path('/workspace/data/xzz_data/AVA_Kinetics_competition_audit_v1')
SOURCE_MANIFEST = Path('/workspace/data/xzz_data/DAHUA/datasets/public_sources_v1/manifests/multisource_candidate_pool_v1.csv')
FFMPEG = Path('/workspace/code/envs/info_gcn/bin/ffmpeg')
FFPROBE = Path('/workspace/code/envs/info_gcn/bin/ffprobe')
BATCH = 'multisource_v1'
ALLOWED_DATASETS = {'UT_Interaction', 'CMU_Tagging_Chasing'}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def safe(value: str) -> str:
    return ''.join(char if char.isalnum() or char in '._-' else '_' for char in value).strip('._') or 'sample'


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def probe(path: Path) -> dict[str, str]:
    result = run([
        str(FFPROBE), '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_frames,duration',
        '-show_entries', 'format=duration,size', '-of', 'json', str(path),
    ])
    if result.returncode:
        raise RuntimeError(f'ffprobe failed for {path}: {result.stderr[-800:]}')
    data = json.loads(result.stdout)
    stream = (data.get('streams') or [{}])[0]
    fmt = data.get('format') or {}
    duration = float(stream.get('duration') or fmt.get('duration') or 0)
    width, height = int(stream.get('width') or 0), int(stream.get('height') or 0)
    if duration <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f'no usable video stream: {path}')
    try:
        numerator, denominator = str(stream.get('r_frame_rate', '0/1')).split('/', 1)
        fps = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    quality = 'good'
    if duration < 2 or fps < 5 or width < 160 or height < 120:
        quality = 'poor'
    elif duration < 5 or width < 320 or height < 240:
        quality = 'usable'
    return {
        'codec': str(stream.get('codec_name') or ''), 'duration': f'{duration:.3f}',
        'fps': f'{fps:.3f}', 'width': str(width), 'height': str(height),
        'frame_count': str(stream.get('nb_frames') or ''),
        'file_size': str(fmt.get('size') or path.stat().st_size),
        'technical_quality': quality,
    }


def validate_decodable(path: Path) -> None:
    result = run([str(FFMPEG), '-nostdin', '-v', 'error', '-i', str(path), '-f', 'null', '-'])
    if result.returncode:
        raise RuntimeError(f'ffmpeg validation failed for {path}: {result.stderr[-800:]}')


def source_records(manifest: Path) -> list[dict[str, str]]:
    rows, _ = read_csv(manifest)
    records = [row for row in rows if row.get('source_dataset') in ALLOWED_DATASETS]
    if not records:
        raise ValueError('no UT-Interaction or CMU Tagging rows found in source manifest')
    ids = [row.get('candidate_id', '') for row in records]
    if len(ids) != len(set(ids)) or any(not item for item in ids):
        raise ValueError('source candidate IDs are missing or duplicated')
    missing = [row.get('review_clip_path', '') for row in records if not Path(row.get('review_clip_path', '')).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} review clips are missing; first: {missing[0]}')
    return records


def destination(audit_root: Path, record: dict[str, str]) -> Path:
    return audit_root / 'videos_probe' / BATCH / safe(record['source_dataset']) / f"{safe(record['candidate_id'])}.mp4"


def materialize(source: Path, target: Path) -> str:
    if target.exists() and target.stat().st_size:
        measured = probe(target)
        if measured['codec'] == 'h264':
            return 'existing'
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    source_info = probe(source)
    if source_info['codec'] == 'h264' and source.suffix.lower() == '.mp4':
        try:
            os.link(source, target)
            return 'hardlink'
        except OSError:
            shutil.copy2(source, target)
            return 'copy'
    temporary = target.with_name(f'{target.stem}.partial.mp4')
    temporary.unlink(missing_ok=True)
    result = run([
        str(FFMPEG), '-nostdin', '-y', '-hide_banner', '-loglevel', 'error', '-i', str(source),
        '-map', '0:v:0', '-map', '0:a?', '-c:v', 'libopenh264', '-b:v', '2M',
        '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', str(temporary),
    ])
    if result.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f'ffmpeg failed for {source}: {result.stderr[-800:]}')
    try:
        measured = probe(temporary)
        if measured['codec'] != 'h264':
            raise RuntimeError(f"unexpected output codec for {source}: {measured['codec']}")
        validate_decodable(temporary)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return 'transcoded'


def build_vlm_rows(rows: list[dict[str, str]], audit_root: Path) -> list[dict[str, str]]:
    vlm_path = audit_root / 'candidate_manifests' / 'vlm_soft_review_v3.csv'
    base, _ = read_csv(vlm_path) if vlm_path.exists() else ([], [])
    by_id = {row.get('video_id', ''): row for row in base if row.get('video_id')}
    prefix = 'AI coarse suggestion: '
    for row in rows:
        suggestion = row.get('ava_actions', '')
        video_id = row.get('video_id', '')
        if video_id and not by_id.get(video_id, {}).get('vlm_proposed_label') and suggestion.startswith(prefix):
            by_id[video_id] = {
                'video_id': video_id,
                'vlm_proposed_label': suggestion[len(prefix):], 'vlm_confidence': '',
                'vlm_rationale': 'Imported AI coarse-screened result; not human GT.',
            }
    return list(by_id.values())


def load_builder(path: Path):
    spec = importlib.util.spec_from_file_location('ava_probe_v3_multisource', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load page builder: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--audit-root', type=Path, default=AUDIT_ROOT)
    result.add_argument('--source-manifest', type=Path, default=SOURCE_MANIFEST)
    result.add_argument('--dry-run', action='store_true')
    return result


def main() -> int:
    args = parser().parse_args()
    audit_root = args.audit_root.resolve()
    manifest = audit_root / 'candidate_manifests' / 'probe_extraction_manifest_v3.csv'
    page = audit_root / 'audit_gallery' / 'index.html'
    builder_path = audit_root / 'scripts' / 'probe_v3_completion_pipeline.py'
    if not all(path.is_file() for path in (manifest, page, builder_path, FFMPEG, FFPROBE)):
        raise FileNotFoundError('required audit or FFmpeg files are missing')
    records = source_records(args.source_manifest.resolve())
    summary = {
        'batch': BATCH, 'records': len(records),
        'by_dataset': dict(sorted(Counter(row['source_dataset'] for row in records).items())),
        'by_source_label': dict(sorted(Counter(row.get('source_label', '') for row in records).items())),
        'dry_run': args.dry_run,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    existing, fields = read_csv(manifest)
    new_ids = {f"{BATCH}_{safe(row['source_dataset'])}_{safe(row['candidate_id'])}" for row in records}
    collisions = new_ids & {row.get('video_id', '') for row in existing}
    if collisions:
        raise ValueError(f'{len(collisions)} candidate IDs already exist; first: {sorted(collisions)[0]}')
    if args.dry_run:
        codec_counts = Counter(probe(Path(row['review_clip_path']))['codec'] for row in records)
        print(json.dumps({'source_video_codecs': dict(sorted(codec_counts.items()))}, ensure_ascii=False))
        return 0

    batch_rows: list[dict[str, str]] = []
    actions: Counter[str] = Counter()
    for index, record in enumerate(records, 1):
        source = Path(record['review_clip_path']).resolve()
        target = destination(audit_root, record)
        action = materialize(source, target)
        actions[action] += 1
        measured = probe(target)
        if measured['codec'] != 'h264':
            raise RuntimeError(f'non-H.264 media remained after import: {target}')
        validate_decodable(target)
        video_id = f"{BATCH}_{safe(record['source_dataset'])}_{safe(record['candidate_id'])}"
        family = safe(record.get('suggested_target_family') or 'unclassified')
        batch_rows.append({
            'video_id': video_id, 'source': record['source_dataset'],
            'candidate_group': f'{BATCH}_{safe(record["source_dataset"])}_{family}',
            'kinetics_label': record.get('source_label', ''),
            'ava_actions': record.get('source_original_label', ''),
            'mapping_status': 'PUBLIC_MULTISOURCE_CANDIDATE', 'shard': '',
            'extract_status': 'imported', 'decode_status': 'ok', 'local_path': str(target),
            'duration': measured['duration'], 'fps': measured['fps'], 'width': measured['width'],
            'height': measured['height'], 'frame_count': measured['frame_count'],
            'file_size': measured['file_size'], 'technical_quality': measured['technical_quality'],
            'contact_sheet_path': '', 'extract_error': '',
            'candidate_reason': record.get('candidate_reason', ''), 'same_scope': 'public_multisource_v1',
            'timestamp': record.get('start_time', ''), 'bbox_person': '', 'source_split': '',
        })
        if index % 25 == 0 or index == len(records):
            print(f'prepared {index}/{len(records)}', flush=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    manifests = audit_root / 'candidate_manifests'
    backup_dir = manifests / 'backups'
    backup_dir.mkdir(parents=True, exist_ok=True)
    manifest_backup = backup_dir / f'probe_extraction_manifest_v3.before_{BATCH}_{timestamp}.csv'
    page_backup = backup_dir / f'audit_gallery_index.before_{BATCH}_{timestamp}.html'
    shutil.copy2(manifest, manifest_backup)
    shutil.copy2(page, page_backup)
    export_path = manifests / f'{BATCH}_candidates.csv'
    export_fields = list(batch_rows[0]) + ['source_candidate_id', 'source_review_clip_path', 'imported_at']
    export_rows = [
        {**row, 'source_candidate_id': source['candidate_id'], 'source_review_clip_path': source['review_clip_path'], 'imported_at': now()}
        for row, source in zip(batch_rows, records)
    ]
    try:
        write_csv(manifest, existing + batch_rows, fields)
        write_csv(export_path, export_rows, export_fields)
        load_builder(builder_path).build_web(existing + batch_rows, build_vlm_rows(existing + batch_rows, audit_root))
        expected = sum(row.get('decode_status') == 'ok' and row.get('split_status') != 'split_parent' for row in existing + batch_rows)
        actual = (audit_root / 'audit_gallery' / 'index.html').read_text(encoding='utf-8').count("<section class='sample'")
        if actual != expected:
            raise RuntimeError(f'page sample count mismatch: expected {expected}, got {actual}')
    except Exception:
        shutil.copy2(manifest_backup, manifest)
        shutil.copy2(page_backup, page)
        raise
    print(json.dumps({
        'status': 'complete', 'imported': len(batch_rows), 'page_samples': actual,
        'media_actions': dict(actions), 'export': str(export_path),
        'manifest_backup': str(manifest_backup), 'page_backup': str(page_backup),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
