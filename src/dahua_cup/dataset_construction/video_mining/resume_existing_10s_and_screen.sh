#!/usr/bin/env bash
# Resume only material already present on disk: slice previously unprocessed
# raw videos to 10-second clips, then screen the complete local clip pool.
# No network download is performed by this script.
set -euo pipefail

log_file="/mnt/f/campus6-miner/resume_existing.log"
exec >>"${log_file}" 2>&1

echo "[$(date --iso-8601=seconds)] resume existing data: 10s slice then screen"
source /home/fjp/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /home/fjp/projects/behaviour_recognition

export CAMPUS6_CLIP_DURATION=10
export CAMPUS6_CLIP_OVERLAP=2

python -u - <<'PY'
from dahua_cup.dataset_construction.video_mining import config
from dahua_cup.dataset_construction.video_mining.slice_clips import slice_video

pending = []
for label in config.KEYWORDS:
    raw_dir = config.DOWNLOADS_DIR / label
    if not raw_dir.is_dir():
        continue
    for video in sorted(raw_dir.rglob("*.mp4")):
        first_clip = config.CLIPS_DIR / label / f"{video.stem}_clip0000.mp4"
        if not first_clip.is_file() or first_clip.stat().st_size <= 500:
            pending.append((label, video))

print(f"[slice] {len(pending)} raw videos have no prior clip series", flush=True)
for number, (label, video) in enumerate(pending, 1):
    clips = slice_video(video, config.CLIPS_DIR / label)
    print(f"[slice] {number}/{len(pending)} {video.name}: {len(clips)} clips", flush=True)
PY

echo "[$(date --iso-8601=seconds)] slicing done; start screening existing pool"
exec /home/fjp/projects/behaviour_recognition/dahua_cup/dataset_construction/video_mining/run_local_dashscope.sh
