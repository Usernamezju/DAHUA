#!/usr/bin/env bash
# Download and slice one bounded, deduplicated candidate batch.
# This intentionally does not invoke the VLM classifier: the existing local
# screening process remains the sole consumer of the manifest.
set -euo pipefail

log_file="/mnt/f/campus6-miner/mining_producer.log"
exec >>"${log_file}" 2>&1

echo "[$(date --iso-8601=seconds)] candidate mining batch starts"
source /home/fjp/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /home/fjp/projects/behaviour_recognition

# A successful download limit, rather than an attempt limit.  The yt-dlp
# archive makes repeated batches advance beyond sources already collected.
export CAMPUS6_YTDLP_MAX_DOWNLOADS="${CAMPUS6_YTDLP_MAX_DOWNLOADS:-2}"
export CAMPUS6_YTDLP_MAX_DURATION="${CAMPUS6_YTDLP_MAX_DURATION:-120}"
export CAMPUS6_CLIP_DURATION="${CAMPUS6_CLIP_DURATION:-10}"
export CAMPUS6_CLIP_OVERLAP="${CAMPUS6_CLIP_OVERLAP:-2}"

python -u - <<'PY'
from dahua_cup.dataset_construction.video_mining.download import run_download_phase
from dahua_cup.dataset_construction.video_mining.slice_clips import run_slice_phase

# Slice only the raw videos returned in this batch.  Earlier five-second clips
# retain their original paths and are never produced again under a new length.
run_slice_phase(run_download_phase())
PY

echo "[$(date --iso-8601=seconds)] candidate mining batch complete"
