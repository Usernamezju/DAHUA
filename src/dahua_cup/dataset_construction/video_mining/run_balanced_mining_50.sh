#!/usr/bin/env bash
# Download exactly up to 50 new raw candidates per Campus6 class, resumably.
# No VLM API requests occur here; screening is started after mining completes.
set -euo pipefail

log_file="/mnt/f/campus6-miner/balanced_mining_50.log"
exec >>"${log_file}" 2>&1

echo "[$(date --iso-8601=seconds)] balanced mining starts (50 videos/class)"
source /home/fjp/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /home/fjp/projects/behaviour_recognition

export CAMPUS6_CLIP_DURATION=10
export CAMPUS6_CLIP_OVERLAP=2
python -u -m dahua_cup.dataset_construction.video_mining.balanced_mining \
  --target-per-class 50

echo "[$(date --iso-8601=seconds)] balanced mining complete"
