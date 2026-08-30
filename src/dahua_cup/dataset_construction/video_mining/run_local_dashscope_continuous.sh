#!/usr/bin/env bash
# Poll the local clip pool and submit only manifest-missing clips to DashScope.
# The local key file is user-owned and never written to logs or this repository.
set -euo pipefail

log_file="/mnt/f/campus6-miner/dashscope_screening.log"
exec >>"${log_file}" 2>&1

source /home/fjp/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /home/fjp/projects/behaviour_recognition

key_file="${CAMPUS6_DASHSCOPE_KEY_FILE:-/home/fjp/.config/campus6/dashscope_api_key}"
if [[ ! -r "${key_file}" ]]; then
  echo "DashScope local key file is missing or unreadable: ${key_file}" >&2
  exit 2
fi
DASHSCOPE_API_KEY=$(<"${key_file}")
export DASHSCOPE_API_KEY
export CAMPUS6_CLASSIFIER_BACKEND="dashscope"
poll_seconds="${CAMPUS6_SCREEN_POLL_SECONDS:-90}"

while true; do
  echo "[$(date --iso-8601=seconds)] local DashScope screening scan starts"
  python -u -m dahua_cup.dataset_construction.video_mining.run_pipeline --screen || \
    echo "[$(date --iso-8601=seconds)] screening scan exited nonzero; retry after ${poll_seconds}s"
  sleep "${poll_seconds}"
done
