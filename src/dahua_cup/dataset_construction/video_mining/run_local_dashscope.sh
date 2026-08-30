#!/usr/bin/env bash
# Launch local DashScope screening from a user-owned local secret file.
set -euo pipefail

log_file="/mnt/f/campus6-miner/dashscope_screening.log"
exec >>"${log_file}" 2>&1

echo "[$(date --iso-8601=seconds)] local DashScope screening starts"
source /home/fjp/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /home/fjp/projects/behaviour_recognition

key_file="${CAMPUS6_DASHSCOPE_KEY_FILE:-/home/fjp/.config/campus6/dashscope_api_key}"
if [[ ! -r "${key_file}" ]]; then
  echo "DashScope local key file is missing or unreadable: ${key_file}" >&2
  exit 2
fi

# Command substitution removes a trailing newline; the file itself is created
# with mode 600 and is outside the repository.
DASHSCOPE_API_KEY=$(<"${key_file}")
export DASHSCOPE_API_KEY
if [[ -z "${DASHSCOPE_API_KEY}" ]]; then
  echo "DashScope local key file is empty" >&2
  exit 2
fi
export CAMPUS6_CLASSIFIER_BACKEND="dashscope"

exec python -u -m dahua_cup.dataset_construction.video_mining.run_pipeline --screen
