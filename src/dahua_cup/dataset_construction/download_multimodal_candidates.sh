#!/usr/bin/env bash
set -euo pipefail

repo_root="${DAHUA_REPOSITORY_ROOT:-/workspace/code/DAHUA}"
data_root="${DAHUA_DATA_ROOT:-/workspace/data/xzz_data/DAHUA}"
python_bin="${DAHUA_SCREENING_PYTHON:-/root/miniconda3/envs/skel_gcn38/bin/python}"
export DAHUA_YTDLP="${DAHUA_YTDLP:-${repo_root}/.tools/yt-dlp}"
export DAHUA_FFMPEG="${DAHUA_FFMPEG:-/root/miniconda3/envs/skel_gcn38/bin/ffmpeg}"
config="${DAHUA_CANDIDATE_CONFIG:-${repo_root}/dahua_cup/dataset_construction/configs/multimodal_candidates.json}"
log_root="${DAHUA_SCREENING_LOG_ROOT:-${data_root}/runtime/logs/dataset_construction}"

mkdir -p "${log_root}"
timestamp="$(date '+%Y%m%d_%H%M%S')"
log_file="${log_root}/candidate_download_${timestamp}.log"

cd "${repo_root}"
set -o pipefail
{
  needs_tools=true
  quota_args=(--require-quotas)
  for argument in "$@"; do
    if [[ "${argument}" == "--dry-run" ]]; then
      needs_tools=false
      quota_args=()
    fi
  done
  if [[ "${needs_tools}" == true ]]; then
    bash dahua_cup/dataset_construction/setup_candidate_tools.sh
  fi

  "${python_bin}" -m dahua_cup.dataset_construction.download_candidate_sources \
    --config "${config}" \
    "$@"

  "${python_bin}" -m dahua_cup.dataset_construction.build_candidate_manifest \
    --config "${config}" \
    "${quota_args[@]}"
} 2>&1 | tee "${log_file}"
