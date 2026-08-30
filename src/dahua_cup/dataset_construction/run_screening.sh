#!/usr/bin/env bash
set -euo pipefail

repo_root="${DAHUA_REPOSITORY_ROOT:-/workspace/code/DAHUA}"
data_root="${DAHUA_DATA_ROOT:-/workspace/data/xzz_data/DAHUA}"
python_bin="${DAHUA_SCREENING_PYTHON:-/root/miniconda3/envs/skel_gcn38/bin/python}"
input_root="${DAHUA_SCREENING_INPUT:-${data_root}/datasets/campus6_candidates}"
output_root="${DAHUA_SCREENING_OUTPUT:-${data_root}/datasets/campus6_screened}"
log_root="${DAHUA_SCREENING_LOG_ROOT:-${data_root}/runtime/logs/dataset_construction}"
config="${DAHUA_SCREENING_CONFIG:-${repo_root}/dahua_cup/dataset_construction/configs/hosted_screening.json}"
credentials_file="${DAHUA_SCREENING_CREDENTIALS:-${repo_root}/.secrets/campus6_screening.env}"

if [[ -f "${credentials_file}" ]]; then
  # shellcheck disable=SC1090
  source "${credentials_file}"
fi

mkdir -p "${input_root}" "${output_root}" "${log_root}"

if [[ ! -x "${python_bin}" ]]; then
  echo "[campus6-screening] Python is not executable: ${python_bin}" >&2
  exit 2
fi
if [[ ! -f "${config}" ]]; then
  echo "[campus6-screening] Configuration does not exist: ${config}" >&2
  exit 2
fi

timestamp="$(date '+%Y%m%d_%H%M%S')"
log_file="${log_root}/hosted_screening_${timestamp}.log"

echo "[campus6-screening] repository: ${repo_root}"
echo "[campus6-screening] candidates: ${input_root}"
echo "[campus6-screening] output: ${output_root}"
echo "[campus6-screening] log: ${log_file}"
echo "[campus6-screening] GPU is not used by this workflow"

cd "${repo_root}"
set -o pipefail
source_args=(--input "${input_root}")
for argument in "$@"; do
  if [[ "${argument}" == "--manifest" || "${argument}" == "--input" ]]; then
    source_args=()
    break
  fi
done
"${python_bin}" -m dahua_cup.dataset_construction.screen_dataset \
  --config "${config}" \
  "${source_args[@]}" \
  --output "${output_root}" \
  "$@" \
  2>&1 | tee "${log_file}"
