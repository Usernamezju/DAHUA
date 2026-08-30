#!/usr/bin/env bash
set -euo pipefail

repo_root="${DAHUA_REPOSITORY_ROOT:-/workspace/code/DAHUA}"
data_root="${DAHUA_DATA_ROOT:-/workspace/data/xzz_data/DAHUA}"
python_bin="${DAHUA_SCREENING_PYTHON:-/root/miniconda3/envs/skel_gcn38/bin/python}"
candidate_config="${DAHUA_CANDIDATE_CONFIG:-${repo_root}/dahua_cup/dataset_construction/configs/multimodal_candidates.json}"
screening_config="${DAHUA_SCREENING_CONFIG:-${repo_root}/dahua_cup/dataset_construction/configs/qwen_incremental_screening.json}"
candidate_root="${DAHUA_INCREMENTAL_CANDIDATE_ROOT:-${data_root}/datasets/campus6_candidates_incremental_qwen}"
output_root="${DAHUA_INCREMENTAL_SCREENING_OUTPUT:-${data_root}/datasets/campus6_screened_qwen_incremental}"
manifest="${candidate_root}/manifests/uav_human_hmdb51.jsonl"
summary="${candidate_root}/manifests/candidate_summary.json"

if [[ ! -x "${python_bin}" ]]; then
  echo "[campus6-qwen-incremental] Python is not executable: ${python_bin}" >&2
  exit 2
fi

mkdir -p "${candidate_root}/manifests" "${output_root}"

cd "${repo_root}"
"${python_bin}" -m dahua_cup.dataset_construction.build_candidate_manifest \
  --config "${candidate_config}" \
  --output "${manifest}" \
  --summary "${summary}" \
  --only-source-dataset UAV-Human \
  --only-source-dataset HMDB51

echo "[campus6-qwen-incremental] manifest: ${manifest}"
echo "[campus6-qwen-incremental] output: ${output_root}"
echo "[campus6-qwen-incremental] reviewers: qwen3-vl-flash + qwen3-vl-plus; disagreements use qwen3-vl-235b-a22b-thinking"

DAHUA_SCREENING_CONFIG="${screening_config}" \
DAHUA_SCREENING_INPUT="${candidate_root}" \
DAHUA_SCREENING_OUTPUT="${output_root}" \
bash "${repo_root}/dahua_cup/dataset_construction/run_screening.sh" \
  --manifest "${manifest}" \
  "$@"
