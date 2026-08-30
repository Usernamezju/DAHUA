#!/usr/bin/env bash
set -euo pipefail

repo_root="${DAHUA_REPOSITORY_ROOT:-/workspace/code/DAHUA}"
data_root="${DAHUA_DATA_ROOT:-/workspace/data/xzz_data/DAHUA}"
python_bin="${DAHUA_SCREENING_PYTHON:-/root/miniconda3/envs/skel_gcn38/bin/python}"
candidate_config="${DAHUA_CANDIDATE_CONFIG:-${repo_root}/dahua_cup/dataset_construction/configs/multimodal_candidates.json}"
screening_config="${DAHUA_SCREENING_CONFIG:-${repo_root}/dahua_cup/dataset_construction/configs/hosted_screening.json}"
manifest="${data_root}/datasets/campus6_candidates/manifests/multimodal_candidates.jsonl"
output="${data_root}/datasets/campus6_screened"
quota_args=(--require-quotas)
for argument in "$@"; do
  if [[ "${argument}" == "--dry-run" ]]; then
    quota_args=()
  fi
done

cd "${repo_root}"

"${python_bin}" -m dahua_cup.dataset_construction.build_candidate_manifest \
  --config "${candidate_config}" \
  "${quota_args[@]}"

bash dahua_cup/dataset_construction/run_screening.sh \
  --config "${screening_config}" \
  --manifest "${manifest}" \
  --output "${output}" \
  --stop-when-quotas-met \
  "$@"
