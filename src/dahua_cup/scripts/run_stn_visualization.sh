#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
code_root="$(cd "${script_dir}/../.." && pwd)"
server_data_root="/workspace/data/xzz_data/DAHUA"
if [[ -d "${server_data_root}" ]]; then
    data_root="${DAHUA_DATA_ROOT:-${server_data_root}}"
else
    data_root="${DAHUA_DATA_ROOT:-${code_root}}"
fi

checkpoint="${DAHUA_STN_CHECKPOINT:-${data_root}/models/stn/set_aware_group/best.pt}"
teacher_jsonl="${DAHUA_STN_TEACHER_JSONL:-${data_root}/datasets/stn/yolo_tracks.jsonl}"
output_root="${DAHUA_STN_VIS_OUTPUT:-${data_root}/runtime/stn_visualization}"
device="${DAHUA_STN_DEVICE:-cuda:0}"
host="${DAHUA_STN_VIS_HOST:-0.0.0.0}"
port="${DAHUA_STN_VIS_PORT:-8010}"
batch_size="${DAHUA_STN_BATCH_SIZE:-4}"
window_step="${DAHUA_STN_WINDOW_STEP:-8}"
python_bin="${DAHUA_STN_PYTHON:-python}"

mkdir -p "${output_root}"
cd "${code_root}"

"${python_bin}" -c "import cv2, fastapi, numpy, torch, uvicorn" >/dev/null

exec "${python_bin}" -m dahua_cup.stn.visualization_app \
    --checkpoint \
    "${checkpoint}" \
    --teacher-jsonl \
    "${teacher_jsonl}" \
    --output-dir \
    "${output_root}" \
    --device \
    "${device}" \
    --batch-size \
    "${batch_size}" \
    --window-step \
    "${window_step}" \
    --host \
    "${host}" \
    --port \
    "${port}" \
    "$@"
