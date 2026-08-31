#!/usr/bin/env bash
set -euo pipefail

code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DAHUA_CODE_ROOT="${DAHUA_CODE_ROOT:-${code_root}}"
export PYTHONPATH="${DAHUA_CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# The demo web serves samples and runs the M1KD student in-process, so it
# needs the ProtoGCN environment (skel_gcn38 on the server).  RTMPose
# extraction for imported videos runs in a separate mmpose>=1.0 environment.
if [[ -n "${DAHUA_DEMO_VENV:-}" ]]; then
  demo_venv="${DAHUA_DEMO_VENV}"
elif [[ -x "/root/miniconda3/envs/skel_gcn38/bin/python" ]]; then
  demo_venv="/root/miniconda3/envs/skel_gcn38"
elif [[ -x "${DAHUA_CODE_ROOT}/.venv/bin/python" ]]; then
  demo_venv="${DAHUA_CODE_ROOT}/.venv"
else
  echo "No demo Python environment found." >&2
  echo "Set DAHUA_DEMO_VENV to an environment with fastapi and ProtoGCN dependencies." >&2
  exit 2
fi
[[ -x "${demo_venv}/bin/python" ]] || {
  echo "Demo Python environment is invalid: ${demo_venv}" >&2
  exit 2
}
export PATH="${demo_venv}/bin:${PATH}"

export DAHUA_DEMO_DATASET="${DAHUA_DEMO_DATASET:-${DAHUA_CODE_ROOT}/dataset}"
export DAHUA_DEMO_POSE_PYTHON="${DAHUA_DEMO_POSE_PYTHON:-/root/miniconda3/envs/rtmpose26/bin/python}"
export DAHUA_DEMO_DEVICE="${DAHUA_DEMO_DEVICE:-cuda:1}"
: "${DAHUA_DEMO_HOST:=0.0.0.0}"
: "${DAHUA_DEMO_PORT:=8010}"
export DAHUA_DEMO_HOST DAHUA_DEMO_PORT

[[ -f "${DAHUA_DEMO_DATASET}/selection.json" ]] || {
  echo "Demo dataset not found under ${DAHUA_DEMO_DATASET}" >&2
  exit 2
}
cd "${DAHUA_CODE_ROOT}"
echo "Activated demo environment: ${demo_venv}"
echo "Starting Dahua demo web on ${DAHUA_DEMO_HOST}:${DAHUA_DEMO_PORT}"
exec "${demo_venv}/bin/python" -m dahua_cup.demo.app
