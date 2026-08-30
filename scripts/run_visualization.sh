#!/usr/bin/env bash
set -euo pipefail

code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DAHUA_CODE_ROOT="${DAHUA_CODE_ROOT:-${code_root}}"
export DAHUA_DATA_ROOT="${DAHUA_DATA_ROOT:-${DAHUA_CODE_ROOT}/runtime-data}"
export DAHUA_VIS_RUNTIME_ROOT="${DAHUA_VIS_RUNTIME_ROOT:-${DAHUA_DATA_ROOT}/runtime/visualization}"
export DAHUA_CAMPUS6_CONFIG="${DAHUA_CAMPUS6_CONFIG:-${DAHUA_CODE_ROOT}/third_party/ProtoGCN/configs/campus6/rtmpose26_k400_2d_gap_full.py}"
export DAHUA_CAMPUS6_CHECKPOINT="${DAHUA_CAMPUS6_CHECKPOINT:-${DAHUA_CODE_ROOT}/models/student/campus6_protogcn_gap_fp32_epoch40.pth}"

: "${DAHUA_VIS_PYTHON:=python}"
: "${DAHUA_RTMPOSE_PYTHON:=${DAHUA_VIS_PYTHON}}"
: "${DAHUA_STUDENT_PYTHON:=${DAHUA_VIS_PYTHON}}"
export DAHUA_VIS_PYTHON DAHUA_RTMPOSE_PYTHON DAHUA_STUDENT_PYTHON

[[ -f "${DAHUA_CAMPUS6_CONFIG}" ]] || { echo "Campus6 config not found" >&2; exit 2; }
[[ -f "${DAHUA_CAMPUS6_CHECKPOINT}" ]] || { echo "Campus6 FP32 checkpoint not found" >&2; exit 2; }
cd "${DAHUA_CODE_ROOT}"
exec "${DAHUA_VIS_PYTHON}" -m dahua_cup.backend.app
