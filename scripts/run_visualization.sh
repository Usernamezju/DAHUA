#!/usr/bin/env bash
set -euo pipefail

code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DAHUA_CODE_ROOT="${DAHUA_CODE_ROOT:-${code_root}}"
export PYTHONPATH="${DAHUA_CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export DAHUA_DATA_ROOT="${DAHUA_DATA_ROOT:-${DAHUA_CODE_ROOT}/runtime-data}"
export DAHUA_VIS_RUNTIME_ROOT="${DAHUA_VIS_RUNTIME_ROOT:-${DAHUA_DATA_ROOT}/runtime/campus6_product}"
export DAHUA_VIS_SOURCE_ROOT="${DAHUA_VIS_SOURCE_ROOT:-${DAHUA_VIS_RUNTIME_ROOT}}"
export DAHUA_VIS_MANIFEST="${DAHUA_VIS_MANIFEST:-${DAHUA_VIS_RUNTIME_ROOT}/campus6_manifest.csv}"
export DAHUA_CAMPUS6_DEPLOY_CONFIG="${DAHUA_CAMPUS6_DEPLOY_CONFIG:-${DAHUA_CODE_ROOT}/third_party/ProtoGCN/configs/campus6/rtmpose26_k400_2d_gap_full.py}"
export DAHUA_CAMPUS6_DEPLOYMENT_CHECKPOINT="${DAHUA_CAMPUS6_DEPLOYMENT_CHECKPOINT:-${DAHUA_CODE_ROOT}/models/student/M1KD.int8.pt}"

: "${DAHUA_VIS_PYTHON:=python}"
: "${DAHUA_RTMPOSE_PYTHON:=${DAHUA_VIS_PYTHON}}"
: "${DAHUA_STUDENT_PYTHON:=${DAHUA_VIS_PYTHON}}"
export DAHUA_VIS_PYTHON DAHUA_RTMPOSE_PYTHON DAHUA_STUDENT_PYTHON

[[ -f "${DAHUA_CAMPUS6_DEPLOY_CONFIG}" ]] || { echo "Campus6 deployment config not found" >&2; exit 2; }
[[ -f "${DAHUA_CAMPUS6_DEPLOYMENT_CHECKPOINT}" ]] || { echo "M1KD INT8 checkpoint not found" >&2; exit 2; }
cd "${DAHUA_CODE_ROOT}"
exec "${DAHUA_VIS_PYTHON}" -m dahua_cup.backend.app
