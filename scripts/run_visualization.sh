#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_visualization.sh [options]

Select an environment by its server name (skel_gcn38, rtmpose26, or llm_env)
or by its absolute environment prefix.  Each role runs with its own absolute
Python interpreter.

Options:
  --web-env ENV          Environment for the FastAPI Web process.
  --rtmpose-env ENV      Environment for RTMPose extraction workers.
  --student-env ENV      Environment for Campus6 student inference workers.
  --teacher-env ENV      Environment for Qwen teacher workers.
  --web-python PATH      Explicit Python executable for the Web process.
  --rtmpose-python PATH  Explicit Python executable for RTMPose workers.
  --student-python PATH  Explicit Python executable for student workers.
  --teacher-python PATH  Explicit Python executable for teacher workers.
  -h, --help             Show this help message.
EOF
}

server_conda_root="/root/miniconda3"

environment_prefix() {
  local environment_name="$1"
  case "${environment_name}" in
    skel_gcn38) printf '%s\n' "${server_conda_root}/envs/skel_gcn38" ;;
    rtmpose26) printf '%s\n' "${server_conda_root}/envs/rtmpose26" ;;
    llm_env) printf '%s\n' "/workspace/code/envs/llm_env" ;;
    /*) printf '%s\n' "${environment_name}" ;;
    *)
      echo "Unknown environment: ${environment_name}" >&2
      echo "Use skel_gcn38, rtmpose26, llm_env, or an absolute environment path." >&2
      return 2
      ;;
  esac
}

environment_python() {
  local environment_name="$1"
  local prefix
  prefix="$(environment_prefix "${environment_name}")"
  if [[ ! -x "${prefix}/bin/python" ]]; then
    echo "Python executable not found for environment ${environment_name}: ${prefix}/bin/python" >&2
    return 2
  fi
  printf '%s\n' "${prefix}/bin/python"
}

while (($#)); do
  case "$1" in
    --web-env) DAHUA_VIS_ENV="$2"; shift 2 ;;
    --rtmpose-env) DAHUA_RTMPOSE_ENV="$2"; shift 2 ;;
    --student-env) DAHUA_STUDENT_ENV="$2"; shift 2 ;;
    --teacher-env) DAHUA_TEACHER_ENV="$2"; shift 2 ;;
    --web-python) DAHUA_VIS_PYTHON="$2"; shift 2 ;;
    --rtmpose-python) DAHUA_RTMPOSE_PYTHON="$2"; shift 2 ;;
    --student-python) DAHUA_STUDENT_PYTHON="$2"; shift 2 ;;
    --teacher-python) DAHUA_TEACHER_PYTHON="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DAHUA_CODE_ROOT="${DAHUA_CODE_ROOT:-${code_root}}"
export PYTHONPATH="${DAHUA_CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# The backend validates these variables as files, so a bare `python` command
# is insufficient.  Defaults keep the Campus6 Web and workers in skel_gcn38,
# while Qwen stays isolated in llm_env.
: "${DAHUA_VIS_ENV:=skel_gcn38}"
: "${DAHUA_RTMPOSE_ENV:=skel_gcn38}"
: "${DAHUA_STUDENT_ENV:=skel_gcn38}"
: "${DAHUA_TEACHER_ENV:=llm_env}"

if [[ -n "${DAHUA_VIS_VENV:-}" ]]; then
  visualization_venv="${DAHUA_VIS_VENV}"
elif [[ -x "$(environment_python "${DAHUA_VIS_ENV}")" ]]; then
  visualization_venv="$(environment_prefix "${DAHUA_VIS_ENV}")"
elif [[ -f "${DAHUA_CODE_ROOT}/.venv/bin/activate" ]]; then
  visualization_venv="${DAHUA_CODE_ROOT}/.venv"
else
  echo "No visualization Python environment found." >&2
  echo "Set DAHUA_VIS_VENV to a virtual environment containing the Web dependencies." >&2
  exit 2
fi

if [[ "${visualization_venv}" == "${server_conda_root}"/envs/* && -f "${server_conda_root}/bin/activate" ]]; then
  source "${server_conda_root}/bin/activate" "${visualization_venv}"
  visualization_python="${visualization_venv}/bin/python"
elif [[ -f "${visualization_venv}/bin/activate" ]]; then
  source "${visualization_venv}/bin/activate"
  visualization_python="${visualization_venv}/bin/python"
elif [[ -x "${visualization_venv}/bin/python" ]]; then
  # A Conda prefix supplied explicitly may not contain bin/activate.  The
  # absolute interpreter still gives every worker the intended environment.
  visualization_python="${visualization_venv}/bin/python"
else
  echo "Visualization virtual environment is invalid: ${visualization_venv}" >&2
  exit 2
fi

export DAHUA_DATA_ROOT="${DAHUA_DATA_ROOT:-${DAHUA_CODE_ROOT}/runtime}"
export DAHUA_VIS_RUNTIME_ROOT="${DAHUA_VIS_RUNTIME_ROOT:-${DAHUA_DATA_ROOT}}"
export DAHUA_VIS_SOURCE_ROOT="${DAHUA_VIS_SOURCE_ROOT:-${DAHUA_VIS_RUNTIME_ROOT}/videos}"
export DAHUA_VIS_MANIFEST="${DAHUA_VIS_MANIFEST:-${DAHUA_VIS_RUNTIME_ROOT}/campus6_manifest.csv}"
export DAHUA_CAMPUS6_DEPLOY_CONFIG="${DAHUA_CAMPUS6_DEPLOY_CONFIG:-${DAHUA_CODE_ROOT}/third_party/ProtoGCN/configs/campus6/rtmpose26_k400_2d_gap_full.py}"
export DAHUA_CAMPUS6_DEPLOYMENT_CHECKPOINT="${DAHUA_CAMPUS6_DEPLOYMENT_CHECKPOINT:-${DAHUA_CODE_ROOT}/models/student/M1KD.int8.pt}"

: "${DAHUA_VIS_PYTHON:=${visualization_python}}"
: "${DAHUA_RTMPOSE_PYTHON:=$(environment_python "${DAHUA_RTMPOSE_ENV}")}"
: "${DAHUA_STUDENT_PYTHON:=$(environment_python "${DAHUA_STUDENT_ENV}")}"
: "${DAHUA_TEACHER_PYTHON:=$(environment_python "${DAHUA_TEACHER_ENV}")}"
: "${DAHUA_VIS_HOST:=0.0.0.0}"
: "${DAHUA_VIS_PORT:=8000}"
for python_bin in "${DAHUA_VIS_PYTHON}" "${DAHUA_RTMPOSE_PYTHON}" "${DAHUA_STUDENT_PYTHON}" "${DAHUA_TEACHER_PYTHON}"; do
  [[ -x "${python_bin}" ]] || { echo "Python executable not found: ${python_bin}" >&2; exit 2; }
done
export DAHUA_VIS_PYTHON DAHUA_RTMPOSE_PYTHON DAHUA_STUDENT_PYTHON DAHUA_TEACHER_PYTHON DAHUA_VIS_HOST DAHUA_VIS_PORT

[[ -f "${DAHUA_CAMPUS6_DEPLOY_CONFIG}" ]] || { echo "Campus6 deployment config not found" >&2; exit 2; }
[[ -f "${DAHUA_CAMPUS6_DEPLOYMENT_CHECKPOINT}" ]] || { echo "M1KD INT8 checkpoint not found" >&2; exit 2; }
cd "${DAHUA_CODE_ROOT}"
echo "Activated visualization environment: ${visualization_venv}"
echo "Worker environments: RTMPose=${DAHUA_RTMPOSE_PYTHON}, student=${DAHUA_STUDENT_PYTHON}, teacher=${DAHUA_TEACHER_PYTHON}"
echo "Starting Dahua Web service on ${DAHUA_VIS_HOST}:${DAHUA_VIS_PORT}"
exec "${DAHUA_VIS_PYTHON}" -m dahua_cup.backend.app
