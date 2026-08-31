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
  --restart              Stop an existing Dahua Web process on the selected port, then start it again.
  -h, --help             Show this help message.
EOF
}

server_conda_root="/root/miniconda3"
restart_web=0

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
    --restart) restart_web=1; shift ;;
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
# skel_gcn38's bundled FFmpeg exposes OpenH264 but not libx264.  OpenH264
# accepts a bitrate rather than x264's preset/CRF options.
export DAHUA_VIS_PREVIEW_CODEC="${DAHUA_VIS_PREVIEW_CODEC:-libopenh264}"

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

port_listener_pids() {
  local proc pid port command_line
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp "sport = :${DAHUA_VIS_PORT}" 2>/dev/null \
      | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u
    return
  elif command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "${DAHUA_VIS_PORT}" 2>/dev/null | tr ' ' '\n' | sed '/^$/d'
    return
  fi
  # Minimal competition containers may not ship ss, lsof, or fuser.  Search
  # only Dahua backend PIDs, then read their exported port from /proc.  This
  # avoids walking every process in a GPU worker container.
  if command -v pgrep >/dev/null 2>&1; then
    while IFS= read -r pid; do
      [[ -r "/proc/${pid}/environ" && -r "/proc/${pid}/cmdline" ]] || continue
      port="$(tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null | sed -n 's/^DAHUA_VIS_PORT=//p' | head -n 1)"
      [[ "${port}" == "${DAHUA_VIS_PORT}" ]] || continue
      command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
      [[ "${command_line}" == *"dahua_cup.backend.app"* ]] || continue
      printf '%s\n' "${pid}"
    done < <(pgrep -f '[d]ahua_cup.backend.app' || true)
    return
  fi
  # Last-resort fallback when process tools are unavailable.
  for proc in /proc/[0-9]*; do
    [[ -r "${proc}/environ" && -r "${proc}/cmdline" ]] || continue
    port="$(tr '\0' '\n' < "${proc}/environ" 2>/dev/null | sed -n 's/^DAHUA_VIS_PORT=//p' | head -n 1)"
    [[ "${port}" == "${DAHUA_VIS_PORT}" ]] || continue
    command_line="$(tr '\0' ' ' < "${proc}/cmdline" 2>/dev/null || true)"
    [[ "${command_line}" == *"dahua_cup.backend.app"* ]] || continue
    pid="${proc##*/}"
    printf '%s\n' "${pid}"
  done
}

mapfile -t listening_pids < <(port_listener_pids)
if ((${#listening_pids[@]})); then
  if (( ! restart_web )); then
    echo "Dahua Web service is already listening on ${DAHUA_VIS_HOST}:${DAHUA_VIS_PORT} (PID ${listening_pids[*]})."
    echo "Reusing the running service; use scripts/run_visualization.sh --restart to restart it explicitly."
    exit 0
  fi
  for pid in "${listening_pids[@]}"; do
    command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
    if [[ "${command_line}" != *"dahua_cup.backend.app"* ]]; then
      echo "Port ${DAHUA_VIS_PORT} is occupied by non-Dahua PID ${pid}; refusing to stop it." >&2
      exit 1
    fi
  done
  echo "Stopping existing Dahua Web service on ${DAHUA_VIS_HOST}:${DAHUA_VIS_PORT} (PID ${listening_pids[*]})"
  kill -TERM "${listening_pids[@]}"
  for _ in {1..20}; do
    sleep 0.25
    mapfile -t listening_pids < <(port_listener_pids)
    ((${#listening_pids[@]})) || break
  done
  if ((${#listening_pids[@]})); then
    echo "Existing Web service did not stop; port ${DAHUA_VIS_PORT} is still occupied by PID ${listening_pids[*]}." >&2
    exit 1
  fi
fi

cd "${DAHUA_CODE_ROOT}"
echo "Activated visualization environment: ${visualization_venv}"
echo "Worker environments: RTMPose=${DAHUA_RTMPOSE_PYTHON}, student=${DAHUA_STUDENT_PYTHON}, teacher=${DAHUA_TEACHER_PYTHON}"
echo "Starting Dahua Web service on ${DAHUA_VIS_HOST}:${DAHUA_VIS_PORT}"
exec "${DAHUA_VIS_PYTHON}" -m dahua_cup.backend.app
