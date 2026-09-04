#!/usr/bin/env bash
# Start the Campus6 Web workbench on a developer workstation.
#
# RGB upload, RTMDet/RTMPose and ProtoGCN run on this computer.  The optional
# Qwen teacher is configured in the Web settings and stays on the remote host.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_local_visualization.sh [options]

Options:
  --web-python PATH      Python with FastAPI, Uvicorn and python-multipart.
  --rtmpose-python PATH  Python with MMDetection, MMPose and OpenCV.
  --student-python PATH  Python with PyTorch, MMCV and ProtoGCN dependencies.
  --host HOST            Bind host (default: 127.0.0.1).
  --port PORT            Bind port (default: 8000).
  --runtime-root PATH    Local runtime/artifact root.
  --dataset-root PATH    Root containing campus6_baseline/campus_increment/campus_all.
  --device MODE          auto (default), gpu, or cpu for local pose + GCN.
  -h, --help             Show this help.

The current active Python is used for all three roles only when role-specific
paths are not supplied.  Separate role environments are recommended.
EOF
}

code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
web_python="${DAHUA_VIS_PYTHON:-$(command -v python)}"
rtmpose_python="${DAHUA_RTMPOSE_PYTHON:-$web_python}"
student_python="${DAHUA_STUDENT_PYTHON:-$web_python}"
host="${DAHUA_VIS_HOST:-127.0.0.1}"
port="${DAHUA_VIS_PORT:-8000}"
runtime_root="${DAHUA_VIS_RUNTIME_ROOT:-${code_root}/runtime/local_web}"
dataset_root="${DAHUA_DATASET_ROOT:-${code_root}/dataset}"
device_mode="${DAHUA_LOCAL_INFERENCE_DEVICE:-auto}"

while (($#)); do
  case "$1" in
    --web-python) web_python="$2"; shift 2 ;;
    --rtmpose-python) rtmpose_python="$2"; shift 2 ;;
    --student-python) student_python="$2"; shift 2 ;;
    --host) host="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --runtime-root) runtime_root="$2"; shift 2 ;;
    --dataset-root) dataset_root="$2"; shift 2 ;;
    --device) device_mode="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for python_bin in "$web_python" "$rtmpose_python" "$student_python"; do
  [[ -x "$python_bin" ]] || { echo "Python executable not found: $python_bin" >&2; exit 2; }
done

"$web_python" - <<'PY'
import fastapi, multipart, uvicorn
print("Web environment OK")
PY
"$rtmpose_python" - <<'PY'
import cv2, mmdet, mmpose
print("RTMDet/RTMPose environment OK")
PY
"$student_python" - <<'PY'
import mmcv, torch
print("ProtoGCN environment OK; CUDA visible:", torch.cuda.is_available())
PY

case "$device_mode" in
  auto|cpu|gpu) ;;
  *) echo "--device must be auto, cpu, or gpu" >&2; exit 2 ;;
esac
if [[ "$device_mode" == "auto" || "$device_mode" == "gpu" ]]; then
  pose_cuda="$($rtmpose_python -c 'import torch; print(int(torch.cuda.is_available()))')"
  student_cuda="$($student_python -c 'import torch; print(int(torch.cuda.is_available()))')"
  if [[ "$pose_cuda" == "1" && "$student_cuda" == "1" ]]; then
    resolved_device="gpu"
  elif [[ "$device_mode" == "gpu" ]]; then
    echo "--device gpu was requested, but RTMPose or ProtoGCN cannot see CUDA." >&2
    exit 2
  else
    resolved_device="cpu"
  fi
else
  resolved_device="cpu"
fi

export DAHUA_CODE_ROOT="$code_root"
export PYTHONPATH="$code_root/src${PYTHONPATH:+:$PYTHONPATH}"
export DAHUA_DATA_ROOT="$runtime_root"
export DAHUA_VIS_RUNTIME_ROOT="$runtime_root"
export DAHUA_VIS_SOURCE_ROOT="${DAHUA_VIS_SOURCE_ROOT:-${runtime_root}/videos}"
export DAHUA_VIS_MANIFEST="${DAHUA_VIS_MANIFEST:-${runtime_root}/campus6_manifest.csv}"
export DAHUA_DATASET_ROOT="$dataset_root"
export DAHUA_VIS_PYTHON="$web_python"
export DAHUA_RTMPOSE_PYTHON="$rtmpose_python"
export DAHUA_STUDENT_PYTHON="$student_python"
export DAHUA_VIS_HOST="$host"
export DAHUA_VIS_PORT="$port"
# Campus6 local default: use the downloaded TensorRT FP16 engines.  The
# Python environment selected by --rtmpose-python must provide
# mmdeploy_runtime and a CUDA-compatible TensorRT installation.
export DAHUA_CAMPUS6_BACKEND="${DAHUA_CAMPUS6_BACKEND:-rtmpose17}"
export DAHUA_POSE_BACKEND="${DAHUA_POSE_BACKEND:-tensorrt_fp16}"
export DAHUA_POSE_DEVICE="${DAHUA_POSE_DEVICE:-cuda:0}"
export DAHUA_POSE_FP16_MODEL_ROOT="${DAHUA_POSE_FP16_MODEL_ROOT:-${code_root}/models/pose/fp16}"
if [[ -n "${DAHUA_TRT_LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${DAHUA_TRT_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -n "${DAHUA_TRT_LD_PRELOAD:-}" ]]; then
  export LD_PRELOAD="${DAHUA_TRT_LD_PRELOAD}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
unset DAHUA_LOCAL_CPU_INFERENCE
export DAHUA_LOCAL_INFERENCE_DEVICE="$resolved_device"
export DAHUA_CAMPUS6_DEPLOY_CONFIG="${DAHUA_CAMPUS6_DEPLOY_CONFIG:-${code_root}/third_party/ProtoGCN/configs/campus6/rtm_s_coco17_k400_2d_gap_full.py}"
export DAHUA_CAMPUS6_DEPLOYMENT_CHECKPOINT="${DAHUA_CAMPUS6_DEPLOYMENT_CHECKPOINT:-${code_root}/models/student/M1KD.int8.pt}"

[[ -f "$DAHUA_CAMPUS6_DEPLOY_CONFIG" ]] || { echo "Campus6 deployment config not found: $DAHUA_CAMPUS6_DEPLOY_CONFIG" >&2; exit 2; }
[[ -f "$DAHUA_CAMPUS6_DEPLOYMENT_CHECKPOINT" ]] || { echo "Campus6 deployment checkpoint not found: $DAHUA_CAMPUS6_DEPLOYMENT_CHECKPOINT" >&2; exit 2; }

if [[ "$DAHUA_CAMPUS6_BACKEND" == "rtmpose17" && "$DAHUA_POSE_BACKEND" == "tensorrt_fp16" ]]; then
  if [[ "$resolved_device" != "gpu" ]]; then
    echo "TensorRT FP16 pose extraction requires a CUDA-visible GPU; use --device gpu or select an MMPose diagnostic backend." >&2
    exit 2
  fi
  "$rtmpose_python" -c "import mmdeploy_runtime, torch; assert torch.cuda.is_available()" >/dev/null
  [[ -f "$DAHUA_POSE_FP16_MODEL_ROOT/rtmdet_nano_person/end2end.engine" ]] || { echo "TensorRT FP16 detector engine not found: $DAHUA_POSE_FP16_MODEL_ROOT/rtmdet_nano_person/end2end.engine" >&2; exit 2; }
  [[ -f "$DAHUA_POSE_FP16_MODEL_ROOT/rtmpose_s/end2end.engine" ]] || { echo "TensorRT FP16 pose engine not found: $DAHUA_POSE_FP16_MODEL_ROOT/rtmpose_s/end2end.engine" >&2; exit 2; }
fi

echo "Starting local Campus6 Web at http://${host}:${port}"
echo "Workers: RTMPose=${rtmpose_python} (${resolved_device}); ProtoGCN=${student_python} (${resolved_device}); Qwen=remote if configured"
exec "$web_python" -m dahua_cup.backend.app
