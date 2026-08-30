#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash dahua_cup/scripts/run_visualization.sh [options]

Options override teacher_routing.yaml:
  --routing-config PATH
  --evolution-config PATH
  --enable-evolution
  --disable-evolution
  --confidence-threshold FLOAT
  --margin-threshold FLOAT
  --pose-quality-threshold FLOAT
  --nanodet-config PATH
  --nanodet-checkpoint PATH
  --nanodet-python PATH
  --nanodet-device DEVICE
  --nanodet-confidence FLOAT
  --nanodet-context-factor FLOAT
  --nanodet-output-size INT
  --nanodet-crowd-policy top2|full-frame
  --enable-nanodet
  --disable-nanodet
  --qwen-model-dir PATH
  --qwen-dtype float16|bfloat16|float32
  --qwen-device-map DEVICE_MAP
  --qwen-attention sdpa|eager|flash_attention_2
  --qwen-max-frames INT
  --qwen-max-new-tokens INT
  --qwen-retries INT
  --pose-detection-confidence FLOAT
  --pose-presence-confidence FLOAT
  --pose-tracking-confidence FLOAT
  --joint-score-threshold FLOAT
  --instability-threshold FLOAT
  --conflict-confidence FLOAT
  --conflict-priority INT
  --pose-priority INT
  --instability-priority INT
  --teacher-review-priority INT
  --teacher-unavailable-priority INT
  -h, --help
EOF
}

while (($#)); do
    case "$1" in
        --routing-config)
            export DAHUA_TEACHER_ROUTING_CONFIG="$2"
            shift 2
            ;;
        --evolution-config)
            export DAHUA_AUTO_EVOLUTION_CONFIG="$2"
            shift 2
            ;;
        --enable-evolution)
            export DAHUA_AUTO_EVOLUTION=1
            shift
            ;;
        --disable-evolution)
            export DAHUA_AUTO_EVOLUTION=0
            shift
            ;;
        --confidence-threshold)
            export DAHUA_TEACHER_TRIGGER_CONFIDENCE="$2"
            shift 2
            ;;
        --margin-threshold)
            export DAHUA_TEACHER_TRIGGER_MARGIN="$2"
            shift 2
            ;;
        --pose-quality-threshold)
            export DAHUA_POSE_QUALITY_THRESHOLD="$2"
            shift 2
            ;;
        --nanodet-config)
            export DAHUA_NANODET_CONFIG="$2"
            shift 2
            ;;
        --nanodet-checkpoint)
            export DAHUA_NANODET_CHECKPOINT="$2"
            shift 2
            ;;
        --nanodet-python)
            export DAHUA_NANODET_PYTHON="$2"
            shift 2
            ;;
        --nanodet-device)
            export DAHUA_NANODET_DEVICE="$2"
            shift 2
            ;;
        --nanodet-confidence)
            export DAHUA_NANODET_CONFIDENCE="$2"
            shift 2
            ;;
        --nanodet-context-factor)
            export DAHUA_NANODET_CONTEXT_FACTOR="$2"
            shift 2
            ;;
        --nanodet-output-size)
            export DAHUA_NANODET_OUTPUT_SIZE="$2"
            shift 2
            ;;
        --nanodet-crowd-policy)
            export DAHUA_NANODET_CROWD_POLICY="$2"
            shift 2
            ;;
        --enable-nanodet)
            export DAHUA_NANODET_ENABLED=1
            shift
            ;;
        --disable-nanodet)
            export DAHUA_NANODET_ENABLED=0
            shift
            ;;
        --qwen-model-dir)
            export DAHUA_QWEN_MODEL_DIR="$2"
            shift 2
            ;;
        --qwen-dtype)
            case "$2" in
                float16|bfloat16|float32) ;;
                *) printf 'ERROR: invalid --qwen-dtype: %s\n' "$2" >&2; exit 2 ;;
            esac
            export DAHUA_QWEN_DTYPE="$2"
            shift 2
            ;;
        --qwen-device-map)
            export DAHUA_QWEN_DEVICE_MAP="$2"
            shift 2
            ;;
        --qwen-attention)
            case "$2" in
                sdpa|eager|flash_attention_2) ;;
                *) printf 'ERROR: invalid --qwen-attention: %s\n' "$2" >&2; exit 2 ;;
            esac
            export DAHUA_QWEN_ATTN_IMPLEMENTATION="$2"
            shift 2
            ;;
        --qwen-max-frames)
            export DAHUA_QWEN_MAX_FRAMES="$2"
            shift 2
            ;;
        --qwen-max-new-tokens)
            export DAHUA_QWEN_MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --qwen-retries)
            export DAHUA_QWEN_RETRIES="$2"
            shift 2
            ;;
        --pose-detection-confidence)
            export DAHUA_MEDIAPIPE_DETECTION_CONFIDENCE="$2"
            shift 2
            ;;
        --pose-presence-confidence)
            export DAHUA_MEDIAPIPE_PRESENCE_CONFIDENCE="$2"
            shift 2
            ;;
        --pose-tracking-confidence)
            export DAHUA_MEDIAPIPE_TRACKING_CONFIDENCE="$2"
            shift 2
            ;;
        --joint-score-threshold)
            export DAHUA_MEDIAPIPE_JOINT_SCORE_THRESHOLD="$2"
            shift 2
            ;;
        --instability-threshold)
            export DAHUA_STUDENT_INSTABILITY_THRESHOLD="$2"
            shift 2
            ;;
        --conflict-confidence)
            export DAHUA_TEACHER_CONFLICT_CONFIDENCE="$2"
            shift 2
            ;;
        --conflict-priority)
            export DAHUA_PRIORITY_TEACHER_CONFLICT="$2"
            shift 2
            ;;
        --pose-priority)
            export DAHUA_PRIORITY_POSE_QUALITY="$2"
            shift 2
            ;;
        --instability-priority)
            export DAHUA_PRIORITY_STUDENT_INSTABILITY="$2"
            shift 2
            ;;
        --teacher-review-priority)
            export DAHUA_PRIORITY_TEACHER_REVIEW="$2"
            shift 2
            ;;
        --teacher-unavailable-priority)
            export DAHUA_PRIORITY_TEACHER_UNAVAILABLE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
code_root="$(cd "${script_dir}/../.." && pwd)"

export DAHUA_CODE_ROOT="${DAHUA_CODE_ROOT:-${code_root}}"
server_data_root="/workspace/data/xzz_data/DAHUA"
if [[ -d "${server_data_root}" ]]; then
    default_data_root="${server_data_root}"
else
    default_data_root="${DAHUA_CODE_ROOT}"
fi
export DAHUA_DATA_ROOT="${DAHUA_DATA_ROOT:-${default_data_root}}"
export DAHUA_VIS_RUNTIME_ROOT="${DAHUA_VIS_RUNTIME_ROOT:-${DAHUA_DATA_ROOT}/runtime/visualization}"
export DAHUA_VIS_HOST="${DAHUA_VIS_HOST:-0.0.0.0}"
export DAHUA_VIS_PORT="${DAHUA_VIS_PORT:-8000}"
# Person localization is an opt-in experiment.  The default pipeline feeds
# the original video directly to MediaPipe so installing NanoDet weights does
# not silently change pose-extraction behavior.
export DAHUA_NANODET_ENABLED="${DAHUA_NANODET_ENABLED:-0}"

server_web_python="/root/miniconda3/envs/skel_gcn38/bin/python"
server_llm_python="/workspace/code/envs/llm_env/bin/python"
server_rtmpose_python="/root/miniconda3/envs/rtmpose26/bin/python"

# The Web process and ProtoGCN require the legacy MMCV environment, while
# MediaPipe Tasks and Qwen require the newer Python environment.  Keep these
# boundaries explicit so launching the Web app from an arbitrary shell cannot
# accidentally run MediaPipe under Python 3.8.
if [[ -x "${server_web_python}" ]]; then
    export DAHUA_VIS_PYTHON="${DAHUA_VIS_PYTHON:-${server_web_python}}"
fi
if [[ -x "${server_llm_python}" ]]; then
    export DAHUA_TEACHER_PYTHON="${DAHUA_TEACHER_PYTHON:-${server_llm_python}}"
fi
if [[ -x "${server_rtmpose_python}" ]]; then
    export DAHUA_RTMPOSE_PYTHON="${DAHUA_RTMPOSE_PYTHON:-${server_rtmpose_python}}"
fi
# Campus6 is the deployment default: direct RTMPose COCO-17 matches the
# trained six-class checkpoint, unlike the legacy MediaPipe/NTU120 path.
export DAHUA_CAMPUS6_BACKEND="${DAHUA_CAMPUS6_BACKEND:-rtmpose17}"
export DAHUA_CAMPUS6_CONFIG="${DAHUA_CAMPUS6_CONFIG:-${DAHUA_CODE_ROOT}/gcn_models/ProtoGCN/configs/campus6/rtmpose26_k400_2d_gap_full.py}"
export DAHUA_CAMPUS6_CHECKPOINT="${DAHUA_CAMPUS6_CHECKPOINT:-${DAHUA_DATA_ROOT}/experiments/ProtoGCN/campus6_rtmpose26_k400_2d_gap_full_manual_v2/best_top1_acc_epoch_40.pth}"

lite_pose_model="${DAHUA_DATA_ROOT}/models/pose_models/mediapipe/pose_landmarker_lite.task"
if [[ -f "${lite_pose_model}" ]]; then
    # Lite + the 33 MiB ProtoGCN checkpoint fit the 50 MiB edge-package gate.
    # Operators can still opt into Heavy by setting DAHUA_MEDIAPIPE_MODEL.
    export DAHUA_MEDIAPIPE_MODEL="${DAHUA_MEDIAPIPE_MODEL:-${lite_pose_model}}"
fi

web_python="${DAHUA_VIS_PYTHON:-python}"
teacher_python="${DAHUA_TEACHER_PYTHON:-${web_python}}"
cd "${DAHUA_CODE_ROOT}"

"${web_python}" -c "import fastapi, mmcv, torch, uvicorn" >/dev/null
if [[ "${DAHUA_CAMPUS6_BACKEND}" == "rtmpose17" || "${DAHUA_CAMPUS6_BACKEND}" == "1" || "${DAHUA_CAMPUS6_BACKEND}" == "true" ]]; then
    "${DAHUA_RTMPOSE_PYTHON:-${web_python}}" -c "import cv2, mmpose, mmcv, torch" >/dev/null
    [[ -f "${DAHUA_CAMPUS6_CONFIG}" ]] || { printf 'ERROR: Campus6 config not found\n' >&2; exit 2; }
    [[ -f "${DAHUA_CAMPUS6_CHECKPOINT}" ]] || { printf 'ERROR: Campus6 checkpoint not found\n' >&2; exit 2; }
else
    pose_python="${DAHUA_POSE_PYTHON:-${web_python}}"
    "${pose_python}" -c "import mediapipe; from mediapipe.tasks import python" >/dev/null
fi
"${teacher_python}" -c "import torch, transformers" >/dev/null
exec "${web_python}" -m dahua_cup.backend.app
