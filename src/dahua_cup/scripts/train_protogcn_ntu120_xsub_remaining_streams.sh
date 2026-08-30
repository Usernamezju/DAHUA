#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PROTOGCN_ROOT="${PROTOGCN_ROOT:-$PROJECT_ROOT/gcn_models/ProtoGCN}"
ANN_FILE="${ANN_FILE:-/workspace/data/xzz_data/DAHUA/datasets/NTU/ProtoGCN/ntu120_3danno.pkl}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/workspace/data/xzz_data/DAHUA/experiments/ProtoGCN}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-48}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.1}"
EPOCHS="${EPOCHS:-150}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
WANDB_PROJECT="${WANDB_PROJECT:-dahua-protogcn}"
WANDB_MODE="${WANDB_MODE:-offline}"

modalities=(b k jm bm km)
stream_names=(bone kbone joint-motion bone-motion kbone-motion)
run_dirs=(
  ntu120_xsub_bone
  ntu120_xsub_kbone
  ntu120_xsub_joint_motion
  ntu120_xsub_bone_motion
  ntu120_xsub_kbone_motion
)

if [[ ! -d "$PROTOGCN_ROOT" ]]; then
  echo "ProtoGCN directory not found: $PROTOGCN_ROOT" >&2
  exit 1
fi

if [[ ! -f "$ANN_FILE" ]]; then
  echo "Annotation file not found: $ANN_FILE" >&2
  exit 1
fi

IFS=',' read -r -a visible_gpus <<< "$GPU_IDS"
if [[ "${#visible_gpus[@]}" -ne "$NUM_GPUS" ]]; then
  echo "GPU_IDS contains ${#visible_gpus[@]} devices, but NUM_GPUS=$NUM_GPUS" >&2
  exit 1
fi

source /root/miniconda3/bin/activate /root/miniconda3/envs/skel_gcn38
cd "$PROTOGCN_ROOT"

for index in "${!modalities[@]}"; do
  modality="${modalities[$index]}"
  stream_name="${stream_names[$index]}"
  run_dir="$EXPERIMENT_ROOT/${run_dirs[$index]}"
  config="configs/ntu120_xsub/${modality}.py"
  complete_marker="$run_dir/.training_complete"

  if [[ -f "$complete_marker" ]]; then
    echo "Skipping completed stream: $stream_name"
    continue
  fi

  mkdir -p "$run_dir" "$run_dir/wandb"

  echo "Starting ProtoGCN stream: $stream_name"
  echo "Config: $config"
  echo "Work directory: $run_dir"

  CUDA_VISIBLE_DEVICES="$GPU_IDS" \
  WANDB_DIR="$run_dir/wandb" \
  bash tools/dist_train.sh \
    "$config" \
    "$NUM_GPUS" \
    --validate \
    --test-last \
    --test-best \
    --ann-file "$ANN_FILE" \
    --work-dir "$run_dir" \
    --batch-size-per-gpu "$BATCH_SIZE_PER_GPU" \
    --workers-per-gpu "$WORKERS_PER_GPU" \
    --learning-rate "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    --log-interval "$LOG_INTERVAL" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "ntu120-3d-xsub-$stream_name" \
    --wandb-mode "$WANDB_MODE" \
    2>&1 | tee -a "$run_dir/train_console.log"

  touch "$complete_marker"
  echo "Completed ProtoGCN stream: $stream_name"
done

echo "All remaining NTU120 XSub streams completed."
