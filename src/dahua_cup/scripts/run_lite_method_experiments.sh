#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
GCN_ROOT="${GCN_ROOT:-$PROJECT_ROOT/gcn_models}"
METHOD="${1:-}"
WANDB_ENTITY_ARG="${WANDB_ENTITY_ARG:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
DEVICE_ARGS="${DEVICE_ARGS:-0 1 2 3}"

if [[ -z "$METHOD" ]]; then
  echo "Usage: $0 {reweight|decouple|gap_lite}"
  echo "Optional env: WANDB_ENTITY_ARG=3250105852-zhejiang-university GPU_IDS=0,1,2,3 DEVICE_ARGS='0 1 2 3'"
  exit 1
fi

WANDB_ARGS=()
if [[ -n "$WANDB_ENTITY_ARG" ]]; then
  WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY_ARG")
fi

case "$METHOD" in
  reweight)
    cd "$GCN_ROOT/CTR-GCN"
    source /root/miniconda3/bin/activate /root/miniconda3/envs/skel_gcn38
    CUDA_VISIBLE_DEVICES="$GPU_IDS" python main.py \
      --config config/nturgbd120-cross-subject/ctrgcn_ntu120_xsub_lite_reweight_joint.yaml \
      --device $DEVICE_ARGS \
      "${WANDB_ARGS[@]}"
    ;;
  decouple)
    cd "$GCN_ROOT/CTR-GCN"
    source /root/miniconda3/bin/activate /root/miniconda3/envs/skel_gcn38
    CUDA_VISIBLE_DEVICES="$GPU_IDS" python main.py \
      --config config/nturgbd120-cross-subject/ctrgcn_ntu120_xsub_lite_decouple_joint.yaml \
      --device $DEVICE_ARGS \
      "${WANDB_ARGS[@]}"
    ;;
  gap_lite)
    cd "$GCN_ROOT/GAP"
    source /root/miniconda3/bin/activate /root/miniconda3/envs/skel_gcn38
    CUDA_VISIBLE_DEVICES="$GPU_IDS" python main_multipart_ntu.py \
      --config config/nturgbd120-cross-subject/lst_joint_lite_eval_val.yaml \
      --device $DEVICE_ARGS \
      "${WANDB_ARGS[@]}"
    ;;
  *)
    echo "Unknown method: $METHOD"
    echo "Expected one of: reweight, decouple, gap_lite"
    exit 1
    ;;
esac
