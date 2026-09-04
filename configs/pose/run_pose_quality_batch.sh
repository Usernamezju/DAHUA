#!/usr/bin/env bash
set -euo pipefail

MODE="$1"
LIST="$2"
OUT_ROOT="$3"
PYTHON_BIN="$4"
POSE_ROOT="$5"
NANO_CHECKPOINT="$6"
DET_CONFIG="$7"
POSE_CHECKPOINT="$8"

mkdir -p "$OUT_ROOT/predictions"
i=0
while IFS= read -r line; do
    video=$(printf '%s\n' "$line" | sed -n 's/.*"video": "\([^"]*\)".*/\1/p')
    out=$(printf '%s/predictions/%s_%02d.json' "$OUT_ROOT" "$MODE" "$i")
    echo "[$i] $video"
    if [ "$MODE" = "fp16" ]; then
        env LD_PRELOAD=/root/miniconda3/envs/rtmpose26/lib/libiomp5.so \
            LD_LIBRARY_PATH=/workspace/code/envs/rtmpose_int8/lib/python3.8/site-packages/tensorrt_libs:/workspace/code/envs/rtmpose_int8/lib \
            "$PYTHON_BIN" /workspace/code/DAHUA/.runtime/verify_int8_frames.py \
            --video "$video" \
            --detector-model-dir "$POSE_ROOT/rtmdet_nano_person" \
            --pose-model-dir "$POSE_ROOT/rtmpose_s" \
            --output "$out" --device cuda:0 --max-frames 30 --score 0.15 \
            --backend-label tensorrt_fp16
    else
        env LD_PRELOAD=/root/miniconda3/envs/rtmpose26/lib/libiomp5.so \
            LD_LIBRARY_PATH=/workspace/code/envs/rtmpose_int8/lib/python3.8/site-packages/tensorrt_libs:/workspace/code/envs/rtmpose_int8/lib \
            "$PYTHON_BIN" /workspace/code/DAHUA/.runtime/verify_fp32_frames.py \
            --video "$video" \
            --pose2d-weights "$POSE_CHECKPOINT" \
            --det-weights "$NANO_CHECKPOINT" \
            --det-config "$DET_CONFIG" \
            --output "$out" --device cuda:0 --max-frames 30 --score 0.15
    fi
    i=$((i + 1))
done < "$LIST"
echo "completed=$i"
