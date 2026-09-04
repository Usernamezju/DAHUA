#!/usr/bin/env bash
set -euo pipefail

LIST="$1"
OUT_ROOT="$2"
PYTHON_BIN="$3"
DETECTOR_ROOT="$4"
POSE_ROOT="$5"
MAX_FRAMES="${6:-100}"

mkdir -p "$OUT_ROOT/features"
i=0
while IFS= read -r line; do
    video=$(printf '%s\n' "$line" | sed -n 's/.*"video": "\([^"]*\)".*/\1/p')
    label=$(printf '%s\n' "$line" | sed -n 's/.*"label_id": \([0-9]*\).*/\1/p')
    out=$(printf '%s/features/sample_%02d_label%s_rtm_fp16.npz' "$OUT_ROOT" "$i" "$label")
    echo "[$i] $video -> $out"
    env PYTHONPATH=/workspace/code/DAHUA \
        LD_PRELOAD=/root/miniconda3/envs/rtmpose26/lib/libiomp5.so \
        LD_LIBRARY_PATH=/workspace/code/envs/rtmpose_int8/lib/python3.8/site-packages/tensorrt_libs:/workspace/code/envs/rtmpose_int8/lib \
        "$PYTHON_BIN" /workspace/code/DAHUA/dahua_cup/pipeline/rtmpose17_pose_worker.py \
        --video "$video" \
        --feature "$out" \
        --device cuda:0 \
        --max-frames "$MAX_FRAMES" \
        --bbox-score 0.15 \
        --joint-score-threshold 0.20 \
        --extractor-id rtm-nano-s-tensorrt-fp16.v1 \
        --backend mmdeploy \
        --detector-model-dir "$DETECTOR_ROOT" \
        --pose-model-dir "$POSE_ROOT"
    i=$((i + 1))
done < "$LIST"
echo "completed=$i"
