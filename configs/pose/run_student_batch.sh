#!/usr/bin/env bash
set -euo pipefail

LIST="$1"
FEATURE_ROOT="$2"
OUT_ROOT="$3"
PYTHON_BIN="$4"
CHECKPOINT="$5"
CONFIG="$6"
LABEL_MAP="$7"
DEVICE="${8:-cuda:0}"

mkdir -p "$OUT_ROOT"
i=0
while IFS= read -r line; do
    label=$(printf '%s\n' "$line" | sed -n 's/.*"label_id": \([0-9]*\).*/\1/p')
    feature=$(printf '%s/features/sample_%02d_label%s_rtm_fp16.npz' "$FEATURE_ROOT" "$i" "$label")
    output=$(printf '%s/sample_%02d.json' "$OUT_ROOT" "$i")
    echo "[$i] $feature -> $output"
    env PYTHONPATH=/workspace/code/DAHUA \
        "$PYTHON_BIN" /workspace/code/DAHUA/dahua_cup/pipeline/rtmpose17_student_worker.py \
        --sample-id "sample_$(printf '%02d' "$i")" \
        --feature "$feature" \
        --output "$output" \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --checkpoint-format quantized \
        --label-map "$LABEL_MAP" \
        --device "$DEVICE"
    i=$((i + 1))
done < "$LIST"
echo "completed=$i"
