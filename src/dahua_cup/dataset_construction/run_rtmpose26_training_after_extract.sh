#!/usr/bin/env bash
# Poll the long RTMPose extraction every ten minutes and then run both planned
# Campus6 2D ProtoGCN transfer experiments.  It is intentionally resumable:
# all successful pose files are retained and a stopped extractor is relaunched
# at full source-frame density (never with the optional frame cap).
set -euo pipefail

ROOT=/workspace/code/DAHUA
DATA=/workspace/data/xzz_data/AVA_Kinetics_competition_audit_v1
MANIFEST="$DATA/candidate_manifests/rtmpose26_campus6_manifest_v1.jsonl"
FEATURES="$DATA/rtmpose26_features_v1"
ANN_DIR="$DATA/rtmpose26_coco17"
ANN="$ANN_DIR/annotations_protogcn_2d.pkl"
MODEL_DIR=/workspace/data/xzz_data/DAHUA/models/protogcn_pretrained/kinetics_skeleton_2d_joint
PRETRAINED="$MODEL_DIR/k400_j1_backbone_only.pth"
EXP_ROOT=/workspace/data/xzz_data/DAHUA/experiments/ProtoGCN
STATUS="$DATA/rtmpose26_training_automation.log"
RESULTS="$DATA/rtmpose26_training_results"
EXPECTED=$(grep -cve '^\s*$' "$MANIFEST")

mkdir -p "$ANN_DIR" "$RESULTS"
exec >>"$STATUS" 2>&1
echo "[$(date '+%F %T')] automation started: expected=$EXPECTED"

restart_extractor() {
  echo "[$(date '+%F %T')] extractor absent; restarting full-frame extraction"
  tmux new-session -d -s campus6_rtmpose26 \
    "export CUDA_VISIBLE_DEVICES=3; source /root/miniconda3/etc/profile.d/conda.sh && conda activate rtmpose26 && cd $ROOT && python -m dahua_cup.pipeline.extract_rtmpose26_batch --manifest $MANIFEST --output-dir $FEATURES --device cuda:0 --frame-stride 1 --max-frames 0 > $DATA/rtmpose26_full.log 2>&1"
}

while :; do
  READY=$(find "$FEATURES" -type f -name '*.npz' | wc -l)
  echo "[$(date '+%F %T')] pose files: $READY/$EXPECTED"
  if [ "$READY" -ge "$EXPECTED" ]; then
    break
  fi
  if ! pgrep -f 'dahua_cup.pipeline.extract_rtmpose26_batch' >/dev/null; then
    tmux has-session -t campus6_rtmpose26 2>/dev/null && tmux kill-session -t campus6_rtmpose26 || true
    restart_extractor
  fi
  sleep 600
done

echo "[$(date '+%F %T')] extraction complete; building grouped COCO-17 annotations"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate skel_gcn38
cd "$ROOT"
python dahua_cup/dataset_construction/build_ava_rtmpose26_protogcn2d_annotations.py \
  --manifest "$MANIFEST" --features "$FEATURES" --output "$ANN"

if [ ! -f "$PRETRAINED" ]; then
  python -c "import torch; src='$MODEL_DIR/best_top1_acc_epoch_150.pth'; dst='$PRETRAINED'; x=torch.load(src,map_location='cpu'); x['state_dict']={k:v for k,v in x['state_dict'].items() if not k.startswith('cls_head.')}; torch.save(x,dst)"
fi

run_experiment() {
  NAME=$1
  CONFIG=$2
  WORK="$EXP_ROOT/$NAME"
  if [ -e "$WORK/latest.pth" ] || [ -e "$WORK/best_top1_acc_epoch_1.pth" ]; then
    WORK="${WORK}_$(date '+%Y%m%d_%H%M%S')"
  fi
  LOG="$RESULTS/${NAME}.log"
  mkdir -p "$WORK"
  echo "[$(date '+%F %T')] starting $NAME" | tee "$LOG"
  CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29631 \
    gcn_models/ProtoGCN/tools/train.py "$CONFIG" \
    --work-dir "$WORK" --ann-file "$ANN" --seed 20260827 --deterministic \
    --validate --test-last --test-best >>"$LOG" 2>&1
  BEST=$(find "$WORK" -maxdepth 1 -type f -name 'best_top1_acc_epoch_*.pth' -printf '%f\n' | sort -V | tail -1)
  if [ -z "$BEST" ]; then
    echo "[$(date '+%F %T')] $NAME has no best checkpoint" | tee -a "$LOG"
    return 1
  fi
  # train.py already runs the held-out test for its selected best checkpoint
  # and writes best_pred.pkl.  Reusing it avoids a second test.py invocation
  # whose direct-script import path is not initialised in this codebase.
  if [ ! -f "$WORK/best_pred.pkl" ]; then
    echo "[$(date '+%F %T')] $NAME did not produce best_pred.pkl" | tee -a "$LOG"
    return 1
  fi
  python dahua_cup/dataset_construction/summarize_campus6_predictions.py \
    --annotations "$ANN" --predictions "$WORK/best_pred.pkl" \
    --split test --output "$RESULTS/${NAME}_test_metrics.json" | tee -a "$LOG"
  echo "[$(date '+%F %T')] completed $NAME using $BEST" | tee -a "$LOG"
}

run_experiment campus6_rtmpose26_k400_2d_head_only \
  gcn_models/ProtoGCN/configs/campus6/rtmpose26_k400_2d_head_only.py
run_experiment campus6_rtmpose26_k400_2d_full \
  gcn_models/ProtoGCN/configs/campus6/rtmpose26_k400_2d_full.py
echo "[$(date '+%F %T')] all requested experiments completed"
