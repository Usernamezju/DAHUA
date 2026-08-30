#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/code/DAHUA}"
GCN_ROOT="${GCN_ROOT:-$PROJECT_ROOT/gcn_models}"
REPO_ROOT="${REPO_ROOT:-/workspace/code/DAHUA}"
EXP_ROOT="${EXP_ROOT:-/workspace/data/xzz_data/DAHUA/experiments}"
OUT_DIR="${OUT_DIR:-/workspace/data/xzz_data/DAHUA/analysis/ntu120_longtail}"
MANIFEST="${MANIFEST:-$REPO_ROOT/analysis/default_manifest_server.json}"
DEVICE_ARGS="${DEVICE_ARGS:-0}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
DEFAULT_ENV_CMD="${DEFAULT_ENV_CMD:-source /workspace/code/envs/llm_env/bin/activate}"
AGCN_ENV_CMD="${AGCN_ENV_CMD:-$DEFAULT_ENV_CMD}"
CTR_ENV_CMD="${CTR_ENV_CMD:-source /root/miniconda3/bin/activate /root/miniconda3/envs/skel_gcn38}"
INFOGCN_ENV_CMD="${INFOGCN_ENV_CMD:-source /root/miniconda3/bin/activate /workspace/code/envs/info_gcn}"
ANALYSIS_ENV_CMD="${ANALYSIS_ENV_CMD:-}"

latest_file() {
  local pattern="$1"
  python - "$pattern" <<'PY'
import glob, os, sys
matches = glob.glob(sys.argv[1], recursive=True)
matches.sort(key=lambda p: (os.path.getmtime(p), p), reverse=True)
print(matches[0] if matches else "")
PY
}

activate_env() {
  local env_cmd="$1"
  if [[ -n "$env_cmd" ]]; then
    echo "[env] $env_cmd"
    eval "$env_cmd"
  fi
}

run_2sagcn_test() {
  local config="$1"
  local weights_pattern="$2"
  local work_dir="$3"
  local name="$4"
  local repo="$GCN_ROOT/2s-AGCN"
  local weights
  weights="$(latest_file "$weights_pattern")"
  if [[ -z "$weights" || ! -f "$repo/main.py" || ! -f "$config" ]]; then
    echo "[skip][2s-AGCN] $name"
    return 0
  fi
  mkdir -p "$work_dir"
  echo "[test][2s-AGCN] $name"
  (
    activate_env "$AGCN_ENV_CMD"
    cd "$repo"
    python main.py --config "$config" --phase test --save-score True --weights "$weights" --work-dir "$work_dir" --device $DEVICE_ARGS
  )
}

run_ctrgcn_test() {
  local config="$1"
  local weights_pattern="$2"
  local work_dir="$3"
  local name="$4"
  local repo="$GCN_ROOT/CTR-GCN"
  local weights
  weights="$(latest_file "$weights_pattern")"
  if [[ -z "$weights" || ! -f "$repo/main.py" || ! -f "$config" ]]; then
    echo "[skip][CTR-GCN] $name"
    return 0
  fi
  mkdir -p "$work_dir"
  echo "[test][CTR-GCN] $name"
  (
    activate_env "$CTR_ENV_CMD"
    cd "$repo"
    python main.py --config "$config" --phase test --save-score True --weights "$weights" --work-dir "$work_dir" --device $DEVICE_ARGS
  )
}

run_infogcn_test() {
  local config="$1"
  local weights_pattern="$2"
  local work_dir="$3"
  local name="$4"
  local repo="$GCN_ROOT/infogcn"
  local weights script
  weights="$(latest_file "$weights_pattern")"
  if [[ -z "$weights" || ! -d "$repo" || ! -f "$config" ]]; then
    echo "[skip][InfoGCN] $name"
    return 0
  fi
  script="${INFOGCN_TEST_SCRIPT:-}"
  if [[ -z "$script" ]]; then
    for candidate in "$repo/main.py" "$repo/test.py" "$repo/evaluate.py"; do
      [[ -f "$candidate" ]] && script="$candidate" && break
    done
  fi
  if [[ -z "$script" || ! -f "$script" ]]; then
    echo "[skip][InfoGCN] $name: set INFOGCN_TEST_SCRIPT=/path/to/script.py"
    return 0
  fi
  mkdir -p "$work_dir"
  echo "[test][InfoGCN] $name"
  (
    activate_env "$INFOGCN_ENV_CMD"
    cd "$repo"
    python "$script" --config "$config" --phase test --save-score True --weights "$weights" --work-dir "$work_dir" --device $DEVICE_ARGS
  )
}

if [[ "$RUN_INFERENCE" == "1" ]]; then
  run_2sagcn_test \
    "$GCN_ROOT/2s-AGCN/config/nturgbd-cross-subject/test_joint_longtail.yaml" \
    "$EXP_ROOT/2s-AGCN/runs/ntu120_xsub_longtail_joint*.pt" \
    "$EXP_ROOT/2s-AGCN/ntu120_xsub_longtail_joint" \
    "xsub joint"

  run_2sagcn_test \
    "$GCN_ROOT/2s-AGCN/config/nturgbd-cross-set/test_joint_longtail.yaml" \
    "$EXP_ROOT/2s-AGCN/runs/ntu120_xset_longtail_joint*.pt" \
    "$EXP_ROOT/2s-AGCN/ntu120_xset_longtail_joint" \
    "xset joint"

  run_ctrgcn_test \
    "$GCN_ROOT/CTR-GCN/config/nturgbd120-cross-subject/ctrgcn_ntu120_xsub_longtail_joint.yaml" \
    "$EXP_ROOT/CTR-GCN/runs/ntu120_xsub_longtail_joint*.pt" \
    "$EXP_ROOT/CTR-GCN/ntu120_xsub_longtail_joint" \
    "xsub joint"

  run_ctrgcn_test \
    "$GCN_ROOT/CTR-GCN/config/nturgbd120-cross-set/ctrgcn_ntu120_xset_longtail_joint.yaml" \
    "$EXP_ROOT/CTR-GCN/runs/ntu120_xset_longtail_joint*.pt" \
    "$EXP_ROOT/CTR-GCN/ntu120_xset_longtail_joint" \
    "xset joint"

  run_infogcn_test \
    "${INFOGCN_XSUB_CONFIG:-$GCN_ROOT/infogcn/config/nturgbd120-cross-subject/infogcn_ntu120_xsub_longtail_joint.yaml}" \
    "$EXP_ROOT/InfoGCN/runs/ntu120_NTU120_CSub_LT*.pt" \
    "$EXP_ROOT/InfoGCN/ntu120_NTU120_CSub_LT" \
    "xsub joint"

  run_infogcn_test \
    "${INFOGCN_XSET_CONFIG:-$GCN_ROOT/infogcn/config/nturgbd120-cross-set/infogcn_ntu120_xset_longtail_joint.yaml}" \
    "$EXP_ROOT/InfoGCN/runs/ntu120_NTU120_CSet_LT*.pt" \
    "$EXP_ROOT/InfoGCN/ntu120_NTU120_CSet_LT" \
    "xset joint"
fi

(
  activate_env "$ANALYSIS_ENV_CMD"
  python "$REPO_ROOT/analysis/generate_experiment_analysis.py" --manifest "$MANIFEST" --output-dir "$OUT_DIR"
)
echo "[done] analysis output: $OUT_DIR"
