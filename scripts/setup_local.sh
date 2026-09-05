#!/usr/bin/env bash
# One-shot local setup: role environments plus edge model assets.
#
# This script prepares the local (inference) side of the Campus6 delivery:
#
#   1. the role environments used by scripts/run_local_visualization.sh:
#        - Web + RTMDet/RTMPose (MMCV 2.x stack, Python 3.10),
#        - ProtoGCN student inference (MMCV 1.5.0 stack, Python 3.8),
#        - optional TensorRT FP16 pose environment (MMDeploy stack,
#          GPU only, --provision-pose-fp16), and
#   2. the ``models-edge.tar.gz`` release asset: the TensorRT FP16 pose
#      engines under models/pose/fp16/ and the M1KD INT8 student checkpoint.
#
# The Qwen teacher environment is deliberately not provisioned here: Qwen
# stays on the configured remote server and is only called after the local
# student uncertainty gate fires.
#
# Installation commands mirror the from-scratch CPU deployment test
# (environment names dahua_test_web / dahua_test_gcn there); requirements/
# local.txt is not used because it mixes the ProtoGCN MMCV 1.5.0 chain with
# the MMCV 2.x Web/pose stack in one environment.
#
# Usage: bash scripts/setup_local.sh [options]
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_local.sh [options]

Options:
  --tag TAG           GitHub release tag (default: models-v1.1.0)
  --repo OWNER/NAME   GitHub repository (default: Usernamezju/DAHUA)
  --work-dir DIR      Download scratch directory (default: ~/temp)
  --conda-root DIR    Conda installation root (default: ~/miniconda3)
  --web-env NAME      Web + RTMDet/RTMPose environment (default: dahua_web)
  --gcn-env NAME      ProtoGCN student environment (default: dahua_gcn)
  --pose-env NAME     TensorRT FP16 pose environment (default: dahua_pose_fp16)
  --device DEVICE     cpu or gpu; selects the torch install source for the
                      web and gcn environments (default: cpu)
  --torch-index URL   PyTorch pip index for the web environment in gpu mode
                      (default: https://download.pytorch.org/whl/cu121)
  --provision-pose-fp16
                      Also create the GPU-only TensorRT FP16 pose environment
                      (default: off; TensorRT comes from the NVIDIA/CUDA
                      distribution, not pip)
  --skip-env          Skip environment provisioning
  --skip-models       Skip the model asset download
  --verify-files      After extraction, verify every file with
                      sha256sum -c models/SHA256SUMS.edge
  -h, --help          Show this help.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tag="models-v1.1.0"
repo="Usernamezju/DAHUA"
work_dir="${HOME}/temp"
conda_root="${HOME}/miniconda3"
web_env="dahua_web"
gcn_env="dahua_gcn"
pose_env="dahua_pose_fp16"
device="cpu"
torch_index="https://download.pytorch.org/whl/cu121"
provision_pose_fp16=0
skip_env=0
skip_models=0
verify_files=0
mmcv_index="https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html"

while (($#)); do
  case "$1" in
    --tag) tag="$2"; shift 2 ;;
    --repo) repo="$2"; shift 2 ;;
    --work-dir) work_dir="$2"; shift 2 ;;
    --conda-root) conda_root="$2"; shift 2 ;;
    --web-env) web_env="$2"; shift 2 ;;
    --gcn-env) gcn_env="$2"; shift 2 ;;
    --pose-env) pose_env="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --torch-index) torch_index="$2"; shift 2 ;;
    --provision-pose-fp16) provision_pose_fp16=1; shift ;;
    --skip-env) skip_env=1; shift ;;
    --skip-models) skip_models=1; shift ;;
    --verify-files) verify_files=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$device" != "cpu" && "$device" != "gpu" ]]; then
  echo "Invalid --device: ${device} (expected cpu or gpu)" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Environment provisioning
# ---------------------------------------------------------------------------
if [[ "$skip_env" -eq 0 ]]; then
  conda_bin="$(command -v conda || true)"
  if [[ -z "$conda_bin" ]]; then
    conda_bin="${conda_root}/bin/conda"
  fi
  if [[ ! -x "$conda_bin" ]]; then
    echo "Conda not found at ${conda_bin}" >&2
    exit 2
  fi

  # --- Web + RTMDet/RTMPose (MMCV 2.x stack) -------------------------------
  web_python="${conda_root}/envs/${web_env}/bin/python"
  if [[ -x "$web_python" ]]; then
    echo "Environment ${web_env} already present at ${web_python}; skipped."
  else
    echo "Creating ${web_env} (Python 3.10, MMCV 2.x stack)"
    "$conda_bin" create -n "$web_env" python=3.10 -y
    if [[ "$device" == "cpu" ]]; then
      "$web_python" -m pip install torch==2.1.2 torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cpu
      mmcv_prefix="https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html"
    else
      "$web_python" -m pip install torch==2.1.2 torchvision==0.16.2 \
        --index-url "$torch_index"
      mmcv_prefix="https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html"
    fi
    "$web_python" -m pip install mmcv==2.1.0 -f "$mmcv_prefix"
    "$web_python" -m pip install \
      mmengine==0.10.5 mmdet==3.3.0 mmpose==1.3.2 \
      opencv-python fastapi uvicorn python-multipart
    "$web_python" -m pip install -e "${repo_root}[web,test]"
  fi

  # --- ProtoGCN student (MMCV 1.5.0 stack) ---------------------------------
  gcn_python="${conda_root}/envs/${gcn_env}/bin/python"
  if [[ -x "$gcn_python" ]]; then
    echo "Environment ${gcn_env} already present at ${gcn_python}; skipped."
  else
    echo "Creating ${gcn_env} (Python 3.8, MMCV 1.5.0 stack)"
    "$conda_bin" create -n "$gcn_env" python=3.8 -y
    if [[ "$device" == "cpu" ]]; then
      "$gcn_python" -m pip install torch==1.10.2+cpu torchvision==0.11.3+cpu \
        -f https://download.pytorch.org/whl/torch_stable.html
    else
      # Same PyTorch/CUDA Conda build as the server skel_gcn38 environment.
      "$conda_bin" install -n "$gcn_env" -y \
        -c pytorch pytorch=1.10.2 torchvision=0.11.3 torchaudio=0.10.2 cudatoolkit=11.3
    fi
    "$gcn_python" -m pip install -r "${repo_root}/requirements/skel.txt" \
      -f "$mmcv_index"
    "$gcn_python" -m pip install python-multipart
  fi

  # --- TensorRT FP16 pose (optional, GPU only) -----------------------------
  if [[ "$provision_pose_fp16" -eq 1 ]]; then
    pose_python="${conda_root}/envs/${pose_env}/bin/python"
    if [[ -x "$pose_python" ]]; then
      echo "Environment ${pose_env} already present at ${pose_python}; skipped."
    else
      echo "Creating ${pose_env} (MMDeploy 1.3.1 stack)"
      "$conda_bin" create -n "$pose_env" python=3.8 -y
      "$pose_python" -m pip install -r "${repo_root}/requirements/pose_fp16.txt"
    fi
    echo "Note: TensorRT 8.6.1 and torch 1.10.2 must come from the NVIDIA/CUDA"
    echo "distribution; pip must not replace them.  Source"
    echo "configs/pose/rtmpose_fp16.env.example before starting the worker."
  fi
fi

# ---------------------------------------------------------------------------
# Model asset download
# ---------------------------------------------------------------------------
if [[ "$skip_models" -eq 0 ]]; then
  mkdir -p "$work_dir"
  asset="models-edge.tar.gz"
  base_url="https://github.com/${repo}/releases/download/${tag}"
  archive="${work_dir}/${asset}"
  digest_file="${work_dir}/${asset}.sha256"

  echo "Downloading ${base_url}/${asset}"
  curl -fL --retry 3 -o "${archive}.part" "${base_url}/${asset}"
  mv "${archive}.part" "$archive"

  echo "Downloading ${base_url}/${asset}.sha256"
  curl -fL --retry 3 -o "${digest_file}.part" "${base_url}/${asset}.sha256"
  mv "${digest_file}.part" "$digest_file"

  expected="$(awk '{print $1}' "$digest_file")"
  actual="$(sha256sum "$archive" | awk '{print $1}')"
  if [[ "$expected" != "$actual" ]]; then
    echo "SHA-256 mismatch:" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    exit 1
  fi
  echo "SHA-256 verified: $actual"

  echo "Unpacking into ${repo_root}"
  tar -xzf "$archive" -C "$repo_root"

  missing=0
  for required in \
    "models/MANIFEST.edge.json" \
    "models/student/M1KD.int8.pt" \
    "models/student/M1KD.int8.pt.json" \
    "models/pose/fp16/rtmdet_nano_person/end2end.engine" \
    "models/pose/fp16/rtmpose_s/end2end.engine"; do
    if [[ ! -f "${repo_root}/${required}" ]]; then
      echo "${required} missing after extraction" >&2
      missing=1
    fi
  done
  [[ "$missing" -eq 0 ]] || exit 1

  if [[ "$verify_files" -eq 1 ]]; then
    (cd "$repo_root" && sha256sum -c models/SHA256SUMS.edge)
  fi
  echo "Edge model assets ready in ${repo_root}/models"
fi

# ---------------------------------------------------------------------------
# Post-setup notes
# ---------------------------------------------------------------------------
cat <<EOF

Local setup finished. Notes:

- Point scripts/run_local_visualization.sh at the created environments:

    bash scripts/run_local_visualization.sh \\
      --web-python ${conda_root}/envs/${web_env}/bin/python \\
      --rtmpose-python ${conda_root}/envs/${web_env}/bin/python \\
      --student-python ${conda_root}/envs/${gcn_env}/bin/python \\
      --device ${device}

- TensorRT FP16 backend: requires the pose environment created with
  --provision-pose-fp16, a CUDA-visible GPU, and
  --rtmpose-python ${conda_root}/envs/${pose_env}/bin/python.
- CPU diagnostic backend (DAHUA_POSE_BACKEND=mmpose_fp32) needs no packaged
  weights; MMPose pulls its default weights online, or download the FP32
  rtmdet-s/rtmpose-s checkpoints manually from the source_url recorded in
  models/MANIFEST.json.
- The Qwen teacher is not a local dependency; configure the remote Qwen SSH
  connection once in 系统设置.
EOF
