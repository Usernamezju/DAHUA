#!/usr/bin/env bash
# One-shot server setup: GCN training environment plus server model assets.
#
# This script prepares the server side of the Campus6 delivery:
#
#   1. the ``skel_gcn38`` Conda environment (Python 3.8, PyTorch 1.10.2 /
#      CUDA 11.3, the frozen requirements/skel.txt set) used for GCN
#      training/regression and the Web service, and
#   2. the ``models-server.tar.gz`` release asset: the FP32 GCN
#      training/regression baseline and the GAP semantic embeddings.
#
# The command sequence for the Conda environment is the same as
# ``_provision_training_env`` in ``src/dahua_cup/backend/remote.py`` (the
# automatic rebuild path used by the Web UI) and must be kept in sync with it.
# An existing environment is never modified.
#
# Local-inference models (TensorRT FP16 pose engines, M1KD INT8 student) are
# NOT part of this package.  See the note printed at the end of the script and
# README ## Model download for how the server Web service keeps working
# without them.
#
# Usage: bash scripts/setup_server.sh [options]
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_server.sh [options]

Options:
  --tag TAG           GitHub release tag (default: models-v1.1.0)
  --repo OWNER/NAME   GitHub repository (default: Usernamezju/DAHUA)
  --work-dir DIR      Download scratch directory (default: ~/temp)
  --conda-root DIR    Conda installation root (default: /root/miniconda3)
  --env-name NAME     GCN training environment name (default: skel_gcn38)
  --provision-teacher Create a .venv-qwen teacher environment from
                      requirements/server.txt (default: off; Qwen model files
                      are never downloaded by this script)
  --skip-env          Skip environment provisioning
  --skip-models       Skip the model asset download
  --verify-files      After extraction, verify every file with
                      sha256sum -c models/SHA256SUMS.server
  -h, --help          Show this help.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tag="models-v1.1.0"
repo="Usernamezju/DAHUA"
work_dir="${HOME}/temp"
conda_root="/root/miniconda3"
env_name="skel_gcn38"
provision_teacher=0
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
    --env-name) env_name="$2"; shift 2 ;;
    --provision-teacher) provision_teacher=1; shift ;;
    --skip-env) skip_env=1; shift ;;
    --skip-models) skip_models=1; shift ;;
    --verify-files) verify_files=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Environment provisioning
# ---------------------------------------------------------------------------
if [[ "$skip_env" -eq 0 ]]; then
  conda_bin="$(command -v conda || true)"
  if [[ -z "$conda_bin" ]]; then
    conda_bin="${conda_root}/bin/conda"
  fi
  env_python="${conda_root}/envs/${env_name}/bin/python"

  if [[ -x "$env_python" ]]; then
    echo "Environment ${env_name} already present at ${env_python}; skipped."
  else
    echo "Creating ${env_name} (mirrors backend.remote._provision_training_env)"
    if [[ ! -x "$conda_bin" ]]; then
      echo "Conda not found at ${conda_bin}" >&2
      exit 2
    fi
    if [[ ! -f "${repo_root}/requirements/skel.txt" ]]; then
      echo "requirements/skel.txt missing; push the repository and re-run" >&2
      exit 2
    fi
    "$conda_bin" create -n "$env_name" python=3.8 -y
    "$conda_bin" install -n "$env_name" -y \
      -c pytorch pytorch=1.10.2 torchvision=0.11.3 torchaudio=0.10.2 cudatoolkit=11.3
    "$env_python" -m pip install --upgrade pip
    "$env_python" -m pip install -r "${repo_root}/requirements/skel.txt" \
      -f "$mmcv_index"
  fi

  if [[ "$provision_teacher" -eq 1 ]]; then
    teacher_venv="${repo_root}/.venv-qwen"
    if [[ -x "${teacher_venv}/bin/python" ]]; then
      echo "Teacher environment ${teacher_venv} already present; skipped."
    else
      echo "Creating ${teacher_venv} from requirements/server.txt"
      python3 -m venv "$teacher_venv"
      "${teacher_venv}/bin/python" -m pip install --upgrade pip
      "${teacher_venv}/bin/python" -m pip install -r "${repo_root}/requirements/server.txt"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# Model asset download
# ---------------------------------------------------------------------------
if [[ "$skip_models" -eq 0 ]]; then
  mkdir -p "$work_dir"
  asset="models-server.tar.gz"
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
    "models/MANIFEST.server.json" \
    "models/student/campus6_protogcn_gap_fp32_epoch40.pth" \
    "models/semantic/campus6_gap_clip_v3.npy"; do
    if [[ ! -f "${repo_root}/${required}" ]]; then
      echo "${required} missing after extraction" >&2
      missing=1
    fi
  done
  [[ "$missing" -eq 0 ]] || exit 1

  if [[ "$verify_files" -eq 1 ]]; then
    (cd "$repo_root" && sha256sum -c models/SHA256SUMS.server)
  fi
  echo "Server model assets ready in ${repo_root}/models"
fi

# ---------------------------------------------------------------------------
# Post-setup notes
# ---------------------------------------------------------------------------
cat <<EOF

Server setup finished. Notes:

- This package does NOT contain local-inference models (TensorRT FP16 pose
  engines, M1KD INT8 student).  Before starting scripts/run_visualization.sh
  on this server, point the Web service at the server-side runtime student
  checkpoint (config.py:517 fallback) and a pose backend available here, e.g.:

    export DAHUA_CAMPUS6_DEPLOYMENT_CHECKPOINT=/workspace/data/xzz_data/DAHUA/experiments/acceptance/campus6/m1kd_best_full/M1KD.runtime.int8.pt
    export DAHUA_POSE_BACKEND=<existing server pose backend>   # default tensorrt_fp16 needs engines

  Alternatively run scripts/setup_local.sh --skip-env to fetch the edge
  package on this machine as well.

- Qwen model files are intentionally not bundled.  Set DAHUA_QWEN_MODEL_DIR
  (default /workspace/data/public_data/Qwen3-VL-8B-Instruct) and configure
  the teacher environment via 系统设置 or --provision-teacher.
EOF
