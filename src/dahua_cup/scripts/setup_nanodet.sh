#!/usr/bin/env bash
set -Eeuo pipefail

# Install the official NanoDet inference package and download the matching
# legacy NanoDet-m COCO checkpoint. NanoDet-m-416 is the default because it has
# the same parameter count as NanoDet-m-320 and is better suited to small people.
# Only runtime dependencies are installed: the official requirements also pull
# ONNX export packages that are unnecessary for PyTorch inference and may need a
# local CMake toolchain on Python 3.8.

DATA_ROOT="${DAHUA_DATA_ROOT:-/workspace/data/xzz_data/DAHUA}"
MODEL_ROOT="${DAHUA_MODEL_ROOT:-${DATA_ROOT}/models}"
SOURCE_ROOT="${DAHUA_THIRD_PARTY_ROOT:-${DATA_ROOT}/third_party}"
MODEL_ROOT_EXPLICIT=0
SOURCE_ROOT_EXPLICIT=0
PYTHON_BIN="${DAHUA_VIS_PYTHON:-python}"
VARIANT="nanodet-m-416"
NANODET_REF="v1.0.0"
NO_PROXY_MODE=0
SKIP_INSTALL=0
DRY_RUN=0

usage() {
    cat <<EOF
Usage: bash dahua_cup/scripts/setup_nanodet.sh [options]

Download the official NanoDet source, install its Python dependencies and
download a matching legacy NanoDet-m COCO checkpoint.

Options:
  --variant NAME       nanodet-m-416 (default) or nanodet-m
  --data-root PATH     Data root (default: ${DATA_ROOT})
  --model-root PATH    Model root (default: DATA_ROOT/models)
  --source-root PATH   Third-party source root (default: DATA_ROOT/third_party)
  --python PATH        Python executable (default: ${PYTHON_BIN})
  --skip-install       Download files without changing the Python environment
  --no-proxy           Ignore HTTP(S)/ALL proxy variables for this run
  --dry-run            Print actions without changing files
  -h, --help           Show this help
EOF
}

while (($#)); do
    case "$1" in
        --variant)
            [[ $# -ge 2 ]] || { printf 'ERROR: --variant needs a value\n' >&2; exit 2; }
            VARIANT="$2"
            shift 2
            ;;
        --data-root)
            [[ $# -ge 2 ]] || { printf 'ERROR: --data-root needs a path\n' >&2; exit 2; }
            DATA_ROOT="$2"
            shift 2
            ;;
        --model-root)
            [[ $# -ge 2 ]] || { printf 'ERROR: --model-root needs a path\n' >&2; exit 2; }
            MODEL_ROOT="$2"
            MODEL_ROOT_EXPLICIT=1
            shift 2
            ;;
        --source-root)
            [[ $# -ge 2 ]] || { printf 'ERROR: --source-root needs a path\n' >&2; exit 2; }
            SOURCE_ROOT="$2"
            SOURCE_ROOT_EXPLICIT=1
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || { printf 'ERROR: --python needs a path\n' >&2; exit 2; }
            PYTHON_BIN="$2"
            shift 2
            ;;
        --skip-install)
            SKIP_INSTALL=1
            shift
            ;;
        --no-proxy)
            NO_PROXY_MODE=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'ERROR: unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ((MODEL_ROOT_EXPLICIT == 0)); then
    MODEL_ROOT="${DATA_ROOT}/models"
fi
if ((SOURCE_ROOT_EXPLICIT == 0)); then
    SOURCE_ROOT="${DATA_ROOT}/third_party"
fi

case "${VARIANT}" in
    nanodet-m-416)
        CONFIG_NAME="nanodet-m-416.yml"
        CHECKPOINT_NAME="nanodet_m_416.ckpt"
        GOOGLE_DRIVE_ID="1jY-Um2VDDEhuVhluP9lE70rG83eXQYhV"
        ;;
    nanodet-m)
        CONFIG_NAME="nanodet-m.yml"
        CHECKPOINT_NAME="nanodet_m.ckpt"
        GOOGLE_DRIVE_ID="1ZkYucuLusJrCb_i63Lid0kYyyLvEiGN3"
        ;;
    *)
        printf 'ERROR: unsupported variant: %s\n' "${VARIANT}" >&2
        printf 'Choose nanodet-m-416 or nanodet-m.\n' >&2
        exit 2
        ;;
esac

if ((NO_PROXY_MODE)); then
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
code_root="$(cd "${script_dir}/../.." && pwd)"
nanodet_source="${SOURCE_ROOT}/nanodet-${NANODET_REF}"
nanodet_model_dir="${MODEL_ROOT}/nanodet"
source_config="${nanodet_source}/config/legacy_v0.x_configs/${CONFIG_NAME}"
target_config="${nanodet_model_dir}/${CONFIG_NAME}"
target_checkpoint="${nanodet_model_dir}/${CHECKPOINT_NAME}"
drive_url="https://drive.google.com/file/d/${GOOGLE_DRIVE_ID}/view?usp=sharing"

log() {
    printf '[nanodet-setup] %s\n' "$*"
}

die() {
    printf '[nanodet-setup] ERROR: %s\n' "$*" >&2
    exit 1
}

run() {
    if ((DRY_RUN)); then
        printf '[dry-run]'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

if [[ "${PYTHON_BIN}" == */* ]]; then
    [[ -x "${PYTHON_BIN}" ]] || die "Python executable is unavailable: ${PYTHON_BIN}"
else
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 \
        || die "Python executable is unavailable: ${PYTHON_BIN}"
fi
command -v git >/dev/null 2>&1 || die "git is required"

run mkdir -p "${SOURCE_ROOT}" "${nanodet_model_dir}"

if [[ -d "${nanodet_source}/.git" ]]; then
    log "reuse official source: ${nanodet_source}"
elif [[ -e "${nanodet_source}" ]]; then
    die "source path exists but is not a Git checkout: ${nanodet_source}"
else
    log "clone official NanoDet ${NANODET_REF}"
    run git clone \
        --branch "${NANODET_REF}" \
        --depth 1 \
        https://github.com/RangiLyu/nanodet.git \
        "${nanodet_source}"
fi

if ((DRY_RUN)); then
    log "would copy config to ${target_config}"
elif [[ -f "${source_config}" ]]; then
    install -m 0644 "${source_config}" "${target_config}"
else
    die "matching config is absent from the official checkout: ${source_config}"
fi

if ((!SKIP_INSTALL)); then
    log "check existing PyTorch, TorchVision, NumPy and OpenCV runtime"
    run "${PYTHON_BIN}" -c 'import cv2, numpy, torch, torchvision'
    log "install NanoDet inference-only dependencies into ${PYTHON_BIN}"
    run "${PYTHON_BIN}" -m pip install \
        'omegaconf>=2.0.1,<2.4' \
        'pyaml>=20.4' \
        'pycocotools>=2.0.2' \
        'pytorch-lightning==1.9.5' \
        'torchmetrics>=0.11,<1.6' \
        'termcolor>=1.1' \
        'tabulate>=0.8' \
        'gdown>=4.7,<6'
    run "${PYTHON_BIN}" -m pip install \
        --editable "${nanodet_source}" \
        --no-deps
fi

if ((DRY_RUN)); then
    log "would download ${drive_url} to ${target_checkpoint}"
elif [[ -s "${target_checkpoint}" ]]; then
    log "reuse checkpoint: ${target_checkpoint}"
else
    "${PYTHON_BIN}" -c 'import gdown' >/dev/null 2>&1 \
        || die "gdown is unavailable; remove --skip-install or install gdown"
    checkpoint_partial="${target_checkpoint}.part"
    log "download official ${VARIANT} checkpoint"
    "${PYTHON_BIN}" -m gdown \
        --continue \
        --fuzzy \
        --output "${checkpoint_partial}" \
        "${drive_url}"
    [[ -s "${checkpoint_partial}" ]] \
        || die "downloaded checkpoint is empty: ${checkpoint_partial}"
    mv "${checkpoint_partial}" "${target_checkpoint}"
fi

if ((DRY_RUN)); then
    log "would load the downloaded model on CPU for validation"
else
    log "validate package, configuration and checkpoint on CPU"
    PYTHONPATH="${code_root}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON_BIN}" - "${target_config}" "${target_checkpoint}" <<'PY'
import sys

import torch

from dahua_cup.person_localization.nanodet_backend import NanoDetPersonDetector

config_path, checkpoint_path = sys.argv[1:]
detector = NanoDetPersonDetector(
    config=config_path,
    checkpoint=checkpoint_path,
    device="cpu",
)
parameter_count = sum(parameter.numel() for parameter in detector.model.parameters())
print(f"[nanodet-setup] torch={torch.__version__}")
print(f"[nanodet-setup] classes={len(detector.class_names)} person_id={detector.person_class_id}")
print(f"[nanodet-setup] parameters={parameter_count:,}")
PY
fi

log "complete"
log "config: ${target_config}"
log "checkpoint: ${target_checkpoint}"
log "python: ${PYTHON_BIN}"
log "Web launch:"
printf 'bash dahua_cup/scripts/run_visualization.sh \\\n'
printf '  --enable-nanodet \\\n'
printf '  --nanodet-config %q \\\n' "${target_config}"
printf '  --nanodet-checkpoint %q \\\n' "${target_checkpoint}"
printf '  --nanodet-python %q \\\n' "${PYTHON_BIN}"
printf '  --nanodet-device cpu\n'
