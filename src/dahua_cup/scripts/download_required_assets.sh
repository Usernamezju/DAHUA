#!/usr/bin/env bash
set -Eeuo pipefail

# Download the external assets required by the DAHUA campus pipeline.
# MediaPipe and ProtoGCN weights are managed separately because multiple model
# sizes/checkpoints are supported. The default set excludes RGB videos and Qwen.

INCLUDE_QWEN=0
INCLUDE_MEDIAPIPE_LITE=0
DRY_RUN=0
NO_PROXY_MODE=0

DATA_ROOT="${DAHUA_DATA_ROOT:-/workspace/data/xzz_data/DAHUA}"
MODEL_ROOT="${DAHUA_MODEL_ROOT:-${DATA_ROOT}/models}"
MODEL_ROOT_EXPLICIT=0

NTU120_2D_URL="https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu120_2d.pkl"
QWEN_REPO="Qwen/Qwen3-VL-8B-Instruct"
MEDIAPIPE_LITE_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

usage() {
  printf '%s\n' \
    "Usage: bash dahua_cup/scripts/download_required_assets.sh [options]" \
    "" \
    "Default downloads:" \
    "  - NTU120 2D skeleton annotations (no RGB videos)" \
    "" \
    "Options:" \
    "  --data-root PATH    Data root (default: ${DATA_ROOT})" \
    "  --model-root PATH   Downloaded-model root (default: DATA_ROOT/models)" \
    "  --include-qwen      Also download Qwen3-VL-8B-Instruct (~18 GB)" \
    "  --include-mediapipe-lite" \
    "                      Download compact pose weights for the edge bundle" \
    "  --no-proxy          Ignore HTTP(S)/ALL proxy variables for this run" \
    "  --dry-run           Print actions without downloading" \
    "  -h, --help          Show this help"
}

while (($#)); do
  case "$1" in
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
    --include-qwen)
      INCLUDE_QWEN=1
      shift
      ;;
    --include-mediapipe-lite)
      INCLUDE_MEDIAPIPE_LITE=1
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
  MODEL_ROOT="${DAHUA_MODEL_ROOT:-${DATA_ROOT}/models}"
fi

if ((NO_PROXY_MODE)); then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true
fi

NTU_DIR="${DATA_ROOT}/datasets/ntu120_2d"
QWEN_DIR="${MODEL_ROOT}/Qwen3-VL-8B-Instruct"
MEDIAPIPE_DIR="${MODEL_ROOT}/pose_models/mediapipe"
CHECKSUM_DIR="${MODEL_ROOT}/checksums"
CHECKSUM_FILE="${CHECKSUM_DIR}/required_assets.sha256"

log() {
  printf '[assets] %s\n' "$*"
}

die() {
  printf '[assets] ERROR: %s\n' "$*" >&2
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

detect_downloader() {
  if command -v curl >/dev/null 2>&1; then
    DOWNLOADER="curl"
  elif command -v wget >/dev/null 2>&1; then
    DOWNLOADER="wget"
  else
    die "curl or wget is required"
  fi
}

download_file() {
  local url="$1"
  local destination="$2"
  local label="$3"
  local partial="${destination}.part"

  if [[ -s "$destination" ]]; then
    log "skip ${label}: ${destination} already exists"
    return
  fi
  run mkdir -p "$(dirname "$destination")"
  log "download ${label}"
  if ((DRY_RUN)); then
    log "source: ${url}"
    return
  fi
  if [[ "$DOWNLOADER" == "curl" ]]; then
    curl --fail --location --continue-at - --retry 5 --retry-delay 3 \
      --connect-timeout 30 --progress-bar --output "$partial" "$url"
  else
    wget --continue --tries=5 --timeout=30 --output-document="$partial" "$url"
  fi
  [[ -s "$partial" ]] || die "downloaded empty file for ${label}"
  mv "$partial" "$destination"
}

available_gib() {
  df -Pk "$1" | awk 'NR == 2 {printf "%d", $4 / 1024 / 1024}'
}

validate_pickle_header() {
  local path="$1"
  if ((DRY_RUN)); then
    return
  fi
  python - "$path" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if path.stat().st_size < 1024 * 1024:
    raise SystemExit(f"annotation file is unexpectedly small: {path}")
with path.open("rb") as stream:
    prefix = stream.read(2)
if not prefix:
    raise SystemExit(f"annotation file is empty: {path}")
print(f"[assets] validated NTU annotation size: {path.stat().st_size / 2**30:.2f} GiB")
PY
}

download_qwen() {
  if [[ -f "${QWEN_DIR}/model.safetensors.index.json" ]]; then
    local shard_count
    shard_count="$(find "$QWEN_DIR" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l)"
    if ((shard_count >= 4)); then
      log "skip Qwen3-VL: found ${shard_count} weight shards"
      return
    fi
  fi
  command -v hf >/dev/null 2>&1 || die "the 'hf' command is required for --include-qwen; install huggingface_hub first"
  log "download Qwen3-VL-8B-Instruct to ${QWEN_DIR}"
  run hf download "$QWEN_REPO" --local-dir "$QWEN_DIR"
  if ((!DRY_RUN)); then
    [[ -f "${QWEN_DIR}/model.safetensors.index.json" ]] || die "Qwen model index is missing"
    local shard_count
    shard_count="$(find "$QWEN_DIR" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l)"
    ((shard_count >= 4)) || die "Qwen snapshot is incomplete: only ${shard_count} shards found"
  fi
}

write_checksums() {
  if ((DRY_RUN)); then
    log "would write SHA256 manifest: ${CHECKSUM_FILE}"
    return
  fi
  mkdir -p "$CHECKSUM_DIR"
  : > "$CHECKSUM_FILE"
  local path
  for path in \
    "${NTU_DIR}/ntu120_2d.pkl" \
    "${MEDIAPIPE_DIR}/pose_landmarker_lite.task"; do
    [[ -f "$path" ]] && sha256sum "$path" >> "$CHECKSUM_FILE"
  done
  if ((INCLUDE_QWEN)); then
    find "$QWEN_DIR" -maxdepth 1 -type f \
      \( -name '*.safetensors' -o -name '*.json' \) -print0 \
      | sort -z | xargs -0 -r sha256sum >> "$CHECKSUM_FILE"
  fi
}

detect_downloader
run mkdir -p "$NTU_DIR" "$CHECKSUM_DIR"

if ((!DRY_RUN)); then
  free_space="$(available_gib "$DATA_ROOT")"
  required_space=12
  ((INCLUDE_QWEN)) && required_space=35
  log "available disk space: ${free_space} GiB; recommended minimum: ${required_space} GiB"
  ((free_space >= required_space)) || die "insufficient disk space under ${DATA_ROOT}"
fi

download_file "$NTU120_2D_URL" "${NTU_DIR}/ntu120_2d.pkl" "NTU120 2D skeleton annotations"
validate_pickle_header "${NTU_DIR}/ntu120_2d.pkl"

if ((INCLUDE_MEDIAPIPE_LITE)); then
  download_file \
    "$MEDIAPIPE_LITE_URL" \
    "${MEDIAPIPE_DIR}/pose_landmarker_lite.task" \
    "MediaPipe Pose Landmarker Lite"
fi

if ((INCLUDE_QWEN)); then
  download_qwen
else
  log "Qwen3-VL download skipped (use --include-qwen only if the server has no complete snapshot)"
fi

write_checksums

log "complete"
log "NTU120-2D: ${NTU_DIR}/ntu120_2d.pkl"
if ((INCLUDE_MEDIAPIPE_LITE)); then
  log "MediaPipe Lite: ${MEDIAPIPE_DIR}/pose_landmarker_lite.task"
fi
log "checksums: ${CHECKSUM_FILE}"
