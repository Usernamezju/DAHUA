#!/usr/bin/env bash
set -euo pipefail

repo_root="${DAHUA_REPOSITORY_ROOT:-/workspace/code/DAHUA}"
tools_root="${DAHUA_CANDIDATE_TOOLS:-${repo_root}/.tools}"
yt_dlp="${tools_root}/yt-dlp"

mkdir -p "${tools_root}"

if [[ ! -x "${yt_dlp}" ]]; then
  echo "[candidate-tools] downloading official standalone yt-dlp"
  curl -fL \
    --retry 5 \
    --retry-delay 3 \
    --continue-at - \
    https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux \
    --output "${yt_dlp}.part"
  mv "${yt_dlp}.part" "${yt_dlp}"
  chmod 755 "${yt_dlp}"
fi

"${yt_dlp}" --version

ffmpeg_bin="${DAHUA_FFMPEG:-/root/miniconda3/envs/skel_gcn38/bin/ffmpeg}"
if [[ ! -x "${ffmpeg_bin}" ]]; then
  echo "[candidate-tools] ffmpeg is missing: ${ffmpeg_bin}" >&2
  exit 2
fi
"${ffmpeg_bin}" -version | head -1

echo "[candidate-tools] ready"
