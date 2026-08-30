#!/usr/bin/env bash
#
# Download and prepare the KTH, BEHAVE and LIMU manual-review candidates.
#
# The default output root intentionally follows the server's existing
# ``datasets/vedio`` spelling. Downloads use ``.part`` files so interrupted
# transfers can be resumed safely.

set -euo pipefail

default_root="/workspace/data/xzz_data/DAHUA/datasets/vedio"
candidate_root="${default_root}"
python_bin="${DAHUA_VIS_PYTHON:-python}"
ffmpeg_bin="${DAHUA_FFMPEG:-ffmpeg}"
disable_proxy=0
force_prepare=0

usage() {
    cat <<EOF
Usage: bash dahua_cup/scripts/download_behavior_candidates.sh [options]

Download KTH, BEHAVE and LIMU candidates, extract them, cut BEHAVE clips and
write manual_label_manifest.csv.

Options:
  --root PATH       Output root (default: ${default_root})
  --python PATH     Python used by preparation scripts (default: ${python_bin})
  --ffmpeg PATH     FFmpeg executable (default: ${ffmpeg_bin})
  --no-proxy        Ignore configured HTTP/HTTPS proxy variables
  --force-prepare   Rebuild BEHAVE clips even if its manifest already exists
  -h, --help        Show this help
EOF
}

while (($#)); do
    case "$1" in
        --root)
            candidate_root="$2"
            shift 2
            ;;
        --python)
            python_bin="$2"
            shift 2
            ;;
        --ffmpeg)
            ffmpeg_bin="$2"
            shift 2
            ;;
        --no-proxy)
            disable_proxy=1
            shift
            ;;
        --force-prepare)
            force_prepare=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
code_root="$(cd "${script_dir}/../.." && pwd)"
candidate_root="$(mkdir -p "${candidate_root}" && cd "${candidate_root}" && pwd)"

for command in curl unzip; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        printf 'required command is unavailable: %s\n' "${command}" >&2
        exit 1
    fi
done
if [[ "${python_bin}" == */* ]]; then
    if [[ ! -x "${python_bin}" ]]; then
        printf 'Python executable is unavailable: %s\n' "${python_bin}" >&2
        exit 1
    fi
elif ! command -v "${python_bin}" >/dev/null 2>&1; then
    printf 'Python executable is unavailable: %s\n' "${python_bin}" >&2
    exit 1
fi
if [[ "${ffmpeg_bin}" == */* ]]; then
    if [[ ! -x "${ffmpeg_bin}" ]]; then
        printf 'FFmpeg executable is unavailable: %s\n' "${ffmpeg_bin}" >&2
        exit 1
    fi
elif ! command -v "${ffmpeg_bin}" >/dev/null 2>&1; then
    printf 'FFmpeg executable is unavailable: %s\n' "${ffmpeg_bin}" >&2
    exit 1
fi

curl_options=(
    --fail
    --location
    --retry 5
    --retry-delay 3
    --connect-timeout 30
)
if ((disable_proxy)); then
    curl_options+=(--noproxy '*')
fi

download() {
    local url="$1"
    local destination="$2"
    local partial="${destination}.part"
    mkdir -p "$(dirname "${destination}")"
    if [[ -s "${destination}" ]]; then
        printf '[skip] %s\n' "${destination}"
        return
    fi
    printf '[download] %s\n' "${url}"
    curl "${curl_options[@]}" \
        --continue-at - \
        --output "${partial}" \
        "${url}"
    mv "${partial}" "${destination}"
}

printf '\n[1/6] Downloading KTH archives\n'
kth_base="https://www.csc.kth.se/cvap/actions"
for action in walking jogging running boxing; do
    download \
        "${kth_base}/${action}.zip" \
        "${candidate_root}/kth/archives/${action}.zip"
done

printf '\n[2/6] Extracting KTH archives\n'
for action in walking jogging running boxing; do
    destination="${candidate_root}/kth/extracted/${action}"
    mkdir -p "${destination}"
    unzip -q -o \
        "${candidate_root}/kth/archives/${action}.zip" \
        -d "${destination}"
done

printf '\n[3/6] Downloading BEHAVE annotations and AVI segments\n'
behave_base="https://groups.inf.ed.ac.uk/vision/DATASETS/BEHAVEDATA/INTERACTIONS"
behave_raw="${candidate_root}/behave/raw"
download "${behave_base}/markup.txt" "${behave_raw}/markup.txt"
for video in \
    1-11200.avi \
    11500-17450.avi \
    18000-23700.avi \
    24300-35200.avi \
    35450-47160.avi \
    47300-58400.avi \
    59800-66750.avi \
    67210-76800.avi
do
    download "${behave_base}/${video}" "${behave_raw}/${video}"
done

printf '\n[4/6] Preparing BEHAVE interaction clips\n'
encoder_list="$("${ffmpeg_bin}" -hide_banner -encoders 2>/dev/null || true)"
if [[ "${encoder_list}" == *"libx264"* ]]; then
    behave_codec="libx264"
elif [[ "${encoder_list}" == *"libopenh264"* ]]; then
    behave_codec="libopenh264"
elif [[ "${encoder_list}" == *" mpeg4 "* ]]; then
    behave_codec="mpeg4"
else
    printf 'FFmpeg has no supported MP4 encoder (libx264/libopenh264/mpeg4)\n' >&2
    exit 1
fi
behave_prepared="${candidate_root}/behave/prepared"
prepared_clip="$(
    find "${behave_prepared}/clips" -type f -name '*.mp4' -print -quit \
        2>/dev/null || true
)"
if ((${force_prepare} == 0)) \
    && [[ -s "${behave_prepared}/candidate_manifest.csv" ]] \
    && [[ -n "${prepared_clip}" ]]; then
    printf '[skip] prepared BEHAVE clips already exist\n'
else
    "${python_bin}" "${code_root}/dahua_cup/scripts/prepare_behave_manual_labels.py" \
        --source-dir "${behave_raw}" \
        --output-dir "${behave_prepared}" \
        --ffmpeg "${ffmpeg_bin}" \
        --codec "${behave_codec}"
fi

printf '\n[5/6] Downloading and extracting LIMU videos\n'
limu_archive="${candidate_root}/limu/archives/video.zip"
download \
    "https://limu.ait.kyushu-u.ac.jp/dataset/download_count/download.php?download=13" \
    "${limu_archive}"
mkdir -p "${candidate_root}/limu/extracted"
unzip -q -o "${limu_archive}" -d "${candidate_root}/limu/extracted"

expected_limu="${candidate_root}/limu/extracted/video-interaction"
if [[ ! -d "${expected_limu}" ]]; then
    discovered_limu="$(
        find "${candidate_root}/limu/extracted" \
            -type d -name video-interaction -print -quit
    )"
    if [[ -z "${discovered_limu}" ]]; then
        printf 'LIMU archive has no video-interaction directory\n' >&2
        exit 1
    fi
    ln -s "${discovered_limu}" "${expected_limu}"
fi

printf '\n[6/6] Building the unified Web review manifest\n'
"${python_bin}" "${code_root}/dahua_cup/scripts/build_manual_label_manifest.py" \
    --root "${candidate_root}" \
    --output "${candidate_root}/manual_label_manifest.csv"

kth_count="$(
    find "${candidate_root}/kth/extracted" -type f -iname '*.avi' | wc -l
)"
behave_count="$(
    find "${candidate_root}/behave/prepared/clips" -type f -iname '*.mp4' | wc -l
)"
limu_count="$(
    find "${expected_limu}" -type f -iname '*.avi' | wc -l
)"
manifest_count="$(
    tail -n +2 "${candidate_root}/manual_label_manifest.csv" | wc -l
)"

cat <<EOF

Download and preparation complete.
  root:     ${candidate_root}
  KTH:      ${kth_count} AVI files
  BEHAVE:   ${behave_count} MP4 clips
  LIMU:     ${limu_count} AVI files
  manifest: ${manifest_count} rows

Configure the Web server with:
  export DAHUA_VIS_SOURCE_ROOT=${candidate_root}
  export DAHUA_VIS_MANIFEST=${candidate_root}/manual_label_manifest.csv
EOF
