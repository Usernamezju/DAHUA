#!/usr/bin/env bash
# Download the Campus6 delivery dataset pinned to a GitHub release and verify it.
#
# The dataset under dataset/campus_all/ is not tracked by Git.  This script
# fetches the campus_all_release_*.tar release asset, checks its SHA-256, and
# unpacks it into dataset/campus_all/ inside the repository root.  The expected
# digest is pinned below and can be overridden with a .sha256 sidecar uploaded
# next to the archive.
#
# Usage: bash scripts/download_dataset.sh [--tag TAG] [--repo OWNER/REPO]
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/download_dataset.sh [options]

Options:
  --tag TAG       GitHub release tag (default: campus6-data-v1.0.0)
  --repo OWNER/NAME
                  GitHub repository (default: Usernamezju/DAHUA)
  --work-dir DIR  Download scratch directory (default: ~/temp)
  -h, --help      Show this help.

The archive is unpacked into the repository root so that
dataset/campus_all/annotations_with_all.pkl ends up at
<repo>/dataset/campus_all/annotations_with_all.pkl.  Downloading needs no
GitHub authentication for public repositories.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tag="campus6-data-v1.0.0"
repo="Usernamezju/DAHUA"
work_dir="${HOME}/temp"

# SHA-256 published in the release notes of campus6-data-v1.0.0.  A
# <asset>.sha256 sidecar uploaded next to the archive, when present, wins.
PINNED_SHA256="2990addbc56cba1953a8cc4f1f178e0a4320aa4407a2434ed77f3eb3c99d9248"

while (($#)); do
  case "$1" in
    --tag) tag="$2"; shift 2 ;;
    --repo) repo="$2"; shift 2 ;;
    --work-dir) work_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$work_dir"
asset="campus_all_release_20260905.tar"
base_url="https://github.com/${repo}/releases/download/${tag}"
archive="${work_dir}/${asset}"
digest_file="${work_dir}/${asset}.sha256"

echo "Downloading ${base_url}/${asset}"
curl -fL --retry 3 -o "${archive}.part" "${base_url}/${asset}"
mv "${archive}.part" "$archive"

# Prefer the published sidecar; fall back to the pinned release-notes digest.
if curl -fsL --retry 1 -o "${digest_file}.part" "${base_url}/${asset}.sha256"; then
  mv "${digest_file}.part" "$digest_file"
  expected="$(awk '{print $1}' "$digest_file")"
  echo "Using SHA-256 from published sidecar"
else
  rm -f "${digest_file}.part"
  expected="$PINNED_SHA256"
  echo "No sidecar published; using pinned release-notes SHA-256"
fi

actual="$(sha256sum "$archive" | awk '{print $1}')"
if [[ "$expected" != "$actual" ]]; then
  echo "SHA-256 mismatch:" >&2
  echo "  expected: $expected" >&2
  echo "  actual:   $actual" >&2
  exit 1
fi
echo "SHA-256 verified: $actual"

echo "Unpacking into ${repo_root}/dataset"
mkdir -p "${repo_root}/dataset"
tar -xf "$archive" -C "${repo_root}/dataset"

if [[ ! -f "${repo_root}/dataset/campus_all/annotations_with_all.pkl" ]]; then
  echo "dataset/campus_all/annotations_with_all.pkl missing after extraction" >&2
  exit 1
fi
echo "Dataset ready in ${repo_root}/dataset/campus_all"
