#!/usr/bin/env bash
# Download the Campus6 model assets pinned to a GitHub release and verify them.
#
# The weights under models/ are not tracked by Git.  This script fetches the
# models.tar.gz release asset, checks its SHA-256 against the published
# models.tar.gz.sha256 sidecar, unpacks it into the repository root, and
# verifies the resulting models/MANIFEST.json exists.
#
# Usage: bash scripts/download_models.sh [--tag TAG] [--repo OWNER/REPO]
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/download_models.sh [options]

Options:
  --tag TAG       GitHub release tag (default: v1.0.0-rc1)
  --repo OWNER/NAME
                  GitHub repository (default: Usernamezju/DAHUA)
  --work-dir DIR  Download scratch directory (default: ~/temp)
  -h, --help      Show this help.

The asset is unpacked into the repository root (the directory that contains
models/MANIFEST.json after extraction).  Downloading needs no GitHub
authentication for public repositories.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tag="v1.0.0-rc1"
repo="Usernamezju/DAHUA"
work_dir="${HOME}/temp"

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
asset="models.tar.gz"
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

if [[ ! -f "${repo_root}/models/MANIFEST.json" ]]; then
  echo "models/MANIFEST.json missing after extraction" >&2
  exit 1
fi
echo "Model assets ready in ${repo_root}/models"
