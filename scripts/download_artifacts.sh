#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${UNIBA_ARTIFACT_BASE_URL:-}"
OUT_DIR="${1:-artifacts}"
WITH_PREPARED_CACHE=0

if [[ "${1:-}" == "--with-prepared-cache" ]]; then
  OUT_DIR="artifacts"
  WITH_PREPARED_CACHE=1
elif [[ "${2:-}" == "--with-prepared-cache" ]]; then
  WITH_PREPARED_CACHE=1
fi

if [[ -z "${BASE_URL}" ]]; then
  echo "Set UNIBA_ARTIFACT_BASE_URL to the URL containing UniBA release artifacts." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

download_one() {
  local name="$1"
  local url="${BASE_URL%/}/${name}"
  echo "Downloading ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "${OUT_DIR}/${name}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${OUT_DIR}/${name}" "${url}"
  else
    echo "curl or wget is required" >&2
    exit 1
  fi
}

download_one SHA256SUMS
download_one uniba_raw_pdb.tar.zst
download_one uniba_checkpoints.tar.gz

if [[ "${WITH_PREPARED_CACHE}" == "1" ]]; then
  download_one uniba_prepared_features.tar.zst
  download_one uniba_prepared_graphs.tar.zst
fi

if command -v sha256sum >/dev/null 2>&1; then
  (cd "${OUT_DIR}" && sha256sum -c SHA256SUMS)
else
  echo "sha256sum not found; checksum verification skipped" >&2
fi
