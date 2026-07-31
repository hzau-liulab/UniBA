#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ASSET_DIR="${UNIBA_EXTERNAL_ASSET_DIR:-${PROJECT_ROOT}/external_assets}"
MINT_URL="${MINT_URL:-https://huggingface.co/varunullanat2012/mint/resolve/main/mint.ckpt}"
MINT_SHA256="84a4016365997cd9f0bccb07d746fa8f076ffd8e45aa0cbcf4e50a037161a342"

mkdir -p "${ASSET_DIR}/mint" "${ASSET_DIR}/esm_cache"
mint_checkpoint="${ASSET_DIR}/mint/mint.ckpt"

if [[ ! -s "${mint_checkpoint}" ]]; then
  curl -fL --retry 5 --retry-delay 5 "${MINT_URL}" -o "${mint_checkpoint}.part"
  mv "${mint_checkpoint}.part" "${mint_checkpoint}"
fi
printf '%s  %s\n' "${MINT_SHA256}" "${mint_checkpoint}" | sha256sum -c -

TORCH_HOME="${ASSET_DIR}/esm_cache" "${PYTHON_BIN}" - <<'PY'
import esm

esm.pretrained.esm2_t33_650M_UR50D()
esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
print("ESM2 and ESM-IF checkpoints are available.")
PY

echo "External model assets are under ${ASSET_DIR}."
