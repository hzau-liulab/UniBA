#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-.}"
cd "${PROJECT_ROOT}"

restore_tar() {
  local archive="$1"
  if [[ -f "${archive}" ]]; then
    echo "Restoring ${archive}"
    case "${archive}" in
      *.tar.gz|*.tgz) tar -xzf "${archive}" ;;
      *.tar.zst) tar --use-compress-program=unzstd -xf "${archive}" ;;
      *.tar) tar -xf "${archive}" ;;
      *) echo "Unsupported archive: ${archive}" >&2; exit 1 ;;
    esac
  fi
}

mkdir -p ablation/UniBA_wo_res
if [[ -d artifacts/checkpoints/UniBA_wo_res ]]; then
  cp -a artifacts/checkpoints/UniBA_wo_res/. ablation/UniBA_wo_res/
fi

restore_tar artifacts/uniba_raw_pdb.tar.zst
restore_tar artifacts/uniba_prepared_features.tar.zst
restore_tar artifacts/uniba_prepared_graphs.tar.zst
restore_tar artifacts/uniba_checkpoints.tar.gz

echo "Artifact restore complete."
