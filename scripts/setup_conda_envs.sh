#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-conda}"
MAIN_ENV="${MAIN_ENV:-uniba_repro}"
PLIP_ENV="${PLIP_ENV:-uniba_plip}"

"${CONDA_BIN}" create -y -n "${MAIN_ENV}" python=3.10.20 pip
"${CONDA_BIN}" run -n "${MAIN_ENV}" python -m pip install \
  torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
"${CONDA_BIN}" run -n "${MAIN_ENV}" python -m pip install \
  torch-scatter==2.1.2+pt20cu118 \
  -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
"${CONDA_BIN}" run -n "${MAIN_ENV}" python -m pip install \
  -r "${PROJECT_ROOT}/requirements.txt"

"${CONDA_BIN}" create -y -n "${PLIP_ENV}" -c conda-forge \
  python=3.10.20 openbabel=3.1.1 pip
"${CONDA_BIN}" run -n "${PLIP_ENV}" python -m pip install \
  numpy==1.26.4 plip==2.4.0

echo "Created ${MAIN_ENV} and ${PLIP_ENV}."
