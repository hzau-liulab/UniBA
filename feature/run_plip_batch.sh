#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_TYPE="${DATA_TYPE:-case}"
PLIP_BIN="${PLIP_BIN:-plip}"
PLIP_OVERWRITE="${PLIP_OVERWRITE:-0}"

pdb_folder="${PDB_FOLDER:-${PROJECT_ROOT}/data/pdb_files/${DATA_TYPE}_mapped}"
output_folder="${PLIP_OUTPUT_ROOT:-${PROJECT_ROOT}/feature/plip_feat/${DATA_TYPE}}"
pair_list_file="${PAIR_LIST_FILE:-${PROJECT_ROOT}/data/PPB-Affinity/${DATA_TYPE}_pair.txt}"

mkdir -p "${output_folder}"

if [[ ! -f "${pair_list_file}" ]]; then
  echo "Missing pair list: ${pair_list_file}" >&2
  exit 1
fi

format_chains() {
  local str="$1"
  local out=""
  local i
  for ((i = 0; i < ${#str}; i++)); do
    out="${out}'${str:$i:1}',"
  done
  echo "${out%,}"
}

while IFS= read -r pair; do
  pair="$(echo "${pair}" | tr -d '\r' | xargs)"
  [[ -z "${pair}" ]] && continue

  echo "Processing pair: ${pair}"

  pdb_file="${pdb_folder}/${pair}.pdb"
  output_dir="${output_folder}/${pair}"

  if [[ ! -f "${pdb_file}" ]]; then
    echo "Missing PDB: ${pdb_file}" >&2
    continue
  fi

  if [[ "${PLIP_OVERWRITE}" == "1" ]]; then
    rm -rf "${output_dir}"
  fi
  mkdir -p "${output_dir}"

  chain1="$(echo "${pair}" | cut -d'_' -f2)"
  chain2="$(echo "${pair}" | cut -d'_' -f3 | cut -d'.' -f1)"

  if [[ "${pair}" == "6ysq_AC_G" ]]; then
    mapped_c1="A"
    mapped_c2="B"
  else
    all_input_chains="${chain1}${chain2}"
    af3_chain_ids="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    declare -A chain_map=()

    for ((i = 0; i < ${#all_input_chains}; i++)); do
      old="${all_input_chains:$i:1}"
      new="${af3_chain_ids:$i:1}"
      chain_map["${old}"]="${new}"
    done

    mapped_c1=""
    for ((i = 0; i < ${#chain1}; i++)); do
      c="${chain1:$i:1}"
      mapped_c1+="${chain_map[$c]}"
    done

    mapped_c2=""
    for ((i = 0; i < ${#chain2}; i++)); do
      c="${chain2:$i:1}"
      mapped_c2+="${chain_map[$c]}"
    done
  fi

  c1="$(format_chains "${mapped_c1}")"
  c2="$(format_chains "${mapped_c2}")"
  chains_arg="[[${c1}], [${c2}]]"

  "${PLIP_BIN}" -f "${pdb_file}" -o "${output_dir}" --chains "${chains_arg}" -x -t
done < "${pair_list_file}"
