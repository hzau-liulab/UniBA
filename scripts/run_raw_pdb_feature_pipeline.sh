#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_FILE=""
ONLY_STAGES=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_raw_pdb_feature_pipeline.sh [options]

Options:
  --env-file <path>  Load variables from a shell env file
  --stages <list>    Comma-separated stage list: data,mint,seqstr,handnorm,plip,cgprep,energy,graphs
  --dry-run          Print commands without executing them
  --help             Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --stages)
      ONLY_STAGES="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS="${DATASETS:-ppi,aai,tcr-pmhc}"
RUN_DATA_PROCESSING="${RUN_DATA_PROCESSING:-0}"
MINT_ENABLED="${MINT_ENABLED:-1}"
SEQ_STR_ENABLED="${SEQ_STR_ENABLED:-0}"
PLIP_ENABLED="${PLIP_ENABLED:-1}"
ENERGY_ENABLED="${ENERGY_ENABLED:-0}"
CG_INPUT_PREP_ENABLED="${CG_INPUT_PREP_ENABLED:-0}"
RES_GRAPH_ENABLED="${RES_GRAPH_ENABLED:-1}"
CG_GRAPH_ENABLED="${CG_GRAPH_ENABLED:-1}"
CG_GRAPH_MODE="${CG_GRAPH_MODE:-both}"
STOP_ON_MISSING="${STOP_ON_MISSING:-1}"

should_run_stage() {
  local stage="$1"
  if [[ -z "${ONLY_STAGES}" ]]; then
    return 0
  fi
  [[ ",${ONLY_STAGES}," == *",${stage},"* ]]
}

run_shell_cmd() {
  local label="$1"
  local cmd="$2"
  echo
  echo "==> ${label}"
  echo "    ${cmd}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  bash -lc "${cmd}"
}

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[dry-run missing] ${path}" >&2
      return 0
    fi
    echo "Missing required path: ${path}" >&2
    [[ "${STOP_ON_MISSING}" == "1" ]] && exit 1
  fi
}

format_template() {
  local template="$1"
  local dataset="$2"
  local pair_list="${PROJECT_ROOT}/data/PPB-Affinity/${dataset}_pair.txt"
  template="${template//\{root\}/${PROJECT_ROOT}}"
  template="${template//\{python\}/${PYTHON_BIN}}"
  template="${template//\{dataset\}/${dataset}}"
  template="${template//\{pair_list\}/${pair_list}}"
  template="${template//\{esm_path\}/${ESM_PATH:-}}"
  template="${template//\{dssp\}/${DSSP_BIN:-}}"
  template="${template//\{ghecom\}/${GHECOM_BIN:-}}"
  template="${template//\{psaia\}/${PSAIA_DIR:-}}"
  template="${template//\{martinize\}/${MARTINIZE_SCRIPT:-}}"
  template="${template//\{python2\}/${MARTINIZE_PYTHON:-python2}}"
  template="${template//\{gmx\}/${GMX_BIN:-gmx}}"
  printf '%s' "${template}"
}

IFS=',' read -r -a DATASET_ARRAY <<< "${DATASETS}"

if should_run_stage data; then
  if [[ "${RAW_INPUT_PREP_ENABLED:-1}" == "1" ]]; then
    run_shell_cmd "Prepare per-chain PDB/FASTA inputs" "cd \"${PROJECT_ROOT}\" && \"${PYTHON_BIN}\" data/prepare_raw_pdb_inputs.py --project-root \"${PROJECT_ROOT}\" --datasets \"${DATASETS}\" ${RAW_INPUT_PREP_EXTRA_ARGS:-}"
  elif [[ -n "${DATA_PREP_CMD:-}" ]]; then
    data_cmd="${DATA_PREP_CMD}"
    run_shell_cmd "Data processing" "cd \"${PROJECT_ROOT}\" && ${data_cmd}"
  else
    echo "Data preparation is disabled; skipping the data stage."
  fi
fi

for dataset in "${DATASET_ARRAY[@]}"; do
  dataset="$(echo "${dataset}" | xargs)"
  [[ -z "${dataset}" ]] && continue

  pair_list="${PROJECT_ROOT}/data/PPB-Affinity/${dataset}_pair.txt"
  require_file "${pair_list}"

  if should_run_stage mint && [[ "${MINT_ENABLED}" == "1" ]]; then
    require_file "${MINT_CHECKPOINT:-}"
    mint_csv="${PROJECT_ROOT}/feature/mint/data/${dataset}_sequences_for_mint.csv"
    require_file "${mint_csv}"
    run_shell_cmd \
      "MINT sequence embeddings (${dataset})" \
      "cd \"${PROJECT_ROOT}\" && \"${PYTHON_BIN}\" feature/get_mint_emb.py --config \"${MINT_CONFIG}\" --checkpoint \"${MINT_CHECKPOINT}\" --csv \"${mint_csv}\" --pair-list \"${pair_list}\" --output \"${PROJECT_ROOT}/feature/seq_features/${dataset}_mint_embeddings.pt\" --device \"${MINT_DEVICE:-cuda:0}\""
  fi

  if should_run_stage seqstr && [[ "${SEQ_STR_ENABLED}" == "1" ]]; then
    run_shell_cmd "Sequence/structure/handcrafted features (${dataset})" "$(format_template "${SEQ_STR_CMD_TEMPLATE}" "${dataset}")"
  fi

  if should_run_stage handnorm && [[ "${SEQ_STR_ENABLED}" == "1" ]]; then
    run_shell_cmd "Normalize handcrafted structure features (${dataset})" "$(format_template "${HAND_NORM_CMD_TEMPLATE}" "${dataset}")"
  fi

  if should_run_stage plip && [[ "${PLIP_ENABLED}" == "1" ]]; then
    run_shell_cmd \
      "PLIP XML reports (${dataset})" \
      "cd \"${PROJECT_ROOT}\" && DATA_TYPE=\"${dataset}\" PROJECT_ROOT=\"${PROJECT_ROOT}\" PAIR_LIST_FILE=\"${pair_list}\" PLIP_BIN=\"${PLIP_BIN:-plip}\" PLIP_OVERWRITE=\"${PLIP_OVERWRITE:-0}\" bash feature/run_plip_batch.sh"
    run_shell_cmd \
      "Collect PLIP features (${dataset})" \
      "cd \"${PROJECT_ROOT}\" && \"${PYTHON_BIN}\" feature/get_interaction_feature.py --pair-list \"${pair_list}\" --plip-root \"${PROJECT_ROOT}/feature/plip_feat/${dataset}\" --output \"${PROJECT_ROOT}/feature/plip_feat/${dataset}_interaction.pkl\""
  fi

  if should_run_stage cgprep && [[ "${CG_INPUT_PREP_ENABLED}" == "1" ]]; then
    if [[ -z "${CG_INPUT_PREP_CMD:-}" ]]; then
      echo "Missing CG_INPUT_PREP_CMD while CG_INPUT_PREP_ENABLED=1" >&2
      exit 1
    fi
    run_shell_cmd "Prepare Martini/Gromacs coarse-grained inputs (${dataset})" "$(format_template "${CG_INPUT_PREP_CMD}" "${dataset}")"
  fi

  if should_run_stage energy && [[ "${ENERGY_ENABLED}" == "1" ]]; then
    run_shell_cmd "Energy features (${dataset})" "$(format_template "${ENERGY_CMD_TEMPLATE}" "${dataset}")"
  fi

  if should_run_stage graphs; then
    if [[ "${RES_GRAPH_ENABLED}" == "1" ]]; then
      run_shell_cmd \
        "Residue graph (${dataset})" \
        "cd \"${PROJECT_ROOT}\" && \"${PYTHON_BIN}\" data/build_res_graph.py --data-type \"${dataset}\" --project-root \"${PROJECT_ROOT}\" --cdr-json \"${CDR_JSON:-${PROJECT_ROOT}/data/cdr_sequences.json}\" --label-json \"${LABEL_JSON:-${PROJECT_ROOT}/data/affinity_data/all_data_with_multi_labels.json}\""
    fi
    if [[ "${CG_GRAPH_ENABLED}" == "1" ]]; then
      if [[ "${CG_GRAPH_MODE}" == "both" ]]; then
        modes=(inter intra)
      else
        modes=("${CG_GRAPH_MODE}")
      fi
      for mode in "${modes[@]}"; do
        run_shell_cmd \
          "Coarse-grained ${mode} graph (${dataset})" \
          "cd \"${PROJECT_ROOT}\" && \"${PYTHON_BIN}\" data/build_cg_graph.py --data-type \"${dataset}\" --pair-list \"${pair_list}\" --mode \"${mode}\" --project-root \"${PROJECT_ROOT}\""
      done
    fi
  fi
done
