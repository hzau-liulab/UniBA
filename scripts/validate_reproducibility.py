#!/usr/bin/env python3
"""Strict checks for UniBA paper reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_SPLITS = {
    "all_train.txt": 2513,
    "test_pair.txt": 619,
    "test1_pair.txt": 79,
    "test2_pair.txt": 90,
    "test3_pair.txt": 34,
    "test4_pair.txt": 166,
    "train_cv1.txt": 2198,
    "train_cv2.txt": 2198,
    "train_cv3.txt": 2198,
    "train_cv4.txt": 2198,
    "train_cv5.txt": 2198,
    "val_cv1.txt": 315,
    "val_cv2.txt": 315,
    "val_cv3.txt": 315,
    "val_cv4.txt": 315,
    "val_cv5.txt": 315,
}

CHECKPOINT_REQUIRED = [
    Path("data/PPB-Affinity"),
    Path("data/split_data/cv"),
    Path("data/affinity_data/all_data_with_multi_labels.json"),
    Path("data/cdr_sequences.json"),
    Path("feature/mint/data/ppi_sequences_for_mint.csv"),
    Path("feature/mint/data/aai_sequences_for_mint.csv"),
    Path("feature/mint/data/tcr-pmhc_sequences_for_mint.csv"),
    Path("feature/max_ASA.txt"),
    Path("feature/tr_cv_global_scaler.pkl"),
    Path("data/affinity_data/cg_input/martini_v2.2.itp"),
    Path("data/affinity_data/cg_input/martini_v2.0_ions.itp"),
    Path("data/affinity_data/cg_input/water.gro"),
    Path("data/affinity_data/cg_input/minim.mdp"),
    Path("data/affinity_data/cg_input/min_steep.mdp"),
    Path("data/affinity_data/cg_input/min_cg.mdp"),
    Path("ablation/UniBA_wo_res/UniBA_0_1.pth"),
    Path("ablation/UniBA_wo_res/UniBA_1_6.pth"),
    Path("ablation/UniBA_wo_res/UniBA_2_2.pth"),
    Path("ablation/UniBA_wo_res/UniBA_3_1.pth"),
    Path("ablation/UniBA_wo_res/UniBA_4_51.pth"),
]

RAW_REQUIRED = [
    Path("data/pdb_files/ppi"),
    Path("data/pdb_files/aai"),
    Path("data/pdb_files/tcr-pmhc"),
]

GENERATED_REQUIRED = [
    Path("feature/seq_features"),
    Path("feature/str_features"),
    Path("feature/hand_str_feat_norm"),
    Path("feature/dist_map"),
    Path("feature/energy_features"),
    Path("feature/plip_feat"),
    Path("data/affinity_data/res_graph"),
    Path("data/affinity_data/cg_inter_graph"),
    Path("data/affinity_data/cg_intra_graph"),
]


def count_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())


def fail_missing(paths: list[Path]) -> list[str]:
    return [f"missing: {path}" for path in paths if not path.exists()]


def non_readme_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and item.name != "README.md")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["metadata", "checkpoints", "prepared", "raw", "generated", "all"], default="prepared")
    args = parser.parse_args()

    errors: list[str] = []

    metadata_paths = [
        Path("README.md"),
        Path("data/PPB-Affinity"),
        Path("data/split_data/cv"),
        Path("data/affinity_data/all_data_with_multi_labels.json"),
    ]
    errors.extend(fail_missing(metadata_paths))

    split_dir = Path("data/split_data/cv")
    if split_dir.exists():
        for name, expected in EXPECTED_SPLITS.items():
            path = split_dir / name
            if not path.exists():
                errors.append(f"missing split: {path}")
                continue
            observed = count_lines(path)
            if observed != expected:
                errors.append(f"split count mismatch: {path} observed {observed}, expected {expected}")

    label_json = Path("data/affinity_data/all_data_with_multi_labels.json")
    if label_json.exists():
        labels = json.loads(label_json.read_text(encoding="utf-8"))
        if len(labels) != 3327:
            errors.append(f"label count mismatch: {len(labels)} observed, expected 3327")
        pair_ids: set[str] = set()
        for dataset in ("ppi", "aai", "tcr-pmhc"):
            pair_path = Path("data/PPB-Affinity") / f"{dataset}_pair.txt"
            if not pair_path.exists():
                errors.append(f"missing pair list: {pair_path}")
                continue
            pair_ids.update(
                line.strip()
                for line in pair_path.read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines()
                if line.strip()
            )
        label_ids = set(labels)
        for pair in sorted(pair_ids - label_ids):
            errors.append(f"pair list entry has no label: {pair}")
        for pair in sorted(label_ids - pair_ids):
            errors.append(f"label entry is absent from pair lists: {pair}")

    if args.mode in {"checkpoints", "prepared", "all"}:
        errors.extend(fail_missing(CHECKPOINT_REQUIRED))

    if args.mode in {"raw", "all"}:
        errors.extend(fail_missing(RAW_REQUIRED))
        for dataset in ("ppi", "aai", "tcr-pmhc"):
            pair_path = Path("data/PPB-Affinity") / f"{dataset}_pair.txt"
            pdb_dir = Path("data/pdb_files") / dataset
            if not pair_path.exists() or not pdb_dir.exists():
                continue
            pairs = [
                line.strip()
                for line in pair_path.read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines()
                if line.strip()
            ]
            missing_pdbs = [
                pair for pair in pairs if not (pdb_dir / f"{pair}.pdb").is_file()
            ]
            if missing_pdbs:
                errors.append(
                    f"raw PDBs missing for {dataset}: {len(missing_pdbs)} "
                    f"(first: {missing_pdbs[0]})"
                )

    if args.mode in {"prepared", "generated", "all"}:
        errors.extend(fail_missing(GENERATED_REQUIRED))
        for path in GENERATED_REQUIRED:
            if path.exists() and non_readme_files(path) == 0:
                errors.append(f"generated artifact directory is empty: {path}")

    if errors:
        print("Reproducibility validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Reproducibility validation passed for mode: {args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
