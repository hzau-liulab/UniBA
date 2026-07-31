#!/usr/bin/env python3
"""Validate the portable UniBA GitHub source tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

REQUIRED_DOCS = [Path("README.md")]
REQUIRED_CODE = [
    Path("UniAffinity.py"), Path("arg_parse.py"), Path("model"), Path("utils"),
    Path("data/build_res_graph.py"), Path("data/build_cg_graph.py"),
    Path("data/prepare_raw_pdb_inputs.py"), Path("feature/run_seq_str_features.py"),
    Path("scripts/run_raw_pdb_feature_pipeline.sh"),
    Path("scripts/prepare_martini_cg.py"),
    Path("scripts/raw_pdb_pipeline.env.example"),
    Path("scripts/setup_conda_envs.sh"),
    Path("scripts/download_external_models.sh"),
    Path("envs/uniba-cu118.yml"), Path("envs/plip.yml"),
    Path("configs/UniBA_fold_best_epoch.json"),
    Path("scripts/validate_reproducibility.py"),
    Path("scripts/restore_artifacts.sh"), Path("scripts/download_artifacts.sh"),
    Path("feature/max_ASA.txt"), Path("feature/tr_cv_global_scaler.pkl"),
    Path("data/affinity_data/cg_input/martini_v2.2.itp"),
    Path("data/affinity_data/cg_input/martini_v2.0_ions.itp"),
    Path("data/affinity_data/cg_input/water.gro"),
    Path("data/affinity_data/cg_input/minim.mdp"),
    Path("data/affinity_data/cg_input/min_steep.mdp"),
    Path("data/affinity_data/cg_input/min_cg.mdp"),
]
REQUIRED_METADATA = [
    Path("data/PPB-Affinity"), Path("data/split_data/cv"),
    Path("data/affinity_data/all_data_with_multi_labels.json"),
    Path("data/cdr_sequences.json"),
    Path("feature/mint/data/ppi_sequences_for_mint.csv"),
    Path("feature/mint/data/aai_sequences_for_mint.csv"),
    Path("feature/mint/data/tcr-pmhc_sequences_for_mint.csv"),
]
EXPECTED_SPLITS = {
    "all_train.txt": 2513, "test_pair.txt": 619, "test1_pair.txt": 79,
    "test2_pair.txt": 90, "test3_pair.txt": 34, "test4_pair.txt": 166,
    "train_cv1.txt": 2198, "train_cv2.txt": 2198, "train_cv3.txt": 2198,
    "train_cv4.txt": 2198, "train_cv5.txt": 2198,
    "val_cv1.txt": 315, "val_cv2.txt": 315, "val_cv3.txt": 315,
    "val_cv4.txt": 315, "val_cv5.txt": 315,
}
BLOCKED_SUFFIXES = {".pth", ".pt", ".ckpt", ".xlsx", ".pyc", ".pyo"}
BLOCKED_DIR_PARTS = {"__pycache__", "analysis_test", "artifacts", "archives",
                     "output", "analysis_graph", "ablation"}
BLOCKED_FILES = {Path("scripts/pipeline.env"),
                 Path("scripts/raw_pdb_pipeline.env"),
                 Path("scripts/raw_pdb_pipeline.server.env")}
PRIVATE_MARKERS = ("CCC_" + "server_workspace", "NAcontact/" + "BBB")
LARGE_FILE_LIMIT_BYTES = 25 * 1024 * 1024

def count_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())

def files() -> Iterable[Path]:
    for path in Path(".").rglob("*"):
        if ".git" not in path.parts and (path.is_file() or path.is_symlink()):
            yield path

def validate_metadata() -> bool:
    failed = False
    for path in REQUIRED_METADATA:
        if not path.exists():
            print("Missing metadata:", path)
            failed = True
    split_dir = Path("data/split_data/cv")
    for name, expected in EXPECTED_SPLITS.items():
        path = split_dir / name
        if not path.exists():
            print("Missing split:", path)
            failed = True
        elif count_lines(path) != expected:
            print("Split count mismatch:", path, count_lines(path), expected)
            failed = True
    labels = Path("data/affinity_data/all_data_with_multi_labels.json")
    if labels.exists():
        try:
            value = json.loads(labels.read_text(encoding="utf-8"))
            if len(value) != 3327:
                print("Label count mismatch:", len(value), 3327)
                failed = True
            pair_ids = set()
            for dataset in ("ppi", "aai", "tcr-pmhc"):
                pair_path = Path("data/PPB-Affinity") / f"{dataset}_pair.txt"
                if pair_path.exists():
                    pair_ids.update(
                        line.strip()
                        for line in pair_path.read_text(
                            encoding="utf-8", errors="ignore"
                        ).splitlines()
                        if line.strip()
                    )
            if pair_ids != set(value):
                print("Pair-list and label identifiers do not match.")
                failed = True
        except Exception as exc:
            print("Invalid label JSON:", exc)
            failed = True
    return not failed

def validate_github() -> bool:
    failed = False
    for path in REQUIRED_DOCS + REQUIRED_CODE:
        if not path.exists():
            print("Missing public file:", path)
            failed = True
    if not validate_metadata():
        failed = True
    for path in files():
        if path == Path("scripts/validate_public_release.py"):
            continue
        if path.is_symlink():
            print("Symbolic link is not allowed:", path)
            failed = True
            continue
        if any(part in BLOCKED_DIR_PARTS for part in path.parts):
            print("Generated/runtime path is not allowed:", path)
            failed = True
        if path in BLOCKED_FILES or path.suffix in BLOCKED_SUFFIXES:
            print("Generated/local file is not allowed:", path)
            failed = True
        try:
            if path.stat().st_size > LARGE_FILE_LIMIT_BYTES:
                print("File is larger than 25 MB:", path)
                failed = True
            if path.suffix in {".py", ".sh", ".env", ".md", ".cff"}:
                text = path.read_text(encoding="utf-8", errors="ignore")
                for marker in PRIVATE_MARKERS:
                    if marker in text:
                        print("Private path marker", marker, "found in", path)
                        failed = True
        except OSError as exc:
            print("Cannot inspect", path, exc)
            failed = True
    return not failed

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--github", action="store_true")
    args = parser.parse_args()
    ok = validate_github() if args.github else validate_metadata()
    print("Validation passed." if ok else "Validation failed.")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
