#!/usr/bin/env python3
"""Public CLI wrapper around the legacy sequence/structure feature generator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--data-type", required=True, help="Dataset name, e.g. ppi,aai,tcr-pmhc")
    parser.add_argument("--action", choices=["generate", "normalize", "all"], default="all")
    parser.add_argument("--esm-path", default="", help="Torch/ESM cache or checkpoint path")
    parser.add_argument("--dssp", default="", help="Path to mkdssp")
    parser.add_argument("--ghecom", default="", help="Path to ghecom")
    parser.add_argument("--psaia", default="", help="Path to PSAIA")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    feature_dir = project_root / "feature"
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(feature_dir))
    os.chdir(feature_dir)

    import get_seq_str_feature as legacy  # noqa: PLC0415

    legacy.args = SimpleNamespace(
        esm_path=args.esm_path,
        dssp=args.dssp,
        ghecom=args.ghecom,
        psaia=args.psaia,
    )

    label_json = project_root / "data" / "affinity_data" / "all_data_with_multi_labels.json"
    data_dict = json.loads(label_json.read_text(encoding="utf-8"))

    if args.action in {"generate", "all"}:
        legacy.main(data_dict, args.data_type)
    if args.action in {"normalize", "all"}:
        legacy.hand_feat_scaler(data_dict, args.data_type)
        legacy.rename_hand_str_keys(args.data_type)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
