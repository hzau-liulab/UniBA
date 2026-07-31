#!/usr/bin/env python3
"""Prepare Martini 2.2/Gromacs inputs required by UniBA energy features."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path


ASSET_NAMES = (
    "martini_v2.2.itp",
    "martini_v2.0_ions.itp",
    "water.gro",
    "minim.mdp",
    "min_steep.mdp",
    "min_cg.mdp",
)


def run(command: list[str], cwd: Path, stdin: str | None = None) -> None:
    print(f"[run] {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, input=stdin, text=True, check=True)


def read_pairs(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]


def prepare_topology(topology: Path) -> None:
    text = topology.read_text(encoding="utf-8")
    old_include = '#include "martini.itp"'
    new_include = '\n'.join(
        ('#include "martini_v2.2.itp"', '#include "martini_v2.0_ions.itp"')
    )
    if old_include not in text:
        raise RuntimeError(f"Missing {old_include} in {topology}")
    topology.write_text(text.replace(old_include, new_include).rstrip() + "\n", encoding="utf-8")


def prepare_pair(args: argparse.Namespace, pair: str) -> None:
    root = args.project_root
    raw_pdb = root / "data" / "pdb_files" / f"{args.data_type}_mapped" / f"{pair}.pdb"
    output_root = root / "data" / "affinity_data" / "cg_input" / f"cg_{args.data_type}"
    output_dir = output_root / pair
    complex_id = pair.split(".", 1)[0]
    try:
        _, partner_1, partner_2 = complex_id.split("_")
    except ValueError as exc:
        raise ValueError(f"Invalid pair identifier: {pair}") from exc
    mapped_chains = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(partner_1) + len(partner_2)]
    required_outputs = (
        output_dir / "em.gro",
        output_dir / "em.tpr",
        output_dir / "cg_M2.pdb",
        *(output_dir / f"Protein_{chain}.itp" for chain in mapped_chains),
    )

    if all(path.is_file() and path.stat().st_size > 0 for path in required_outputs):
        print(f"[skip complete] {pair}")
        return
    if not raw_pdb.is_file():
        raise FileNotFoundError(
            f"Missing mapped complex PDB: {raw_pdb}. Run the data preparation "
            "stage before Martini/Gromacs preparation."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"Incomplete output exists: {output_dir}; inspect it or rerun with --overwrite"
            )
        resolved_output = output_dir.resolve()
        if output_root.resolve() not in resolved_output.parents:
            raise RuntimeError(f"Refusing to remove unexpected path: {resolved_output}")
        shutil.rmtree(resolved_output)
    output_dir.mkdir(parents=True)

    run(
        [
            str(args.python2),
            str(args.martinize_script),
            "-f",
            str(raw_pdb),
            "-o",
            "cg_M2.top",
            "-x",
            "cg_M2.pdb",
            "-dssp",
            str(args.dssp),
            "-p",
            "Backbone",
            "-ff",
            "martini22",
            "-v",
        ],
        output_dir,
    )

    asset_dir = root / "data" / "affinity_data" / "cg_input"
    for name in ASSET_NAMES:
        source = asset_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing Martini asset: {source}")
        shutil.copy2(source, output_dir / name)

    prepare_topology(output_dir / "cg_M2.top")
    gmx = str(args.gmx)
    run([gmx, "editconf", "-f", "cg_M2.pdb", "-o", "complex_box.gro", "-c", "-d", "1.2", "-bt", "triclinic"], output_dir)
    run([gmx, "solvate", "-cp", "complex_box.gro", "-cs", "water.gro", "-p", "cg_M2.top", "-o", "protein_sol.gro", "-radius", "0.25"], output_dir)
    run([gmx, "grompp", "-f", "minim.mdp", "-c", "protein_sol.gro", "-p", "cg_M2.top", "-o", "ions.tpr", "-maxwarn", "1"], output_dir)
    run([gmx, "genion", "-s", "ions.tpr", "-p", "cg_M2.top", "-o", "protein_ion.gro", "-pname", "NA+", "-nname", "CL-", "-neutral", "-seed", "3407"], output_dir, stdin="W\n")
    run([gmx, "grompp", "-f", "min_steep.mdp", "-c", "protein_ion.gro", "-p", "cg_M2.top", "-o", "min_steep.tpr", "-maxwarn", "2"], output_dir)
    run([gmx, "mdrun", "-deffnm", "min_steep", "-nt", str(args.threads)], output_dir)
    run([gmx, "grompp", "-f", "min_cg.mdp", "-c", "min_steep.gro", "-p", "cg_M2.top", "-o", "em.tpr", "-maxwarn", "2"], output_dir)
    try:
        run([gmx, "mdrun", "-deffnm", "em", "-nt", str(args.threads)], output_dir)
    except subprocess.CalledProcessError:
        print(
            "[warn] Conjugate-gradient minimization failed; retrying with "
            "steepest descent.",
            flush=True,
        )
        run(
            [
                gmx,
                "grompp",
                "-f",
                "min_steep.mdp",
                "-c",
                "min_steep.gro",
                "-p",
                "cg_M2.top",
                "-o",
                "em_fallback.tpr",
                "-maxwarn",
                "2",
            ],
            output_dir,
        )
        run(
            [gmx, "mdrun", "-deffnm", "em_fallback", "-nt", str(args.threads)],
            output_dir,
        )
        shutil.copy2(output_dir / "em_fallback.tpr", output_dir / "em.tpr")
        shutil.copy2(output_dir / "em_fallback.gro", output_dir / "em.gro")
        (output_dir / "MINIMIZATION_FALLBACK.txt").write_text(
            "Conjugate-gradient minimization failed; em.tpr and em.gro were "
            "generated by a second steepest-descent minimization.\n",
            encoding="utf-8",
        )

    for path in required_outputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected Gromacs output was not created: {path}")
    print(f"[done] {pair}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-type", required=True)
    parser.add_argument("--pair-list", type=Path, default=None)
    parser.add_argument("--martinize-script", type=Path, required=True)
    parser.add_argument("--python2", type=Path, required=True)
    parser.add_argument("--dssp", type=Path, required=True)
    parser.add_argument("--gmx", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    if args.pair_list is None:
        args.pair_list = args.project_root / "data" / "PPB-Affinity" / f"{args.data_type}_pair.txt"
    for name in ("martinize_script", "python2", "dssp", "gmx", "pair_list"):
        path = getattr(args, name)
        if not path.exists():
            parser.error(f"--{name.replace('_', '-')} path does not exist: {path}")
    return args


def main() -> int:
    args = parse_args()
    pairs = read_pairs(args.pair_list)
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    for pair in pairs:
        prepare_pair(args, pair)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
