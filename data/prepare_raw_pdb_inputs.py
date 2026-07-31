#!/usr/bin/env python3
"""Prepare per-chain PDB and FASTA files from raw complex PDB files.

Input:
  data/pdb_files/<dataset>/<pair>.pdb
  data/PPB-Affinity/<dataset>_pair.txt

Output:
  data/pdb_files/<dataset>_pc/<pdb>_<mapped_chain>.pdb
  data/fasta/<dataset>_pc_fasta/<pdb>_<mapped_chain>.fasta

Pair ids use PDBID_partner1_partner2. Chains are mapped in partner order to
A, B, C... to match the feature-generation code path used by UniBA.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from Bio.PDB import PDBIO, PDBParser, Select
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from Bio.PDB.Polypeptide import protein_letters_3to1


AA3_TO_1 = {key.upper(): value for key, value in protein_letters_3to1.items()}


def parse_datasets(value: str) -> list[str]:
    return [item.strip() for item in value.replace(" ", ",").split(",") if item.strip()]


def read_pairs(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]


class ProteinSelect(Select):
    def accept_residue(self, residue) -> bool:
        return residue.id[0] == " "

    def accept_atom(self, atom) -> bool:
        return not atom.is_disordered() or atom.get_altloc() in ("A", " ")


def chain_sequence(chain) -> str:
    seq: list[str] = []
    seen: set[tuple] = set()
    for residue in chain.get_residues():
        if residue.id[0] != " ":
            continue
        key = residue.id
        if key in seen:
            continue
        seen.add(key)
        seq.append(AA3_TO_1.get(residue.resname.upper(), "X"))
    return "".join(seq)


def resolve_source_ids(structure, input_chains: list[str], mapped_ids: list[str]) -> list[str]:
    available = {chain.id for chain in structure.get_chains()}
    if set(input_chains).issubset(available):
        return input_chains
    if set(mapped_ids).issubset(available):
        return mapped_ids
    return [old if old in available else mapped for old, mapped in zip(input_chains, mapped_ids)]


def copy_mapped_structure(structure, source_ids: list[str], mapped_ids: list[str], structure_id: str):
    source_chains = {chain.id: chain for chain in structure.get_chains()}
    output = Structure(structure_id)
    model = Model(0)
    output.add(model)
    missing: list[str] = []
    for source_id, mapped_id in zip(source_ids, mapped_ids):
        chain = source_chains.get(source_id)
        if chain is None:
            missing.append(source_id)
            continue
        copied_chain = chain.copy()
        copied_chain.id = mapped_id
        model.add(copied_chain)
    return output, missing


def prepare_pair(parser: PDBParser, pair: str, dataset: str, root: Path, overwrite: bool) -> tuple[int, int]:
    complex_id = pair.split(".", 1)[0]
    pdb_id, partner1, partner2 = complex_id.split("_")
    all_input_chains = list(partner1) + list(partner2)
    mapped_ids = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(all_input_chains)])

    pdb_file = root / "data" / "pdb_files" / dataset / f"{pair}.pdb"
    if not pdb_file.exists():
        print(f"[missing pdb] {pdb_file}")
        return 0, 1

    out_pdb_dir = root / "data" / "pdb_files" / f"{dataset}_pc"
    out_fasta_dir = root / "data" / "fasta" / f"{dataset}_pc_fasta"
    out_pdb_dir.mkdir(parents=True, exist_ok=True)
    out_fasta_dir.mkdir(parents=True, exist_ok=True)

    structure = parser.get_structure(pair, str(pdb_file))
    source_ids = resolve_source_ids(structure, all_input_chains, mapped_ids)
    mapped_structure, missing_ids = copy_mapped_structure(
        structure, source_ids, mapped_ids, f"{pair}_mapped"
    )
    if missing_ids:
        print(f"[missing chains] {pair}: {', '.join(missing_ids)}")

    written = 0
    missing = len(missing_ids)
    source_chains = {chain.id: chain for chain in structure.get_chains()}
    for source_id, mapped_chain in zip(source_ids, mapped_ids):
        chain = source_chains.get(source_id)
        if chain is None:
            continue

        name = f"{pdb_id}_{mapped_chain}"
        pdb_out = out_pdb_dir / f"{name}.pdb"
        fasta_out = out_fasta_dir / f"{name}.fasta"
        if overwrite or not pdb_out.exists():
            single_structure, _ = copy_mapped_structure(
                structure, [source_id], [mapped_chain], f"{pair}_{mapped_chain}"
            )
            io = PDBIO()
            io.set_structure(single_structure)
            io.save(str(pdb_out), ProteinSelect())
        if overwrite or not fasta_out.exists():
            seq = chain_sequence(chain)
            fasta_out.write_text(f">{name}\n{seq}\n", encoding="utf-8")
        written += 1

    mapped_dir = root / "data" / "pdb_files" / f"{dataset}_mapped"
    mapped_dir.mkdir(parents=True, exist_ok=True)
    mapped_out = mapped_dir / f"{pair}.pdb"
    if overwrite or not mapped_out.exists():
        io = PDBIO()
        io.set_structure(mapped_structure)
        io.save(str(mapped_out), ProteinSelect())

    return written, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="Repository root")
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset names")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-chain files")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    pdb_parser = PDBParser(QUIET=True)
    total_written = 0
    total_missing = 0

    for dataset in parse_datasets(args.datasets):
        pair_file = root / "data" / "PPB-Affinity" / f"{dataset}_pair.txt"
        pairs = read_pairs(pair_file)
        print(f"[dataset] {dataset}: {len(pairs)} pairs")
        for pair in pairs:
            written, missing = prepare_pair(pdb_parser, pair, dataset, root, args.overwrite)
            total_written += written
            total_missing += missing

    print(f"[done] wrote/checked {total_written} chain files; missing chains/PDBs: {total_missing}")
    return 1 if total_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
