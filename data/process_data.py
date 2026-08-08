import sys
import os
import random
proj_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(proj_dir)
from utils.pdb_utils import *
import numpy as np
from arg_parse import *
from openmm.app import PDBFile
import subprocess
from Bio.Data.IUPACData import protein_letters_3to1
import MDAnalysis as mda


def read_chain_block(pdb, pdb_dir, chain_list, mut_chain=None, mutation=None):
    blocks = []
    for ch in chain_list:
        suffix = f".{mutation}" if (mutation and ch == mut_chain) else ""
        pdb_file = os.path.join(pdb_dir, f"{pdb}_{ch}{suffix}.pdb")
        if not os.path.exists(pdb_file):
            print(f"Missing single-chain pdb for : {pdb_file}")
            return None
        with open(pdb_file, "r") as fb:
            blocks.append(fb.read().strip())

    return "\n".join(blocks) if blocks else None


def convert_3to1(resname):
    try:
        return protein_letters_3to1[resname.capitalize()]
    except KeyError:
        return 'X'


def get_chain_seq(u, pdb, chain, out_dir, suffix=''):
    if chain is None:
        return ''

    chain_res = u.select_atoms(f"chainID {chain}").residues
    sequence = ''.join([convert_3to1(res.resname) for res in chain_res])

    fasta_file = os.path.join(out_dir, f"{pdb}_{chain}{suffix}.fasta")
    fasta_header = f">{pdb}_{chain}{suffix}"

    if os.path.exists(fasta_file):
        with open(fasta_file, 'r') as fa:
            fasta_sequence = fa.readlines()[1].strip()
            if fasta_sequence != sequence:
                print(f"Warning: {fasta_file} mismatch, updating...")
                with open(fasta_file, 'w') as fa:
                    fa.write(f"{fasta_header}\n{sequence}")
                return sequence
            else:
                return fasta_sequence
    else:
        with open(fasta_file, 'w') as fa:
            fa.write(f"{fasta_header}\n{sequence}")
        return sequence


pair_list = np.loadtxt(f'./train_pair.txt', dtype=str)
pdb_dir = f"./pdb_files/complex"
pdb1_dir = f"./pdb_files/component1"
pdb2_dir = f"./pdb_files/component2"
fasta_dir = f"./fasta/chains"
fasta1_dir = f"./fasta/component1"
fasta2_dir = f"./fasta/component2"

for i, pair in enumerate(pair_list[:1]):
    print(i, pair)
    parts = pair.split('.', 1)
    complex_id = parts[0]
    mut_info = parts[1] if len(parts) > 1 else None
    mut_chains = list({mut[1] for mut in mut_info.split('_')}) if mut_info else []

    pdb, chain1, chain2 = complex_id.split('_')
    complex_pdb = f'{pdb_dir}/{pdb}.pdb'
    u = mda.Universe(complex_pdb)
    chain1_list, chain2_list = list(chain1), list(chain2)
    seq1, seq2 = '', ''

    suffix = f".{mut_info}" if mut_info else ""
    pc1 = f'{pdb}_{"".join(chain1_list)}{suffix}'
    pc2 = f'{pdb}_{"".join(chain2_list)}{suffix}'

    for chain in chain1_list:
        chain_suffix = f".{mut_info}" if (mut_info and chain in mut_chains) else ""
        seq1 += get_chain_seq(u, pdb, chain, fasta_dir, chain_suffix)
        pdb1_fuc = PDBfuc(complex_pdb, chain, pdb1_dir)

    for chain in chain2_list:
        chain_suffix = f".{mut_info}" if (mut_info and chain in mut_chains) else ""
        seq2 += get_chain_seq(u, pdb, chain, fasta_dir, chain_suffix)
        pdb2_fuc = PDBfuc(complex_pdb, chain, pdb2_dir)

    with open(f'{fasta1_dir}/{pc1}.fasta', 'w') as fa:
        fa.write(f">{pc1}\n{seq1}\n")

    with open(f'{fasta2_dir}/{pc2}.fasta', 'w') as fa:
        fa.write(f">{pc2}\n{seq2}\n")

    pdb12_file = os.path.join(pdb_dir, f"{pdb}_{chain1}_{chain2}{suffix}.pdb")
    if os.path.exists(pdb12_file):
        continue

    component1 = read_chain_block(pdb, pdb1_dir, chain1_list, mut_chains, mut_info)
    component2 = read_chain_block(pdb, pdb2_dir, chain2_list, mut_chains, mut_info)
    if component1 and component2:
        with open(pdb12_file, 'w') as fu:
            fu.write(component1.strip() + "\n" + component2.strip() + "\nEND\n")
        print(f"[OK] Combined components: {pdb12_file}")


# 6. 将原子级 PDB结构 转换成 MARTINI粗粒化模型
# 注意 使用pdbfixer 修复后的WT/MUT结构，再交给 MARTINI
# cd /home/data/yyShen/UniBA/data/cg_input/
# chmod +x martini_steps.sh
# ./martini_steps.sh > cg_sh.out 2>&1 &



