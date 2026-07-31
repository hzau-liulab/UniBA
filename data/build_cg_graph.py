import argparse
import os
import sys
os.environ["PATH"] += os.pathsep + os.path.expanduser("~/Software/HMMER/bin")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import numpy as np
import torch
import MDAnalysis as mda
import utils.cg_protein as cg_protein
# from utils.cg_graphconstruct import *


def write_clean_pdb(selection, out_path):
    selection.write(out_path)
    with open(out_path, "r") as fl:
        lines = fl.readlines()
    clean = []
    for line in lines:
        if line.startswith(("HEADER", "TITLE", "CRYST1", "REMARK")):
            continue
        if line.startswith(("ATOM", "HETATM")):
            # # 插入码判断：line[26] 非空即为插入码
            # if line[26].strip():
            #     continue
            clean.append(line)
    with open(out_path, "w") as fo:
        fo.writelines(clean)


def build_cg_graph(pair_list, project_root, data_type, mode='intra'):
    affinity_dir = os.path.join(project_root, "data", "affinity_data")
    for i, pair in enumerate(pair_list):
        # pair = '5nx1_A_C'
        print(i, pair)
        parts = pair.split('.', 1)
        complex_id = parts[0]
        mut_info = parts[1] if len(parts) > 1 else None
        mut_chains = list({mut[1] for mut in mut_info.split('_')}) if mut_info else []

        pdb, chain1, chain2 = complex_id.split('_')
        chains = {'pc1': list(chain1), 'pc2': list(chain2)}

        if pair in ['6ysq_AC_G', '7bw4_A_BCD']:
            af_chains = {'pc1': ['A'], 'pc2': ['B']}
        else:
            all_input_chains = list(chain1) + list(chain2)
            af3_chain_ids = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            chain_map = {
                old: new
                for old, new in zip(all_input_chains, af3_chain_ids)
            }
            chain1_af3 = ''.join(chain_map[c] for c in chain1)
            chain2_af3 = ''.join(chain_map[c] for c in chain2)
            af_chains = {'pc1': chain1_af3, 'pc2': chain2_af3}

        cg_file = os.path.join(affinity_dir, "cg_input", f"cg_{data_type}", pair)

        graph_dir = os.path.join(affinity_dir, f"cg_{mode}_graph", data_type)
        os.makedirs(graph_dir, exist_ok=True)

        if mode == 'inter':
            graph_file = f"{graph_dir}/{pair}.gh"
            if not os.path.exists(graph_file):
                complete_check, protein = cg_protein.CG22_Protein.from_cg_molecule(cg_file, af_chains)
                protein, closest_contact_distance = protein.protein_cropping()
                if complete_check:
                    torch.save(protein, graph_file)
                    print(f"saved new graph: {graph_file}")
                else:
                    print(f"pair incomplete, skip saving {graph_file}")
                    continue
            else:
                # cg_gh = torch.load(graph_file)
                continue

        elif mode == 'intra':
            graph_files = {
                tag: (
                    f"{graph_dir}/{pdb}_{''.join(chain_group)}"
                    f"{'_' + mut_info if mut_info else ''}.gh")
                for tag, chain_group in chains.items()
            }

            pc1_graph_file, pc2_graph_file = graph_files["pc1"], graph_files["pc2"]

            # pc1_graph_file = f"./affinity_data/cg_intra_graph/{data_type}/{pdb}_{chain1}_{mut_info}.gh"
            # pc2_graph_file = f"./affinity_data/cg_intra_graph/{data_type}/{pdb}_{chain2}_{mut_info}.gh"

            # --- 检查存在性 ---
            # seq_feat_file = f'../feature/seq_features/{data_type}/{pair}.pkl'
            # with open(seq_feat_file, "rb") as f:
            #     seq_feat = pickle.load(f)

            exists1, exists2 = os.path.exists(pc1_graph_file), os.path.exists(pc2_graph_file)
            if exists1 and exists2:
                # for i, graph_file in enumerate([pc1_graph_file, pc2_graph_file]):
                #     # chain_prefix = f"chain{i+1}"
                #     with open(graph_file, 'rb') as f:
                #         graph = pickle.load(f)
                #     # seq = seq_feat[f"{chain_prefix}_seqcoding"]
                #     # graph.seq_feat = torch.as_tensor(seq, dtype=torch.float32)
                #     torch.save(graph, graph_file)
                    # with open(graph_file, 'wb') as f:
                    #     pickle.dump(graph, f)
                print(f"Skip both existing graphs: {os.path.basename(pc1_graph_file)}, {os.path.basename(pc2_graph_file)}")
                continue

            u = mda.Universe(f"{cg_file}/cg_M2.pdb")

            for segid in set(u.atoms.segids):
                path = os.path.join(cg_file, f"cg_{segid}_M2.pdb")
                if os.path.exists(path):
                    os.remove(path)

            pc1_path = os.path.join(cg_file, "cg_pc1_M2.pdb")
            pc2_path = os.path.join(cg_file, "cg_pc2_M2.pdb")

            for path in [pc1_path, pc2_path]:
                if os.path.exists(path):
                    os.remove(path)

            pc1_sel = u.select_atoms(" or ".join([f"segid {seg}" for seg in af_chains['pc1']]))
            pc2_sel = u.select_atoms(" or ".join([f"segid {seg}" for seg in af_chains['pc2']]))

            write_clean_pdb(pc1_sel, pc1_path)
            write_clean_pdb(pc2_sel, pc2_path)

            # --- 构建 CG Graphs ---
            if not exists1:
                pc1_complete, pc1_protein = cg_protein.CG22_Protein.from_cg_molecule(cg_file, af_chains, 'pc1')
                if pc1_complete:
                    torch.save(pc1_protein, pc1_graph_file)
                    # with open(pc1_graph_file, 'wb') as f:
                    #     pickle.dump(pc1_protein, f)
                    print(f"saved new graph: {pc1_graph_file}")
                else:
                    print(f"pc1 incomplete, skip saving {pc1_graph_file}")

            if not exists2:
                pc2_complete, pc2_protein = cg_protein.CG22_Protein.from_cg_molecule(cg_file, af_chains, 'pc2')
                if pc2_complete:
                    torch.save(pc2_protein, pc2_graph_file)
                    # with open(pc2_graph_file, 'wb') as f:
                    #     pickle.dump(pc2_protein, f)
                    print(f"saved new graph: {pc2_graph_file}")
                else:
                    print(f"pc2 incomplete, skip saving {pc2_graph_file}")


# 构建三类数据类型的单链intra(CG)图 #
# /home/yyShen/NAcontact/data/affinity_data/cg_graph/aai_mut/
def parse_args():
    parser = argparse.ArgumentParser(description="Build coarse-grained graphs for a prepared pair list.")
    parser.add_argument("--data-type", default="ppi", help="Dataset name")
    parser.add_argument("--pair-list", default="", help="Optional pair-list path")
    parser.add_argument("--mode", default="inter", choices=["inter", "intra"], help="Graph mode")
    parser.add_argument(
        "--project-root",
        default=str(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))),
        help="Project root",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_type = args.data_type
    project_root = os.path.abspath(args.project_root)
    pair_list_file = args.pair_list or os.path.join(project_root, "data", "PPB-Affinity", f"{data_type}_pair.txt")
    pair_list = np.loadtxt(pair_list_file, dtype=str)
    if getattr(pair_list, "ndim", 1) == 0:
        pair_list = np.array([str(pair_list)])
    build_cg_graph(pair_list, project_root, data_type, args.mode)
