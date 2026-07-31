import argparse
import os
import shutil

os.environ["PATH"] += os.pathsep + os.path.expanduser("~/Software/HMMER/bin")
import torch
import pickle
import numpy as np
import MDAnalysis as mda
# import torch_geometric
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import json
import glob
from Bio.Data.IUPACData import protein_letters_3to1
import re
import csv
from scipy.spatial.distance import cdist


def run_anarci_numbering(sequences, scheme):
    """Import ANARCI only for antibody/TCR datasets that need numbering."""
    try:
        from anarci import run_anarci
    except ImportError as exc:
        raise RuntimeError(
            "ANARCI is required only for automatic CDR numbering of AAI/TCR data. "
            "Install the optional ANARCI environment before building those datasets."
        ) from exc
    return run_anarci(sequences, scheme=scheme)


def get_chain_seq(u, pdb, chain, out_dir, mut_chains=None, mut_info=None):
    def convert_3to1(resname):
        try:
            return protein_letters_3to1[resname.capitalize()]
        except KeyError:
            return ''

    if chain is None:
        return '', None

    chain_res = u.select_atoms(f"chainID {chain}").residues
    sequence = ''.join([convert_3to1(res.resname) for res in chain_res])

    # 关键点：统一用 suffix
    suffix = f".{mut_info}" if (mut_info and chain in mut_chains) else ""
    fasta_file = os.path.join(out_dir, f"{pdb}_{chain}{suffix}.fasta")
    fasta_header = f">{pdb}_{chain}{suffix}"

    if os.path.exists(fasta_file):
        with open(fasta_file, 'r') as fa:
            fasta_sequence = fa.readlines()[1].strip()
            if fasta_sequence != sequence:
                print(f"Warning: {fasta_file} mismatch, updating...")
                with open(fasta_file, 'w') as fa:
                    fa.write(f"{fasta_header}\n{sequence}")
                return sequence, chain_res
            else:
                return fasta_sequence, chain_res
    else:
        with open(fasta_file, 'w') as fa:
            fa.write(f"{fasta_header}\n{sequence}")
        return sequence, chain_res


def compute_dist_map(target_res, ligand_res, dist_map_file):
    all_residues = list(target_res) + list(ligand_res)
    num_residues = len(all_residues)

    if os.path.exists(dist_map_file):
        res_dist_map = np.load(dist_map_file)
        if res_dist_map.shape != (num_residues, num_residues):
            need_recompute = True
        else:
            return res_dist_map
    else:
        need_recompute = True
        os.makedirs(os.path.dirname(dist_map_file), exist_ok=True)
        # cdr_ag_distance = aa_square_distance[np.ix_(cdr_indices, range(target_len, num_residues))]

    if need_recompute:
        positions_by_res = [res.atoms.positions for res in all_residues]
        # num_residues = len(positions_by_res)
        res_dist_map = np.full((num_residues, num_residues), np.inf)

        for m in range(num_residues):
            for n in range(m + 1, num_residues):
                dist_matrix = np.linalg.norm(positions_by_res[m][:, None, :] - positions_by_res[n][None, :, :], axis=-1)
                min_dist = np.min(dist_matrix)
                res_dist_map[m, n] = res_dist_map[n, m] = min_dist

        # # 只计算 CDR × AG 的最小原子距离矩阵
        # cdr_ag_distance = np.full((len(cdr_indices), ligand_len), np.inf)
        # for i, ab_idx in enumerate(cdr_indices):
        #     ab_atoms = target_res[ab_idx].atoms.positions
        #     for j, ag_residue in enumerate(ligand_res):
        #         ag_atoms = ag_residue.atoms.positions
        #         min_dist = np.min(np.linalg.norm(ab_atoms[:, None] - ag_atoms[None, :], axis=-1))
        #         cdr_ag_distance[i, j] = min_dist

        # # 只计算PPI交互残基对的最小原子距离矩阵
        # n1 = len(prot1_res)
        # n2 = len(prot2_res)
        # dist_matrix = np.zeros((n1, n2), dtype=np.float32)
        # for i, res1 in enumerate(prot1_res):
        #     pos1 = res1.atoms.positions  # shape (m1, 3)
        #     for j, res2 in enumerate(prot2_res):
        #         pos2 = res2.atoms.positions  # shape (m2, 3)
        #         dists = cdist(pos1, pos2)
        #         dist_matrix[i, j] = np.min(dists)

        np.save(dist_map_file, res_dist_map)
        return res_dist_map


def load_energy_matrices(energy_path, energy_type=('coul', 'lj')):
    energy_matrices = {}

    for ef in energy_type:
        pattern = os.path.join(energy_path, f"{ef}_*_res.txt")
        files = glob.glob(pattern)
        if not files:
            continue
        for fpath in files:
            fname = os.path.basename(fpath)
            key = fname.replace(f"_res.txt", "")
            energy_matrices[key] = np.loadtxt(fpath)

    return energy_matrices


def zscore_normalize(edge_attr, norm_idx=(0, 1, 2), eps=1e-6):
    """
    norm_idx: 需要标准化的列索引（例如 coul, lj, dist）
    """
    x = edge_attr.clone()

    x_norm = x[:, norm_idx]

    mean = x_norm.mean(dim=0, keepdim=True)
    std = x_norm.std(dim=0, keepdim=True)

    std = torch.where(std < eps, torch.ones_like(std), std)

    x[:, norm_idx] = (x_norm - mean) / std

    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    return x


def detect_interaction_type(res_i, res_j, dist):
    """
    返回 interaction 类型 [H-bond, salt-bridge, hydrophobic, pi-pi] 独热编码
    """
    interaction = [0] * 4

    hbond_residues = {'SER', 'THR', 'TYR', 'ASN', 'GLN', 'ARG', 'ASP', 'GLU', 'HIS', 'LYS'}
    charged_pos = {'ARG', 'LYS', 'HIS'}
    charged_neg = {'ASP', 'GLU'}
    hydrophobic = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO'}
    aromatic = {'PHE', 'TYR', 'TRP', 'HIS'}

    name_i = res_i.resname
    name_j = res_j.resname

    # 氢键（极性残基，距离 < 3.5 Å）
    if name_i in hbond_residues and name_j in hbond_residues and dist <= 3.5:
        interaction[0] = 1

    # 盐桥（正负带电，距离 < 4.0 Å）
    if ((name_i in charged_pos and name_j in charged_neg) or
        (name_j in charged_pos and name_i in charged_neg)) and dist <= 4.0:
        interaction[1] = 1

    # 疏水相互作用（疏水残基，距离 < 5.0 Å）
    if name_i in hydrophobic and name_j in hydrophobic and dist <= 5.0:
        interaction[2] = 1

    # 芳香堆积（两个芳香残基，<6 Å）
    if name_i in aromatic and name_j in aromatic and dist <= 6.0:
        interaction[3] = 1

    return interaction  # 4D 0/1 vector


def compute_local_coord_system(residue):
    try:
        C = residue.atoms.select_atoms('name C')[0].position
        CA = residue.atoms.select_atoms('name CA')[0].position
        N = residue.atoms.select_atoms('name N')[0].position
    except IndexError:
        return None  # 结构不完整

    u = C - CA
    v = N - CA
    b = u / np.linalg.norm(u)
    n = np.cross(u, v)
    n = n / np.linalg.norm(n)
    t = np.cross(b, n)

    R = np.stack([b, n, t], axis=1)  # shape (3, 3)
    return R, CA  # 返回参考系和 CA 坐标


def compute_geom_feat(res_i, res_j):
    info_i = compute_local_coord_system(res_i)
    info_j = compute_local_coord_system(res_j)
    if info_i is None or info_j is None:
        return None  # 缺失关键原子，跳过

    Ri, CA_i = info_i
    Rj, CA_j = info_j

    v = CA_j - CA_i
    dist = np.linalg.norm(v)

    direction = Ri.T @ (v / dist)  # shape: (3,) 将单位向量 v/dist 从全局坐标系转换到局部坐标系 Ri 中
    orientation = Ri.T @ Rj  # shape: (3, 3) 将 Rj 表达成 Ri 局部系中的表示（相对方向）

    geom_feat = np.concatenate([direction.flatten(), orientation.flatten()])  # (3 + 9)
    return list(geom_feat)


def cal_edge_feat(
        local_i, local_j, res_i, res_j, dist,
        energy_matrices=None, energy_key=None, pair_interact_feat=None,
        edge_feat_type=('coul', 'lj', 'geom', 'interact'), reverse=False
):
    """
    计算边特征（能量 + 几何 + 交互），支持双向边。
    对于反向边，geom特征通过交换输入 res_i <-> res_j 自动反转方向向量和旋转矩阵。

    Args:
        local_i, local_j: 残基在各自链中的局部索引
        res_i, res_j: 残基列表
        energy_matrices: 包含能量矩阵的字典，键如 'coul_pc1_pc2', 'lj_pc1_pc2'
        energy_key: 能量矩阵对应键
        dist: 两残基的欧氏距离
        edge_feat_type: 特征类型 ('geom', 'coul', 'lj', 'interact')
        reverse: 是否计算反向边

    Returns:
        feat_tensor: torch.Tensor, shape (feat_dim,)
    """
    if reverse:
        res_i_feat = res_j[local_j]
        res_j_feat = res_i[local_i]
    else:
        res_i_feat = res_i[local_i]
        res_j_feat = res_j[local_j]

    feat_list = []

    if 'coul' in edge_feat_type:
        if energy_matrices is not None and energy_key is not None:
            energy_coul = energy_matrices.get(f"coul_{energy_key}", None)
            feat_list.append(float(energy_coul[local_i, local_j]) if energy_coul is not None else 0.0)
        else:
            feat_list.append(0.0)

    if 'lj' in edge_feat_type:
        if energy_matrices is not None and energy_key is not None:
            energy_lj = energy_matrices.get(f"lj_{energy_key}", None)
            feat_list.append(float(energy_lj[local_i, local_j]) if energy_lj is not None else 0.0)
        else:
            feat_list.append(0.0)

    if 'geom' in edge_feat_type:
        if dist is None:
            raise ValueError("geom feature requires `dist` to be provided")
        geom_feat = compute_geom_feat(res_i_feat, res_j_feat)
        feat_list.extend([float(dist)] + list(geom_feat))

    if 'interact' in edge_feat_type:
        if pair_interact_feat is not None:

            c1, r1 = res_i_feat.atoms.chainIDs[0], res_i_feat.resid
            c2, r2 = res_j_feat.atoms.chainIDs[0], res_j_feat.resid

            key = (c1, r1, c2, r2)
            rev_key = (c2, r2, c1, r1)

            interaction_feat = pair_interact_feat.get(key)
            if interaction_feat is None:
                interaction_feat = pair_interact_feat.get(rev_key, [0, 0, 0, 0, 0])

        else:
            interaction_feat = [0, 0, 0, 0, 0]

        # 保证类型统一
        interaction_feat = list(map(float, interaction_feat))

        feat_list.extend(interaction_feat)

    feat_tensor = torch.tensor(feat_list, dtype=torch.float)
    return feat_tensor


def get_node_features(pair, sel_pc1_idx, sel_pc2_idx, paths):
    if not os.path.exists(paths['seq_path']) or not os.path.exists(paths['str_path']):
        with open('../feature/check.txt', 'a') as fe:
            fe.write(f"{pair}\n")
        return None, None

    with open(paths['seq_path'], 'rb') as fq:
        seq_feat_dict = pickle.load(fq)
    with open(paths['str_path'], 'rb') as fr:
        str_feat_dict = pickle.load(fr)
    with open(paths['hand_str_path'], 'rb') as fd:
        hand_str_feat = pickle.load(fd)

    pc1_seq_feat = seq_feat_dict['chain1_rescoding']  # [num_pc1, 1280]
    pc2_seq_feat = seq_feat_dict['chain2_rescoding']  # [num_pc2, 1280]
    pc1_str_feat = str_feat_dict['chain1_esmif']   # [num_pc1, 512]
    pc2_str_feat = str_feat_dict['chain2_esmif']
    pc1_hand_str_feat = hand_str_feat['chain1_hand_str']       # [num_pc1, 18]
    pc2_hand_str_feat = hand_str_feat['chain2_hand_str']

    pc1_feat = np.concatenate([
        pc1_seq_feat[sel_pc1_idx],
        pc1_str_feat[sel_pc1_idx],
        pc1_hand_str_feat[sel_pc1_idx]
    ], axis=1)

    pc2_feat = np.concatenate([
        pc2_seq_feat[sel_pc2_idx],
        pc2_str_feat[sel_pc2_idx],
        pc2_hand_str_feat[sel_pc2_idx]
    ], axis=1)

    pc1_feat = torch.tensor(pc1_feat, dtype=torch.float)
    pc2_feat = torch.tensor(pc2_feat, dtype=torch.float)

    return pc1_feat, pc2_feat


def get_intra_edge_index(
        pc_res,
        sel_res_idx,
        full_aa_distance,
        num_pc1,
        intra_type='pc1',
        edge_feat_type=('coul', 'lj', 'geom', 'interact'),
        dist_cutoff=4.5,
        bidirectional=True,
        use_knn=False, k=6
):
    """
    在同一条链内部，为选中的残基构建 intra 边
    """
    edge_index, edge_attr, edge_type = [], [], []

    if len(sel_res_idx) <= 1:
        return [], None, [], set()

    if intra_type == "pc1":
        offset = 0
        aa_distance = full_aa_distance[:num_pc1, :num_pc1]
        etype = EDGE_TYPE["intra_pc1"]
    elif intra_type == "pc2":
        offset = num_pc1
        aa_distance = full_aa_distance[num_pc1:, num_pc1:]
        etype = EDGE_TYPE["intra_pc2"]
    else:
        raise ValueError(f"Unknown intra_type: {intra_type}")

    sel_res = sel_res_idx.copy()

    for ri in list(sel_res):
        dist_row = aa_distance[ri]

        # radius
        candidates = np.where(dist_row <= dist_cutoff)[0]
        candidates = candidates[candidates != ri]

        # top-k（可选）
        if use_knn and len(candidates) > k:
            sorted_idx = candidates[np.argsort(dist_row[candidates])]
            neighbors = sorted_idx[:k]
        else:
            neighbors = candidates

        for rj in neighbors:
            dist = dist_row[rj]

    # for i in range(len(sel_res)):
    #     ri = list(sel_res)[i]
    #     for rj in range(len(pc_res)):
    #         if ri == rj:
    #             continue
    #
    #         dist = aa_distance[ri, rj]
    #         if dist > dist_cutoff:
    #             continue

            gi = ri + offset
            gj = rj + offset

            feat = cal_edge_feat(ri, rj, pc_res, pc_res, dist, edge_feat_type=edge_feat_type, reverse=False)

            edge_index.append((gi, gj))
            edge_attr.append(feat)
            edge_type.append(etype)

            if bidirectional:
                rev_feat = cal_edge_feat(ri, rj, pc_res, pc_res, dist, edge_feat_type=edge_feat_type, reverse=True)
                edge_index.append((gj, gi))
                edge_attr.append(rev_feat)
                edge_type.append(etype)

            sel_res.add(rj)

    if len(edge_index) == 0:
        return [], None, [], set()

    edge_attr = torch.stack(edge_attr, dim=0)
    return edge_index, edge_attr, edge_type, sorted(sel_res)


# 构建PPI/TCR-pMHC的dist-based inter图 #

def get_inter_edge_index(
        pc1_res, pc2_res, paths, pair_interact_feat, use_knn=False, k=10,
        inter_edge_cutoff=10.0, inter_label_cutoff=4.5,
        edge_feat_type=('coul', 'lj', 'geom', 'interact'),
        normalize=True, bidirectional=True
):
    num_pc1 = pc2_offset = len(pc1_res)
    num_pc2 = len(pc2_res)

    energy_matrices = load_energy_matrices(paths['energy_path'])
    full_aa_distance = compute_dist_map(pc1_res, pc2_res, paths['dist_map'])
    aa_distance: np.ndarray = full_aa_distance[:num_pc1, num_pc1:]

    edge_index, edge_attr, edge_type = [], [], []
    selected_pc1, selected_pc2 = set(), set()
    pc1_bind_labels = [0] * num_pc1
    pc2_bind_labels = [0] * num_pc2

    energy_key = "pc1_pc2"
    inter_etype = EDGE_TYPE["inter"]

    for i_pc1 in range(num_pc1):
        dist_row = aa_distance[i_pc1]  # shape: (num_pc2, )
        thresh_pc2_indx = np.where(dist_row <= inter_edge_cutoff)[0]

        if len(thresh_pc2_indx) > 0:
            nearest_pc2 = thresh_pc2_indx
        elif use_knn:
            nearest_pc2 = np.argsort(dist_row)[:k]
        else:
            continue

        for j_pc2 in nearest_pc2:
            global_pc2_j = j_pc2 + pc2_offset
            dist = aa_distance[i_pc1, j_pc2]

            if dist <= inter_label_cutoff:
                pc1_bind_labels[i_pc1] = 1
                pc2_bind_labels[j_pc2] = 1

            feat = cal_edge_feat(i_pc1, j_pc2, pc1_res, pc2_res, dist, energy_matrices,
                                 energy_key, pair_interact_feat, edge_feat_type, reverse=False)
            edge_index.append((i_pc1, global_pc2_j))
            edge_attr.append(feat)
            edge_type.append(inter_etype)

            selected_pc1.add(i_pc1)
            selected_pc2.add(j_pc2)

            if bidirectional:
                rev_feat = cal_edge_feat(i_pc1, j_pc2, pc1_res, pc2_res, dist, energy_matrices,
                                         energy_key, pair_interact_feat, edge_feat_type, reverse=True)
                edge_index.append((global_pc2_j, i_pc1))
                edge_attr.append(rev_feat)
                edge_type.append(inter_etype)

    edge_attr = torch.stack(edge_attr, dim=0)  # (num_edges, feat_dim)
    if normalize:
        edge_attr = zscore_normalize(edge_attr, norm_idx=(0, 1, 2))
    #
    # selected_pc1 = sorted(selected_pc1)
    # selected_pc2 = sorted(selected_pc2)

    return edge_index, edge_attr, edge_type, \
           selected_pc1, selected_pc2, \
           np.array(pc1_bind_labels), np.array(pc2_bind_labels), full_aa_distance


def build_graph(pair, pc1_res, pc2_res, paths, pair_interact_feat, cdr_indices=None):

    num_pc1 = len(pc1_res)

    # # 1. 节点列表全局连续编号映射
    # res_type_map = {
    #     **{res.ix: 0 for res in pc1_res},
    #     **{res.ix: 1 for res in pc2_res}
    # }

    # ======================
    # 1. inter edges（全局索引）
    # ======================
    inter_edge_index, inter_edge_attr, inter_edge_type, \
    inter_pc1_idx, inter_pc2_idx, pc1_labels, pc2_labels, full_aa_distance = \
        get_inter_edge_index(
            pc1_res, pc2_res, paths, pair_interact_feat, inter_edge_cutoff=10.0)

    inter_pc1_idx = set(inter_pc1_idx)
    inter_pc2_idx = set(inter_pc2_idx)

    if len(inter_pc1_idx) == 0 or len(inter_pc2_idx) == 0:
        print(f"[Warning] No interface found in {pair}")

    # ======================
    # 2. seed = interface + CDR（只作用 pc1）
    # ======================
    pc1_seed = inter_pc1_idx.copy()
    if cdr_indices is not None:
        pc1_seed |= set(cdr_indices)

    pc2_seed = inter_pc2_idx.copy()

    # ======================
    # 3. intra edges（基于 interface 扩展）
    # ======================
    intra1_edge_index, intra1_edge_attr, intra1_edge_type, sel_pc1_idx = \
        get_intra_edge_index(
            pc1_res, pc1_seed, full_aa_distance, num_pc1,
            intra_type="pc1", dist_cutoff=4.0, use_knn=True)

    intra2_edge_index, intra2_edge_attr, intra2_edge_type, sel_pc2_idx = \
        get_intra_edge_index(
            pc2_res, pc2_seed, full_aa_distance, num_pc1,
            intra_type="pc2", dist_cutoff=4.0, use_knn=True)

    sel_pc1_idx = set(sel_pc1_idx)
    sel_pc2_idx = set(sel_pc2_idx)

    # ======================
    # 4. 构建 node 集合（全局索引）
    # ======================
    pc1_nodes = sorted(sel_pc1_idx)
    pc2_nodes = sorted(sel_pc2_idx)

    pc2_nodes_global = [i + num_pc1 for i in pc2_nodes]

    node_global_index = pc1_nodes + pc2_nodes_global
    node_set = set(node_global_index)

    # ======================
    # 5. 合并所有 edge
    # ======================
    edge_index = inter_edge_index + intra1_edge_index + intra2_edge_index
    edge_attr = torch.cat(
        [inter_edge_attr, intra1_edge_attr, intra2_edge_attr],
        dim=0
    )
    edge_type = torch.tensor(
        inter_edge_type + intra1_edge_type + intra2_edge_type,
        dtype=torch.long
    )

    for s, t in edge_index:
        assert s in node_set and t in node_set

    # ======================
    # 6. 构建 index_map（统一编号edge_index）
    # ======================
    index_map = {old: i for i, old in enumerate(node_global_index)}

    edge_index = torch.tensor(
        [(index_map[s], index_map[t]) for s, t in edge_index],
        dtype=torch.long).t().contiguous()

    node_global_index = torch.tensor(node_global_index, dtype=torch.long)

    # ======================
    # 7. node features
    # ======================
    pc1_feat, pc2_feat = get_node_features(pair, pc1_nodes, pc2_nodes, paths)

    node_feat = torch.cat([pc1_feat, pc2_feat], dim=0)

    # ======================
    # 8. node prior
    # ======================
    N = node_global_index.shape[0]

    is_cdr = torch.zeros(N)
    is_inter10 = torch.zeros(N)
    is_inter4 = torch.zeros(N)

    pc1_mask = node_global_index < num_pc1
    pc2_mask = ~pc1_mask

    # --- CDR
    if cdr_indices is not None:
        cdr_tensor = torch.tensor(list(cdr_indices), dtype=torch.long)
        is_cdr[pc1_mask] = torch.isin(
            node_global_index[pc1_mask], cdr_tensor
        ).float()

    # --- interface 10Å
    inter_pc1_tensor = torch.tensor(list(inter_pc1_idx))
    inter_pc2_tensor = torch.tensor(list(inter_pc2_idx))

    is_inter10[pc1_mask] = torch.isin(
        node_global_index[pc1_mask], inter_pc1_tensor
    ).float()

    is_inter10[pc2_mask] = torch.isin(
        node_global_index[pc2_mask] - num_pc1, inter_pc2_tensor
    ).float()

    # --- interface 4.0Å
    pc1_labels_tensor = torch.tensor(pc1_labels)
    pc2_labels_tensor = torch.tensor(pc2_labels)

    is_inter4[pc1_mask] = pc1_labels_tensor[
        node_global_index[pc1_mask]
    ].float()

    is_inter4[pc2_mask] = pc2_labels_tensor[
        node_global_index[pc2_mask] - num_pc1
    ].float()

    node_prior = torch.stack([is_cdr, is_inter10, is_inter4], dim=-1)

    node_feat = torch.cat([node_feat, node_prior], dim=-1)

    # ======================
    # 9. node chain
    # ======================
    node_chain = torch.tensor(
        [0] * len(pc1_nodes) + [1] * len(pc2_nodes),
        dtype=torch.long
    )

    # ======================
    # 10. label（可选）
    # ======================
    node_label = torch.cat([
        pc1_labels_tensor[pc1_nodes],
        pc2_labels_tensor[pc2_nodes]
    ], dim=0)

    graph = Data(x=node_feat, node_type=node_prior, node_chain=node_chain, node_label=node_label,
                 global_index=node_global_index, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type)

    if cdr_indices is not None:
        graph.cdr_index = cdr_tensor

    return graph



    # if cdr_indices is not None:
    #     is_cdr_pc1 = {res.ix: (res.ix in cdr_indices) for res in pc1_res}
    # else:
    #     is_cdr_pc1 = {}
    #
    # edge_cdr_flag = []
    # for src, dst in edge_index:
    #     flag = 0.0
    #     if src in is_cdr_pc1 and is_cdr_pc1[src]:  # src 是抗体残基且在 CDR
    #         flag = 1.0
    #     elif dst in is_cdr_pc1 and is_cdr_pc1[dst]:
    #         flag = 1.0
    #     edge_cdr_flag.append(flag)
    #
    # edge_cdr_flag = torch.tensor(edge_cdr_flag, dtype=torch.float).unsqueeze(-1)
    # edge_attr = torch.cat([edge_attr, edge_cdr_flag], dim=-1)
    #
    # # 3. 节点特征
    # pc1_feat, pc2_feat = get_node_features(pair, sel_pc1_idx, sel_pc2_idx, paths)
    # # ---- CDR prior ----
    # cdr_prior_pc1 = torch.zeros(len(pc1_res))
    # cdr_prior_pc2 = torch.zeros(len(pc2_res))
    #
    # if cdr_indices is not None:
    #     important_res = set(inter_pc1_idx) & set(cdr_indices)
    #     for i in important_res:
    #         cdr_prior_pc1[i] = 1.0
    #
    # cdr_prior_pc1 = cdr_prior_pc1[sel_pc1_idx]
    # cdr_prior_pc2 = cdr_prior_pc2[sel_pc2_idx]
    #
    # pc1_feat = torch.cat([pc1_feat, cdr_prior_pc1.unsqueeze(-1)], dim=-1)
    # pc2_feat = torch.cat([pc2_feat, cdr_prior_pc2.unsqueeze(-1)], dim=-1)
    #
    # sel_pc1_labels = torch.tensor(pc1_labels[sel_pc1_idx], dtype=torch.float)
    # sel_pc2_labels = torch.tensor(pc2_labels[sel_pc2_idx], dtype=torch.float)
    #
    # # 4. 合并全局残基索引
    # global_indices = sorted(set([e for edge in edge_index for e in edge]))
    # node_type = torch.tensor([res_type_map.get(idx, -1) for idx in global_indices], dtype=torch.long)
    # if -1 in node_type:
    #     raise ValueError(f"Invalid residue index in graph {pair}")
    #
    # # 5. 重新编号所有节点：构建 res_id -> 连续编号 映射，并更新 edge_index
    # index_map = {res_id: i for i, res_id in enumerate(global_indices)}
    # edge_index_mapped = [(index_map[src], index_map[dst]) for src, dst in edge_index]
    # edge_index = torch.tensor(edge_index_mapped, dtype=torch.long).t().contiguous()
    # global_indices = torch.tensor(global_indices)


# 构建AAI的inter图:CDR-KNN #
def get_or_extract_cdrs(pdb_id, cdr_dict, chain_id, sequences, data_type, pair_name=None):
    pdb_id = pdb_id.lower()
    if pdb_id in cdr_dict and chain_id in cdr_dict[pdb_id]:
        return cdr_dict[pdb_id][chain_id]

    # 不在已有字典中，调用工具生成
    cdr_ranges_ab_chothia = [{'H1': (26, 32), 'H2': (52, 56), 'H3': (95, 102)},
                             {'L1': (24, 34), 'L2': (50, 56), 'L3': (89, 97)}]
    cdr_ranges_ab_imgt = [{'H1': (26, 33), 'H2': (51, 56), 'H3': (93, 102)},
                          {'L1': (27, 32), 'L2': (50, 51), 'L3': (89, 97)}]
    cdr_ranges_tcr_imgt = [{'A1': (26, 33), 'A2': (51, 56), 'A3': (93, 102)},
                           {'B1': (27, 32), 'B2': (50, 51), 'B3': (89, 97)}]
    # cdr_ranges_tcr_imgt = [{"A1": (27, 38), "A2": (56, 65), "A3": (105, 117)},
    #                        {"B1": (27, 38), "B2": (56, 65), "B3": (105, 117)}]
    if 'tcr' in data_type.lower():
        schemes_to_try = [('imgt', cdr_ranges_tcr_imgt)]
    else:
        schemes_to_try = [
            ('chothia', cdr_ranges_ab_chothia),
            ('imgt', cdr_ranges_ab_imgt)
        ]

    anarci_out, cdr_ranges_list = None, None

    for scheme, cdr_range in schemes_to_try:
        try:
            anarci_out = run_anarci_numbering(sequences, scheme)
            cdr_ranges_list = cdr_range
            # print(f"[INFO] Success with scheme: {scheme}")
            break
        except Exception as e:
            print(f"[WARN] ANARCI failed with scheme {scheme}: {e}")

    if anarci_out is None or anarci_out[1][0] is None:
        # print(f"[PPI] {pair_name}")
        print(f"[ERROR] ANARCI failed to annotate {pdb_id}")
        return None
        # raise RuntimeError(f"ANARCI failed with IMGT schemes for {pdb_id} ({data_type}).")

    # if scheme == "chothia" and pair_name:
    #     print(f"[AAI] {pair_name}")
    # if scheme == "imgt" and pair_name:
    #     print(f"[TCR] {pair_name}")

    cdrs = {}
    for idx, (result, chain_info) in enumerate(zip(anarci_out[0], anarci_out[1])):
        chain_type = result[0]
        numbered = chain_info[0][0]  # List of ((pos, icode), aa)
        clean_residues = [((pos, icode), aa) for ((pos, icode), aa) in numbered if aa != '-']

        if chain_type in ("H", "A"):
            cdr_ranges_ref = cdr_ranges_list[0]
        elif chain_type in ("L", "B"):
            cdr_ranges_ref = cdr_ranges_list[1]
        else:
            continue

        for region, (start, end) in cdr_ranges_ref.items():
            selected_aas = [
                aa for ((pos, icode), aa) in clean_residues
                if start <= pos <= end  # pos 范围内，icode 不管是 '', A, B... 都保留
            ]
            cdrs[region] = ''.join(selected_aas)

    # cdr_dict[pdb_id][chain_id] = cdrs

    if pdb_id not in cdr_dict:
        cdr_dict[pdb_id] = {}
    cdr_dict[pdb_id][chain_id] = cdrs

    # try:
    #     with open('cdr_sequences.json', "w") as f:
    #         json.dump(cdr_dict, f, indent=2)
    # except Exception as e:
    #     print(f"[WARN] Failed to save CDR json: {e}")

    return cdrs


def get_cdr_res_indices(ab_seq, cdrs):
    """
    在拼接的 ab 序列中查找 CDR 子串的位置，返回对应的全局索引。
    H_seq: 抗体重链残基序列（字符串）
    L_seq: 抗体轻链残基序列（字符串）
    cdrs: CDR 区域序列片段 dict，例如 {'H1': 'BC', 'H2': 'DE', 'H3': 'FG', 'L1': 'OQ', ...}
    """
    # ab_seq = H_seq + L_seq
    cdr_indices = []

    for cdr_seq in cdrs.values():
        start_idx = ab_seq.find(cdr_seq)
        if start_idx != -1:
            cdr_indices.extend(range(start_idx, start_idx + len(cdr_seq)))

    return set(cdr_indices)


def get_aai_edge_index(
        pc1_res, pc2_res, cdr_indices, paths, k=0, inter_edge_cutoff=10.0, inter_label_cutoff=4.5,
        edge_feat_type=('geom', 'coul', 'lj'), normalize=True, bidirectional=True
):
    num_pc1 = pc2_offset = len(pc1_res)
    num_pc2 = len(pc2_res)

    energy_matrices = load_energy_matrices(paths['energy_path'])
    full_aa_distance = compute_dist_map(pc1_res, pc2_res, paths['dist_map'])
    aa_distance: np.ndarray = full_aa_distance[:num_pc1, num_pc1:]

    edge_index, edge_attr, edge_type = [], [], []
    selected_pc2 = set()
    pc1_bind_labels = [0] * num_pc1
    pc2_bind_labels = [0] * num_pc2

    energy_key = "pc1_pc2"
    inter_etype = EDGE_TYPE["inter"]

    for i_pc1 in range(num_pc1):
        dist_row = aa_distance[i_pc1]
        thresh_pc2_indx = np.where(dist_row <= inter_edge_cutoff)[0]
        is_cdr = i_pc1 in cdr_indices

        if is_cdr and len(thresh_pc2_indx) > 0:
            nearest_pc2 = thresh_pc2_indx
        else:
            continue

        #     nearest_pc2 = thresh_pc2_indx if len(thresh_pc2_indx) > 0 else np.argsort(dist_row)[:k]
        # else:
        #     nearest_pc2 = thresh_pc2_indx

        for j_pc2 in nearest_pc2:
            global_pc2_j = j_pc2 + pc2_offset
            dist = aa_distance[i_pc1, j_pc2]

            if dist <= inter_label_cutoff:
                pc1_bind_labels[i_pc1] = 1
                pc2_bind_labels[j_pc2] = 1

            feat = cal_edge_feat(i_pc1, j_pc2, pc1_res, pc2_res, dist, energy_matrices,
                                 energy_key, edge_feat_type, reverse=False)
            edge_index.append((i_pc1, global_pc2_j))
            edge_attr.append(feat)
            selected_pc2.add(j_pc2)
            edge_type.append(inter_etype)

            if bidirectional:
                rev_feat = cal_edge_feat(i_pc1, j_pc2, pc1_res, pc2_res, dist, energy_matrices,
                                         energy_key, edge_feat_type, reverse=True)
                edge_index.append((global_pc2_j, i_pc1))
                edge_attr.append(rev_feat)
                edge_type.append(inter_etype)

    edge_attr = torch.stack(edge_attr, dim=0)  # (num_edges, feat_dim)
    if normalize:
        edge_attr = zscore_normalize(edge_attr)

    return edge_index, edge_attr, edge_type, selected_pc2, \
           np.array(pc1_bind_labels), np.array(pc2_bind_labels), full_aa_distance


def build_cdr_res_graph(pair, cdr_indices, pc1_res, pc2_res, paths):
    res_type_map = {
        **{res.ix: 0 for res in pc1_res},
        **{res.ix: 1 for res in pc2_res}
    }

    # 1. 边索引和属性（使用原始全局编号）
    # inter_edge_index, inter_edge_attr, inter_edge_type, inter_pc2_idx, \
    # pc1_labels, pc2_labels, full_aa_distance = get_aai_edge_index(
    #     pc1_res, pc2_res, cdr_indices, paths, inter_edge_cutoff=10.0)

    inter_edge_index, inter_edge_attr, inter_edge_type, inter_pc1_idx, inter_pc2_idx, \
    pc1_labels, pc2_labels, full_aa_distance = get_inter_edge_index(
        pc1_res, pc2_res, paths, inter_edge_cutoff=10.0)

    if len(inter_pc2_idx) == 0:
        print(f"[Warning] No interface found in {pair}")
    intra1_edge_index, intra1_edge_attr, intra1_edge_type, sel_pc1_idx = \
        get_intra_edge_index(pc1_res, cdr_indices, full_aa_distance, len(pc1_res), dist_cutoff=4.5)

    intra2_edge_index, intra2_edge_attr, intra2_edge_type, sel_pc2_idx = \
        get_intra_edge_index(pc2_res, inter_pc2_idx, full_aa_distance, len(pc1_res), 'pc2', dist_cutoff=4.5)

    edge_index = inter_edge_index + intra1_edge_index + intra2_edge_index
    edge_attr = torch.cat([inter_edge_attr, intra1_edge_attr, intra2_edge_attr], dim=0)
    edge_type = torch.tensor(inter_edge_type + intra1_edge_type + intra2_edge_type, dtype=torch.long)

    # edge_index_pos = [(src, dst) for src, dst in edge_index if src < dst]
    # unique_src = sorted(set([src for src, dst in edge_index_pos]))  # AB 侧节点
    # unique_dst = sorted(set([dst for src, dst in edge_index_pos]))  # AG 侧全局编号

    # 2. 获取节点特征：返回 ab_feat 和 ag_feat
    pc1_feat, pc2_feat = get_node_features(pair, sel_pc1_idx, sel_pc2_idx, paths)
    sel_pc1_labels = torch.tensor(pc1_labels[sel_pc1_idx], dtype=torch.float)
    sel_pc2_labels = torch.tensor(pc2_labels[sel_pc2_idx], dtype=torch.float)

    # 3. 合并全局残基索引：AB 侧节点 + 邻近抗原
    # global_indices = unique_src + unique_dst
    global_indices = sorted(set([e for edge in edge_index for e in edge]))
    node_type = torch.tensor([res_type_map.get(idx, -1) for idx in global_indices], dtype=torch.long)

    # 4. 重新编号所有节点：构建 res_id -> 连续编号 映射，并更新 edge_index
    index_map = {res_id: i for i, res_id in enumerate(global_indices)}
    edge_index_mapped = [(index_map[src], index_map[dst]) for src, dst in edge_index]
    edge_index = torch.tensor(edge_index_mapped, dtype=torch.long).t().contiguous()
    global_indices = torch.tensor(global_indices)

    assert max(edge_index[0]) < len(global_indices), f"Invalid graph for pair {pair}"
    cdr_indices = torch.tensor(list(cdr_indices))

    graph = Data(pc1_node_attr=pc1_feat, pc2_node_attr=pc2_feat, node_type=node_type, cdr_index=cdr_indices,
                 global_index=global_indices, edge_index=edge_index, edge_attr=edge_attr,
                 edge_type=edge_type, pc1_node_label=sel_pc1_labels, pc2_node_label=sel_pc2_labels)

    return graph


# 判别PPI类型
# with open('cdr_sequences.json', 'r') as f:
#     cdr_dict = json.load(f)
#
# path = '/home/yyShen/NAcontact'
# data_type = "ppi_bu"
# pair_list = np.loadtxt(f'./PPB-Affinity/TCR_bu.txt', dtype=str)
#
# for i, pair in enumerate(pair_list):
#     # pair = '5drz_HL_P'
#     # print(i, pair)
#     parts = pair.split('.', 1)
#     complex_id = parts[0]
#     mut_info = parts[1] if len(parts) > 1 else None
#     mut_chains = list({mut[1] for mut in mut_info.split('_')}) if mut_info else []
#     pdb, chain1, chain2 = complex_id.split('_')
#
#     complex_pdb = f"./pdb_files/{data_type}/{pair}.pdb"
#     u = mda.Universe(complex_pdb)
#
#     if 'aai' in data_type:
#         pc1_fasta_dir = f"./fasta/{data_type}_ab_fasta"
#         pc2_fasta_dir = f"./fasta/{data_type}_ag_fasta"
#         chain_str = "HL"
#     elif 'tcr' in data_type:
#         pc1_fasta_dir = f"./fasta/{data_type}_tcr_fasta/"
#         pc2_fasta_dir = f"./fasta/{data_type}_pmhc_fasta/"
#         chain_str = "AB"
#     else:
#         pc1_fasta_dir = pc2_fasta_dir = f"./fasta/{data_type}_pc_fasta"
#         chain_str = "AB"
#     paths = get_data_paths(data_type, pair, path)
#
#     anarci_input = []
#     pc1_seq, pc1_res = "", None
#     for chain, chain_type in zip(chain1, chain_str):
#         seq, res = get_chain_seq(u, pdb, chain, pc1_fasta_dir, mut_chains, mut_info)
#         anarci_input.append((f"{chain_type}", seq))  # ← run_anarci need [(id, seq), ...]
#         pc1_seq += seq
#         pc1_res = res if pc1_res is None else pc1_res + res
#
#     pc2_seq, pc2_res = "", None
#     for chain in chain2:
#         seq, res = get_chain_seq(u, pdb, chain, pc2_fasta_dir, mut_chains, mut_info)
#         pc2_seq += seq
#         pc2_res = res if pc2_res is None else pc2_res + res
#
#     cdrs = get_or_extract_cdrs(pdb, cdr_dict, chain1, anarci_input, 'tcr', pair)
#
# with open('cdr_sequences.json', 'w') as f:
#     json.dump(cdr_dict, f, indent=2)


def build_res_inter_graph(data_type, path, cdr_dict=None):
    """
    统一构建 inter 图：
    - AAI/TCR-pMHC：优先 CDR-KNN 构图，缺失时退化为距离构图
    - PPI：直接距离构图
    """

    interact_path = f"{path}/feature/plip_feat/{data_type}_interaction.pkl"
    with open(interact_path, "rb") as fi:
        interact_dict = pickle.load(fi)

    pair_list = np.atleast_1d(
        np.loadtxt(f'{path}/data/PPB-Affinity/{data_type}_pair.txt', dtype=str)
    )
    for i, pair in enumerate(pair_list):
        # pair = "1mhh_BA_E"
        print(i, pair)
        parts = pair.split('.', 1)
        complex_id = parts[0]
        mut_info = parts[1] if len(parts) > 1 else None
        mut_chains = list({mut[1] for mut in mut_info.split('_')}) if mut_info else []
        pdb, chain1, chain2 = complex_id.split('_')
        pair_type = data_type

        if pair in ['6ysq_AC_G', '7bw4_A_BCD']:
            chain1_af3 = 'A'
            chain2_af3 = 'B'
        else:
            all_input_chains = list(chain1) + list(chain2)
            af3_chain_ids = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            chain_map = {
                old: new
                for old, new in zip(all_input_chains, af3_chain_ids)
            }
            chain1_af3 = ''.join(chain_map[c] for c in chain1)
            chain2_af3 = ''.join(chain_map[c] for c in chain2)

        complex_pdb = f"{path}/data/pdb_files/{data_type}_mapped/{pair}.pdb"
        u = mda.Universe(complex_pdb)
        af_fasta_dir = f"{path}/data/fasta/{data_type}_pc_fasta"

        if "aai" in pair_type:
            chain_str = "HL"
        elif "tcr" in pair_type:
            chain_str = "AB"
        else:
            chain_str = chain1_af3

        paths = get_data_paths(data_type, pair, path)
        graph_file = paths['graph_file']
        pair_interact_feat = interact_dict.get(pair, {})

        if os.path.exists(graph_file):
            continue
        #     # print(i, pair, f"graph already exists, skip.")
        #     with open(graph_file, 'rb') as fd:
        #         graph = pickle.load(fd)
            graph = torch.load(graph_file)
            t=1
        #     if hasattr(graph, "pc1_node_attr"):
        #         continue

        anarci_input = []
        pc1_seq, pc1_res = "", None
        for chain, chain_type in zip(chain1_af3, chain_str):
            seq, res = get_chain_seq(u, pdb, chain, af_fasta_dir, mut_chains, mut_info)
            anarci_input.append((f"{chain_type}", seq))  # ← run_anarci need [(id, seq), ...]
            pc1_seq += seq
            pc1_res = res if pc1_res is None else pc1_res + res

        pc2_seq, pc2_res = "", None
        for chain in chain2_af3:
            seq, res = get_chain_seq(u, pdb, chain, af_fasta_dir, mut_chains, mut_info)
            pc2_seq += seq
            pc2_res = res if pc2_res is None else pc2_res + res

        # 构图逻辑
        cdr_indices = None
        if pair_type in ('aai', 'tcr-pmhc') and cdr_dict is not None:
            cdrs = get_or_extract_cdrs(pdb, cdr_dict, chain1, anarci_input, data_type)
            if cdrs is not None:
                cdr_indices = get_cdr_res_indices(pc1_seq, cdrs)

        graph = build_graph(pair, pc1_res, pc2_res, paths, pair_interact_feat, cdr_indices)
        os.makedirs(os.path.dirname(graph_file), exist_ok=True)
        torch.save(graph, graph_file)

        # graph = None
        # if 'ppi' not in data_type and cdr_dict is not None:
        #     cdrs = get_or_extract_cdrs(pdb, cdr_dict, chain1, anarci_input, data_type)
        #     if cdrs is not None:
        #         cdr_indices = get_cdr_res_indices(pc1_seq, cdrs)
        #         graph = build_res_graph(pair, pc1_res, pc2_res, paths, cdr_indices)
        #         # graph = build_cdr_res_graph(pair, cdr_indices, pc1_res, pc2_res, paths)
        #     # else:
        #     #     print(f"{pair} use dist-based build graph!")
        #
        # if graph is None:
        #     graph = build_res_graph(pair, pc1_res, pc2_res, paths)
        #
        # torch.save(graph, graph_file)
        # with open(graph_file, 'wb') as fg:
        #     pickle.dump(graph, fg)


def get_data_paths(data_type, pair, base_path="/home/yyShen/NAcontact"):
    paths = {
        "base_path": base_path,
        "graph_file": f"{base_path}/data/affinity_data/res_graph/{data_type}/{pair}.gh",
        "dist_map": f"{base_path}/data/affinity_data/res_graph_12A/res_dist_map/{data_type}/{pair}.npy",
        "energy_path": f"{base_path}/feature/energy_features/{data_type}/{pair}",
        "seq_path": f"{base_path}/feature/seq_features/{data_type}/{pair}.pkl",
        "str_path": f"{base_path}/feature/str_features/{data_type}/{pair}.pkl",
        "hand_str_path": f"{base_path}/feature/hand_str_feat_norm/{data_type}/{pair}.pkl"
    }

    return paths


EDGE_TYPE = {
    "intra_pc1": 0,
    "intra_pc2": 1,
    "inter": 2,
}
def parse_args():
    parser = argparse.ArgumentParser(description="Build residue-level graphs for a prepared dataset.")
    parser.add_argument("--data-type", default="ppi", help="Dataset name")
    parser.add_argument(
        "--project-root",
        default=str(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))),
        help="Project root",
    )
    parser.add_argument(
        "--cdr-json",
        default="",
        help="CDR sequence JSON path (default: <project-root>/data/cdr_sequences.json)",
    )
    parser.add_argument(
        "--label-json",
        default="",
        help="Optional label JSON path; accepted for compatibility but not used for graph construction",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    path = os.path.abspath(args.project_root)
    cdr_json = args.cdr_json or os.path.join(path, "data", "cdr_sequences.json")
    if os.path.exists(cdr_json):
        with open(cdr_json, "r") as f:
            cdr_dict = json.load(f)
    elif args.data_type in ("aai", "tcr-pmhc"):
        raise FileNotFoundError(f"CDR sequence file not found: {cdr_json}")
    else:
        cdr_dict = None

    data_type = args.data_type
    build_res_inter_graph(data_type, path, cdr_dict=cdr_dict)

# # 构建AAI的inter图:CDR-KNN #
#
# with open('cdr_sequences.json', 'r') as f:
#     cdr_dict = json.load(f)
#
# path = '/home/yyShen/NAcontact'
# data_type = "aai"
# pair_list = np.loadtxt(f'./PPB-Affinity/{data_type}_pair_renamed.txt', dtype=str)
#
# for i, pair in enumerate(pair_list[400:]):
#     # pair = '5drz_HL_P'
#     print(i, pair)
#     parts = pair.split('.', 1)
#     complex_id = parts[0]
#     mut_info = parts[1] if len(parts) > 1 else None
#     mut_chains = list({mut[1] for mut in mut_info.split('_')}) if mut_info else []
#     pdb, chain1, chain2 = complex_id.split('_')
#
#     complex_pdb = f"./pdb_files/{data_type}/{pair}.pdb"
#     u = mda.Universe(complex_pdb)
#
#     if 'aai' in data_type:
#         pc1_fasta_dir = f"./fasta/{data_type}_ab_fasta"
#         pc2_fasta_dir = f"./fasta/{data_type}_ag_fasta"
#         chain_str = "HL"
#     else:
#         pc1_fasta_dir = f"./fasta/{data_type}_tcr_fasta/"
#         pc2_fasta_dir = f"./fasta/{data_type}_pmhc_fasta/"
#         chain_str = "AB"
#     paths = get_data_paths(data_type, pair, path)
#
#     anarci_input = []
#     pc1_seq, pc1_res = "", None
#     for chain, chain_type in zip(chain1, chain_str):
#         seq, res = get_chain_seq(u, pdb, chain, pc1_fasta_dir, mut_chains, mut_info)
#         anarci_input.append((f"{chain_type}", seq))  # ← run_anarci need [(id, seq), ...]
#         pc1_seq += seq
#         pc1_res = res if pc1_res is None else pc1_res + res
#
#     pc2_seq, pc2_res = "", None
#     for chain in chain2:
#         seq, res = get_chain_seq(u, pdb, chain, pc2_fasta_dir, mut_chains, mut_info)
#         pc2_seq += seq
#         pc2_res = res if pc2_res is None else pc2_res + res
#
#     cdrs = get_or_extract_cdrs(pdb, cdr_dict, chain1, anarci_input, data_type)
#     if cdrs is None:
#         print(f"{pair} use dist-based build graph!")
#         continue
#     # assert cdrs is not None, f"Check {pair}!"
#     # 获取 CDR 残基索引（全局ab索引）
#     cdr_indices = get_cdr_res_indices(pc1_seq, cdrs)
#
#     cdr_res_graph = build_cdr_res_graph(pair, cdr_indices, pc1_res, pc2_res, paths)
#     with open(paths['graph_file'], 'wb') as f:
#         pickle.dump(cdr_res_graph, f)
#
# # ============================================================================================
#
# #
# # 构建ppi/tcr-pmhc的dist-based inter图 #
#
# path = '/home/yyShen/NAcontact'
# data_type = "ppi"
# pair_list = np.loadtxt(f'/home/yyShen/NAcontact/data/PPB-Affinity/{data_type}_pair_2328.txt', dtype=str)
# # pair_list = np.loadtxt(f'/home/yyShen/NAcontact/data/PPB-Affinity/tcr_dist_pair.txt', dtype=str)
#
# for i, pair in enumerate(pair_list):
#     # pair = '1h0t_A_B'
#     print(i, pair)
#     parts = pair.split('.', 1)
#     complex_id = parts[0]
#     mut_info = parts[1] if len(parts) > 1 else None
#     mut_chains = list({mut[1] for mut in mut_info.split('_')}) if mut_info else []
#     pdb, chain1, chain2 = complex_id.split('_')
#     # chains = {'pc1': list(chain1), 'pc2': list(chain2)}
#
#     complex_pdb = f"./pdb_files/{data_type}/{pair}.pdb"
#     u = mda.Universe(complex_pdb)
#
#     if 'ppi' in data_type:
#         pc1_fasta_dir = pc2_fasta_dir = f"./fasta/{data_type}_pc_fasta"
#     else:
#         pc1_fasta_dir = f"./fasta/{data_type}_tcr_fasta/"
#         pc2_fasta_dir = f"./fasta/{data_type}_pmhc_fasta/"
#
#     paths = get_data_paths(data_type, pair, path)
#
#     pc1_seq, pc1_res = "", None
#     for chain in chain1:
#         seq, res = get_chain_seq(u, pdb, chain, pc1_fasta_dir, mut_chains, mut_info)
#         pc1_seq += seq
#         pc1_res = res if pc1_res is None else pc1_res + res
#
#     pc2_seq, pc2_res = "", None
#     for chain in chain2:
#         seq, res = get_chain_seq(u, pdb, chain, pc2_fasta_dir, mut_chains, mut_info)
#         pc2_seq += seq
#         pc2_res = res if pc2_res is None else pc2_res + res
#
#     res_graph = build_inter_res_graph(pair, pc1_res, pc2_res, paths)
#
#     with open(paths['graph_file'], 'wb') as f:
#         pickle.dump(res_graph, f)
