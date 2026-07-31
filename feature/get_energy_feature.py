import MDAnalysis as mda
import numpy as np
from itertools import product
import os
import shutil


def parse_martini_itp(itp_file):
    """
    解析 MARTINI `.itp` 文件中的 Lennard-Jones 参数（C6, C12）
    """
    lj_params = {}

    with open(itp_file, "r") as f:
        lines = f.readlines()

    parse_flag = False
    for line in lines:
        if "[ nonbond_params ]" in line:
            parse_flag = True
            continue
        if parse_flag and line.startswith("["):
            break
        if parse_flag and line.strip():
            if line.startswith(";"):
                continue  # 忽略注释
            parts = line.split()
            if len(parts) >= 5:
                atom1, atom2, func, C6, C12 = parts[:5]
                try:
                    C6, C12 = float(C6), float(C12)  # 转换成浮点数
                    lj_params[(atom1, atom2)] = (C6, C12)
                    lj_params[(atom2, atom1)] = (C6, C12)  # 交换顺序存储
                except ValueError:
                    print(f"Error converting C6, C12: {C6}, {C12} in line: {line}")
                    continue  # 跳过有错误的数据
    return lj_params


def get_lj_params(atom_type1, atom_type2, lj_params):
    """
    获取两个 CG 粒子之间的 Lennard-Jones 参数 (C6, C12)
    """
    return lj_params.get((atom_type1, atom_type2), (0.0, 0.0))  # 若未找到，返回 (0.0, 0.0)


def parse_topology_exclusions(u):
    # 提取排除的相互作用对
    excluded_pairs = set()

    # 提取 bonds（1-2 作用）
    for bond in u.bonds:
        excluded_pairs.add((bond.indices[0], bond.indices[1]))

    # 提取 angles（1-3 作用）
    for angle in u.angles:
        excluded_pairs.add((angle.indices[0], angle.indices[2]))  # 取 1-3

    # 提取 dihedrals（1-4 作用）
    for dih in u.dihedrals:
        excluded_pairs.add((dih.indices[0], dih.indices[3]))  # 取 1-4

    # 过滤跨分子作用
    filtered_exclusions = set()
    for atom_pair in excluded_pairs:
        atom1, atom2 = atom_pair
        chain1, chain2 = u.atoms[atom1].chainID, u.atoms[atom2].chainID  # 用 chainID 区分
        if chain1 == chain2:  # 只保留同一分子链内的作用
            filtered_exclusions.add(atom_pair)

    return filtered_exclusions


# def cal_interactions(res1, res2, excluded_pairs):
#     """计算两个 CG 颗粒/残基之间的库仑 & Lennard-Jones 相互作用，排除 1-2、1-3、1-4 作用对"""
#     energy_coul = 0.0
#     energy_lj = 0.0
#
#     for atom1, atom2 in product(res1.atoms, res2.atoms):
#         pair = (atom1.index, atom2.index)
#
#         # 如果是排除的作用对（1-2、1-3、1-4），跳过计算
#         if pair in excluded_pairs or pair[::-1] in excluded_pairs:
#             continue
#
#         r = np.linalg.norm(atom1.position - atom2.position) / 10.0  # Å 转 nm
#         q1, q2 = atom1.charge, atom2.charge
#
#         # 直接从 MARTINI `.itp` 文件读取 C6 和 C12
#         C6, C12 = get_lj_params(atom1.type, atom2.type, martini_params)  # 需提前解析 .itp 获取 C6、C12
#
#         # 计算 Lennard-Jones 势能
#         if r > 0.4:
#             energy_lj += (C12 / r ** 12) - (C6 / r ** 6)
#
#         # 计算库仑能量（仅在 CG 粒子有电荷时）
#         if r > 0.4 and abs(q1) > 0 and abs(q2) > 0:
#             energy_coul += (k_e / 15) * (q1 * q2) / r  # MARTINI 采用 ε=15
#
#     return energy_coul, energy_lj


# def cal_interactions(res1, res2, excluded_pairs, atom_energy_lj, atom_energy_coul,
#                      atom_offset1, atom_offset2, r_min):
#
#     """ 计算 CG 颗粒/残基之间的库仑 & Lennard-Jones 相互作用 """
#     res_energy_lj = 0.0
#     res_energy_coul = 0.0
#     for atom1, atom2 in product(res1.atoms, res2.atoms):
#         # pair = (atom1.index, atom2.index)
#
#         # if pair in excluded_pairs or pair[::-1] in excluded_pairs:
#         #     continue  # 排除 1-2、1-3、1-4 作用对
#
#         # 计算原子对距离（单位：nm）
#         r = np.linalg.norm(atom1.position - atom2.position) / 10.0  # Å 转 nm
#         q1, q2 = atom1.charge, atom2.charge
#
#         # 直接从 MARTINI `.itp` 文件读取 C6 和 C12
#         C6, C12 = get_lj_params(atom1.type, atom2.type, martini_params)
#
#         # 计算 Lennard-Jones 和 coul 势能
#         if r > r_min:
#             energy_lj = (C12 / r ** 12) - (C6 / r ** 6)
#         else:
#             energy_lj = 0.0
#
#         if r > r_min and abs(q1) > 0 and abs(q2) > 0:
#             energy_coul = (k_e / 15) * (q1 * q2) / r  # MARTINI 采用 ε=15
#         else:
#             energy_coul = 0.0
#
#         # 存入原子对级别矩阵
#         atom_idx1 = atom1.index - atom_offset1
#         atom_idx2 = atom2.index - atom_offset2
#         atom_energy_lj[atom_idx1, atom_idx2] = energy_lj
#         atom_energy_coul[atom_idx1, atom_idx2] = energy_coul
#
#         res_energy_lj += energy_lj
#         res_energy_coul += energy_coul
#
#     return res_energy_lj, res_energy_coul

def cal_interactions(res1, res2, excluded_pairs, atom_energy_lj, atom_energy_coul,
                     atom_offset1, atom_offset2, r_min=0.4, r_cutoff=1.2):
    """计算残基对的相互作用能，并记录原子对能量矩阵"""

    res_energy_lj = 0.0
    res_energy_coul = 0.0

    for atom1, atom2 in product(res1.atoms, res2.atoms):
        # pair = (atom1.index, atom2.index)
        # if pair in excluded_pairs or pair[::-1] in excluded_pairs:
        #     continue

        r = np.linalg.norm(atom1.position - atom2.position) / 10.0  # Å → nm
        if r > r_cutoff:
            continue

        C6, C12 = get_lj_params(atom1.type, atom2.type, martini_params)
        q1, q2 = atom1.charge, atom2.charge

        r_eff = max(r, r_min)
        energy_lj = C12 / r_eff**12 - C6 / r_eff**6
        energy_coul = (k_e / 15.0) * (q1 * q2) / r_eff if (q1 != 0.0 and q2 != 0.0) else 0.0

        # energy_lj = (C12 / r ** 12 - C6 / r ** 6) if r > r_min else 0.0
        # energy_coul = (k_e / 15.0) * (q1 * q2) / r if r > r_min and abs(q1) > 0 and abs(q2) > 0 else 0.0

        # ✅ 累计残基对能量（用于构图）
        res_energy_lj += energy_lj
        res_energy_coul += energy_coul

        # ✅ 保留原子对能量（用于后续可视化/分析）
        atom_idx1 = atom1.index - atom_offset1
        atom_idx2 = atom2.index - atom_offset2
        atom_energy_lj[atom_idx1, atom_idx2] = energy_lj
        atom_energy_coul[atom_idx1, atom_idx2] = energy_coul

    return res_energy_lj, res_energy_coul


# === 读取 MARTINI 参数 ===
project_root = os.environ.get(
    "UNIBA_PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
path = os.environ.get(
    "UNIBA_AFFINITY_DATA_PATH",
    os.path.join(project_root, "data", "affinity_data"),
)
path = os.path.join(path, "")
data_type = os.environ.get("UNIBA_DATA_TYPE", "ppi")
pair_list_path = os.environ.get(
    "UNIBA_PAIR_LIST",
    os.path.join(project_root, "data", "PPB-Affinity", f"{data_type}_pair.txt"),
)
pair_list = np.atleast_1d(np.loadtxt(pair_list_path, dtype=str))
# pair_list = np.loadtxt('/home/yyShen/NAcontact/feature/em_finished.txt', dtype=str)
martini_params = parse_martini_itp(f"{path}cg_input/martini_v2.2.itp")

energy_keys = [
    "pc1_pc2_lj_res", "pc1_pc2_lj_atom",
    "pc1_lj_res", "pc1_lj_atom",
    "pc2_lj_res", "pc2_lj_atom",
    "pc1_pc2_coul_res", "pc1_pc2_coul_atom",
    "pc1_coul_res", "pc1_coul_atom",
    "pc2_coul_res", "pc2_coul_atom"
]
# energy_keys = [
#     "ab_ag_lj_res", "ab_ag_lj_atom",
#     "ab_lj_res", "ab_lj_atom",
#     "ag_lj_res", "ag_lj_atom",
#     "ab_ag_coul_res", "ab_ag_coul_atom",
#     "ab_coul_res", "ab_coul_atom",
#     "ag_coul_res", "ag_coul_atom"
# ]
for i, pair in enumerate(pair_list):
    # pair = '6ysq_AC_G'
    if '.' in pair:
        complex_id = pair.split('.')[0]
    else:
        complex_id = pair
    parts = complex_id.split('_')
    print(i, pair)
    pair_path = f'{path}cg_input/cg_{data_type}/{pair}/'
    dir_path = f"./energy_features/{data_type}/{pair}"

    # energy_dir = f"/home/yyShen/NAcontact/feature/affinity_energy/{pair}/"
    # ab_file = os.path.join(energy_dir, "ab_ag_coul_atom.txt")
    # pab_file = os.path.join(energy_dir, "ab_ag_coul_atom.txt")
    #
    # if os.path.exists(ab_file) and os.path.exists(pab_file):
    #     print(f"Replacing {ab_file} with {pab_file}")
    #     os.remove(ab_file)                     # 删除原始 ab 文件
    #     os.rename(pab_file, ab_file)           # 重命名 pab 文件为 ab 文件
    # else:
    #     print(f"Skipped {pair}: missing file(s)")

    # result_files_exist = all(os.path.exists(f"{dir_path}/{key}.txt") for key in energy_keys)
    # assert result_files_exist, f"Check {pair}"
    result_files_exist = False
    if result_files_exist:
        lj_ab_ag_res = np.loadtxt(f"{dir_path}/{energy_keys[0]}.txt")
        lj_ab_ag_atom = np.loadtxt(f"{dir_path}/{energy_keys[1]}.txt")

        lj_ab_res = np.loadtxt(f"{dir_path}/{energy_keys[2]}.txt")
        lj_ab_atom = np.loadtxt(f"{dir_path}/{energy_keys[3]}.txt")

        lj_ag_res = np.loadtxt(f"{dir_path}/{energy_keys[4]}.txt")
        lj_ag_atom = np.loadtxt(f"{dir_path}/{energy_keys[5]}.txt")

        coul_ab_ag_res = np.loadtxt(f"{dir_path}/{energy_keys[6]}.txt")
        coul_ab_ag_atom = np.loadtxt(f"{dir_path}/{energy_keys[7]}.txt")

        coul_ab_res = np.loadtxt(f"{dir_path}/{energy_keys[8]}.txt")
        coul_ab_atom = np.loadtxt(f"{dir_path}/{energy_keys[9]}.txt")

        coul_ag_res = np.loadtxt(f"{dir_path}/{energy_keys[10]}.txt")
        coul_ag_atom = np.loadtxt(f"{dir_path}/{energy_keys[11]}.txt")

        # === 打印统计信息 ===
        print(pair)
        print(f"coul (Ab-Ag): {np.sum(coul_ab_ag_res)}")
        print(f"Lennard-Jones (Ab-Ag): {np.sum(lj_ab_ag_res)}")

        print(f"coul (Ab-Ab): {np.sum(coul_ab_res) / 2}")
        print(f"Lennard-Jones (Ab-Ab): {np.sum(lj_ab_res) / 2}")

        print(f"coul (Ag-Ag): {np.sum(coul_ag_res) / 2}")
        print(f"Lennard-Jones (Ag-Ag): {np.sum(lj_ag_res) / 2}")

        print(f"Total coul: {np.sum(coul_ab_ag_res) + np.sum(coul_ab_res) / 2 + np.sum(coul_ag_res) / 2}")
        print(f"Total Lennard-Jones: {np.sum(lj_ab_ag_res) + np.sum(lj_ab_res) / 2 + np.sum(lj_ag_res) / 2}")

    else:
        # if os.path.exists(dir_path):
        #     shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)

        if len(parts) == 4:
            pdb, H_chain, L_chain, ag_chain = parts
            ab_chain_id = f"chainID Protein_{H_chain} Protein_{L_chain}"
        elif len(parts) == 3:
            pdb, ab_chain, ag_chain = parts
            # ab_chain_id = f"chainID Protein_{ab_chain}"
        else:
            raise ValueError(f"Invalid complex format: {pair}")

        all_input_chains = list(ab_chain) + list(ag_chain)
        af3_chain_ids = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        chain_map = {
            old: new
            for old, new in zip(all_input_chains, af3_chain_ids)
        }
        ab_mapped = [chain_map[c] for c in ab_chain]
        ag_mapped = [chain_map[c] for c in ag_chain]
        ab_chain_id = " or ".join([f"chainID Protein_{c}" for c in ab_mapped])
        ag_chain_id = " or ".join([f"chainID Protein_{c}" for c in ag_mapped])

        gro_file = f"{pair_path}em.gro"
        tpr_file = f"{pair_path}em.tpr"
        u = mda.Universe(tpr_file, gro_file)

        antibody = u.select_atoms(ab_chain_id)
        antigen = u.select_atoms(ag_chain_id)

        ab_residues = antibody.residues
        ag_residues = antigen.residues

        num_ab_res = len(ab_residues)
        num_ag_res = len(ag_residues)

        num_ab_atoms = len(antibody.atoms)
        num_ag_atoms = len(antigen.atoms)

        # 初始化矩阵
        lj_ab_ag_res = np.zeros((num_ab_res, num_ag_res))
        lj_ab_ag_atom = np.zeros((num_ab_atoms, num_ag_atoms))

        lj_ab_res = np.zeros((num_ab_res, num_ab_res))
        lj_ab_atom = np.zeros((num_ab_atoms, num_ab_atoms))

        lj_ag_res = np.zeros((num_ag_res, num_ag_res))
        lj_ag_atom = np.zeros((num_ag_atoms, num_ag_atoms))

        coul_ab_ag_res = np.zeros((num_ab_res, num_ag_res))
        coul_ab_ag_atom = np.zeros((num_ab_atoms, num_ag_atoms))

        coul_ab_res = np.zeros((num_ab_res, num_ab_res))
        coul_ab_atom = np.zeros((num_ab_atoms, num_ab_atoms))

        coul_ag_res = np.zeros((num_ag_res, num_ag_res))
        coul_ag_atom = np.zeros((num_ag_atoms, num_ag_atoms))

        # CG Lennard-Jones & 库仑计算
        k_e = 138.935485  # kJ·nm·mol^(-1)·e^(-2)（CG 力场 ε ≈ 15）
        r_min = 0.4  # nm, 保守值避免数值爆炸
        r_cutoff = 1.2  # nm, MARTINI 默认截断距离

        excluded_pairs = parse_topology_exclusions(u)

        # 计算抗体-抗原作用 (Ab-Ag)
        for m, res1 in enumerate(ab_residues):
            for n, res2 in enumerate(ag_residues):
                e_c, e_lj = cal_interactions(res1, res2, excluded_pairs, lj_ab_ag_atom, coul_ab_ag_atom,
                                             antibody.atoms[0].index, antigen.atoms[0].index, r_min)
                coul_ab_ag_res[m, n] = e_c
                lj_ab_ag_res[m, n] = e_lj

        # 计算抗体-抗体作用 (Ab-Ab)
        for x, res1 in enumerate(ab_residues):
            for y, res2 in enumerate(ab_residues):
                if x < y:
                    e_c, e_lj = cal_interactions(res1, res2, excluded_pairs, lj_ab_atom, coul_ab_atom,
                                                 antibody.atoms[0].index, antibody.atoms[0].index, r_min)
                    coul_ab_res[x, y] = e_c
                    lj_ab_res[x, y] = e_lj

                    # 对称填充
                    lj_ab_res[y, x] = lj_ab_res[x, y]
                    coul_ab_res[y, x] = coul_ab_res[x, y]
                    lj_ab_atom[y, x] = lj_ab_atom[x, y]
                    coul_ab_atom[y, x] = coul_ab_atom[x, y]

        # 计算抗原-抗原作用 (Ag-Ag)
        for t, res1 in enumerate(ag_residues):
            for k, res2 in enumerate(ag_residues):
                if t < k:
                    e_c, e_lj = cal_interactions(res1, res2, excluded_pairs, lj_ag_atom, coul_ag_atom,
                                                 antigen.atoms[0].index, antigen.atoms[0].index, r_min)
                    coul_ag_res[t, k] = e_c
                    lj_ag_res[t, k] = e_lj

                    # 对称填充
                    lj_ag_res[k, t] = lj_ag_res[t, k]
                    coul_ag_res[k, t] = coul_ag_res[t, k]
                    lj_ag_atom[k, t] = lj_ag_atom[t, k]
                    coul_ag_atom[k, t] = coul_ag_atom[t, k]

        np.savetxt(f"{dir_path}/{energy_keys[0]}.txt", lj_ab_ag_res, fmt="%.3f")
        np.savetxt(f"{dir_path}/{energy_keys[1]}.txt", lj_ab_ag_atom, fmt="%.3f")

        np.savetxt(f"{dir_path}/{energy_keys[2]}.txt", lj_ab_res, fmt="%.3f")
        np.savetxt(f"{dir_path}/{energy_keys[3]}.txt", lj_ab_atom, fmt="%.3f")

        np.savetxt(f"{dir_path}/{energy_keys[4]}.txt", lj_ag_res, fmt="%.3f")
        np.savetxt(f"{dir_path}/{energy_keys[5]}.txt", lj_ag_atom, fmt="%.3f")

        np.savetxt(f"{dir_path}/{energy_keys[6]}.txt", coul_ab_ag_res, fmt="%.3f")
        np.savetxt(f"{dir_path}/{energy_keys[7]}.txt", coul_ab_ag_atom, fmt="%.3f")

        np.savetxt(f"{dir_path}/{energy_keys[8]}.txt", coul_ab_res, fmt="%.3f")
        np.savetxt(f"{dir_path}/{energy_keys[9]}.txt", coul_ab_atom, fmt="%.3f")

        np.savetxt(f"{dir_path}/{energy_keys[10]}.txt", coul_ag_res, fmt="%.3f")
        np.savetxt(f"{dir_path}/{energy_keys[11]}.txt", coul_ag_atom, fmt="%.3f")

        print(pair)
        print(f"coul (Ab-Ag): {np.sum(coul_ab_ag_res)}")
        print(f"Lennard-Jones (Ab-Ag): {np.sum(lj_ab_ag_res)}")

        print(f"coul (Ab-Ab): {np.sum(coul_ab_res) / 2}")
        print(f"Lennard-Jones (Ab-Ab): {np.sum(lj_ab_res) / 2}")

        print(f"coul (Ag-Ag): {np.sum(coul_ag_res) / 2}")
        print(f"Lennard-Jones (Ag-Ag): {np.sum(lj_ag_res) / 2}")

        print(f"Total coul: {np.sum(coul_ab_ag_res) + np.sum(coul_ab_res) / 2 + np.sum(coul_ag_res) / 2}")
        print(f"Total Lennard-Jones: {np.sum(lj_ab_ag_res) + np.sum(lj_ab_res) / 2 + np.sum(lj_ag_res) / 2}")

    #
    # pair = '5xwd_H_D_A'
    # pair_path = f'{path}cg_input/cg_complex/{pair}/'
    # dir_path = f"./affinity_energy/{pair}"
    # # os.makedirs(dir_path, exist_ok=True)
    #
    # if len(pair.split('_')) == 4:
    #     pdb, H_chain, L_chain, ag_chain = pair.split('_')
    #     ab_chain_id = f"chainID Protein_{H_chain} Protein_{L_chain}"
    # else:
    #     pdb, ab_chain, ag_chain = pair.split('_')
    #     ab_chain_id = f"chainID Protein_{ab_chain}"
    #
    # # 读取 GROMACS 文件
    # gro_file = f"{pair_path}em.gro"  # 可以换成 trr 轨迹的最后一帧
    # tpr_file = f"{pair_path}em.tpr"  # 用于获取电荷和 Lennard-Jones 参数
    # u = mda.Universe(tpr_file, gro_file)
    #
    # # 选择抗体 (Protein_H + Protein_L) 和抗原 (Protein_W)
    # antibody = u.select_atoms(ab_chain_id)
    # antigen = u.select_atoms(f"chainID Protein_{ag_chain}")
    #
    # # 获取 CG 颗粒的残基
    # antibody_residues = antibody.residues
    # antigen_residues = antigen.residues
    #
    # # 生成能量矩阵
    # num_antibody = len(antibody_residues)
    # num_antigen = len(antigen_residues)
    #
    # # 初始化能量矩阵
    # coul_ab_ag = np.zeros((num_antibody, num_antigen))  # 抗体-抗原
    # lj_ab_ag = np.zeros((num_antibody, num_antigen))
    #
    # coul_ab_ab = np.zeros((num_antibody, num_antibody))  # 抗体-抗体
    # lj_ab_ab = np.zeros((num_antibody, num_antibody))
    #
    # coul_ag_ag = np.zeros((num_antigen, num_antigen))  # 抗原-抗原
    # lj_ag_ag = np.zeros((num_antigen, num_antigen))
    #
    # # CG Lennard-Jones & 库仑计算
    # k_e = 138.935485  # kJ·nm·mol^(-1)·e^(-2) （CG 力场中 ε ≈ 15 需要考虑）
    #
    # excluded_pairs = parse_topology_exclusions(u)
    #
    # # 计算抗体-抗原作用
    # for m, res1 in enumerate(antibody_residues):
    #     for n, res2 in enumerate(antigen_residues):
    #         e_c, e_lj = cal_interactions(res1, res2, excluded_pairs)
    #         coul_ab_ag[m, n] = e_c
    #         lj_ab_ag[m, n] = e_lj
    #
    # # 计算抗体-抗体内部作用
    # for x, res1 in enumerate(antibody_residues):
    #     for y, res2 in enumerate(antibody_residues):
    #         if x < y:  # 避免重复计算
    #             e_c, e_lj = cal_interactions(res1, res2, excluded_pairs)
    #             coul_ab_ab[x, y] = e_c
    #             lj_ab_ab[x, y] = e_lj
    #             coul_ab_ab[y, x] = e_c
    #             lj_ab_ab[y, x] = e_lj
    #
    # # 计算抗原-抗原内部作用
    # for t, res1 in enumerate(antigen_residues):
    #     for k, res2 in enumerate(antigen_residues):
    #         if t < k:  # 避免重复计算
    #             e_c, e_lj = cal_interactions(res1, res2, excluded_pairs)
    #             coul_ag_ag[t, k] = e_c
    #             lj_ag_ag[t, k] = e_lj
    #             coul_ag_ag[k, t] = e_c
    #             lj_ag_ag[k, t] = e_lj
    #
    # # 保存计算结果
    # np.savetxt(f"{dir_path}/ab_ag_coul_matrix.txt", coul_ab_ag, fmt="%.3f")
    # np.savetxt(f"{dir_path}/ab_ag_lj_matrix.txt", lj_ab_ag, fmt="%.3f")
    # np.savetxt(f"{dir_path}/ab_ab_coul_matrix.txt", coul_ab_ab, fmt="%.3f")
    # np.savetxt(f"{dir_path}/ab_ab_lj_matrix.txt", lj_ab_ab, fmt="%.3f")
    # np.savetxt(f"{dir_path}/ag_ag_coul_matrix.txt", coul_ag_ag, fmt="%.3f")
    # np.savetxt(f"{dir_path}/ag_ag_lj_matrix.txt", lj_ag_ag, fmt="%.3f")

    # 输出总能量
    # print(f"Python 计算的相互作用对数: {len(interaction_pairs)}")
    # print(f"Min r: {min(distances)}, Max r: {max(distances)}, Avg r: {np.mean(distances)}")
    # print(pair)
    # print(f"coul (Ab-Ag): {np.sum(coul_ab_ag_res)}")
    # print(f"Lennard-Jones (Ab-Ag): {np.sum(lj_ab_ag_res)}")
    #
    # print(f"coul (Ab-Ab): {np.sum(coul_ab_ab) / 2}")
    # print(f"Lennard-Jones (Ab-Ab): {np.sum(lj_ab_ab) / 2}")
    #
    # print(f"coul (Ag-Ag): {np.sum(coul_ag_ag) / 2}")
    # print(f"Lennard-Jones (Ag-Ag): {np.sum(lj_ag_ag) / 2}")
    #
    # print(f"Total coul: {np.sum(coul_ab_ag)+np.sum(coul_ab_ab) / 2 + np.sum(coul_ag_ag) / 2}")
    # print(f"Total Lennard-Jones: {np.sum(lj_ab_ag)+np.sum(lj_ab_ab) / 2 + np.sum(lj_ag_ag) / 2}")
