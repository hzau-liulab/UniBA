import glob
import os.path
import string
import numpy as np
import torch
import copy
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import warnings
# from rdkit import Chem
# from torchdrug import utils
from collections.abc import Sequence
# from torchdrug.utils import pretty
# from torchdrug.core import Registry as R
# from collections import defaultdict
# from torch_scatter import scatter_add, scatter_max, scatter_min
from torchdrug.data import Molecule, PackedMolecule, Dictionary, feature
from collections import defaultdict
# from torch_geometric.data import Data
# from utils.torchdrug.data import constant
# from utils.torchdrug import layers
from utils.cg_graphconstruct import *

# general mapping for protein
residue_symbol2abbr = {"GLY": "G", "ALA": "A", "SER": "S", "PRO": "P", "VAL": "V", "THR": "T", "CYS": "C", "ILE": "I",
                       "LEU": "L", "ASN": "N", "ASP": "D", "GLN": "Q", "LYS": "K", "GLU": "E", "MET": "M", "HIS": "H",
                       "PHE": "F", "ARG": "R", "TYR": "Y", "TRP": "W"}

abbr2residue_symbol = {v: k for k, v in residue_symbol2abbr.items()}

bead_info = {
    "AC1":  (0, "C", 0, 0, 0, 72, 0),
    "AC2":  (0, "C", 0, 0, 0, 72, 0),
    "C3":   (3, "C", 0, 0, 0, 72, 0),
    "C5":   (5, "C", 0, 0, 0, 72, 0),
    "N0":   (0, "N", 0, 0, 0, 72, 0),
    "Na":   (1, "N", 0, 1, 0, 72, 0),
    "Nd":   (1, "N", 1, 0, 0, 72, 0),
    "Nda":  (1, "N", 1, 1, 0, 72, 0),
    "P1":   (1, "P", 0, 0, 0, 72, 0),
    "P4":   (4, "P", 0, 0, 0, 72, 0),
    "P5":   (5, "P", 0, 0, 0, 72, 0),
    "Qa":   (1, "Q", 0, 1, 0, 72, +1),
    "Qd":   (1, "Q", 1, 0, 0, 72, -1),
    "SC4":  (4, "C", 0, 0, 1, 45, 0),
    "SC5":  (5, "C", 0, 0, 1, 45, 0),
    "SNd":  (1, "N", 1, 0, 1, 45, 0),
    "SP1":  (1, "P", 0, 0, 1, 45, 0)
}


class CG22_Protein(Molecule):
    # Protein class established with coarse-grain features
    # currently the nodes and edges between nodes are established based on CG
    _meta_types = {"node", "edge", "residue", "graph",
                   "node reference", "edge reference", "residue reference", "graph reference"}

    # stardard residue/atom id mapping
    residue2id = {"GLY": 0, "ALA": 1, "SER": 2, "PRO": 3, "VAL": 4, "THR": 5, "CYS": 6, "ILE": 7, "LEU": 8,
                  "ASN": 9, "ASP": 10, "GLN": 11, "LYS": 12, "GLU": 13, "MET": 14, "HIS": 15, "PHE": 16,
                  "ARG": 17, "TYR": 18, "TRP": 19}
    residue_symbol2id = {"G": 0, "A": 1, "S": 2, "P": 3, "V": 4, "T": 5, "C": 6, "I": 7, "L": 8, "N": 9,
                         "D": 10, "Q": 11, "K": 12, "E": 13, "M": 14, "H": 15, "F": 16, "R": 17, "Y": 18, "W": 19}
    atom_name2id = {"C": 0, "CA": 1, "CB": 2, "CD": 3, "CD1": 4, "CD2": 5, "CE": 6, "CE1": 7, "CE2": 8,
                    "CE3": 9, "CG": 10, "CG1": 11, "CG2": 12, "CH2": 13, "CZ": 14, "CZ2": 15, "CZ3": 16,
                    "N": 17, "ND1": 18, "ND2": 19, "NE": 20, "NE1": 21, "NE2": 22, "NH1": 23, "NH2": 24,
                    "NZ": 25, "O": 26, "OD1": 27, "OD2": 28, "OE1": 29, "OE2": 30, "OG": 31, "OG1": 32,
                    "OH": 33, "OXT": 34, "SD": 35, "SG": 36, "UNK": 37}
    alphabet2id = {c: i for i, c in enumerate(" " + string.ascii_uppercase + string.ascii_lowercase + string.digits)}

    id2residue = {v: k for k, v in residue2id.items()}
    id2residue_symbol = {v: k for k, v in residue_symbol2id.items()}
    id2atom_name = {v: k for k, v in atom_name2id.items()}
    id2alphabet = {v: k for k, v in alphabet2id.items()}

    # coarse-grained molecule id mapping
    type2onehot = {"P": [1, 0, 0, 0], "N": [0, 1, 0, 0], "C": [0, 0, 1, 0], "Q": [0, 0, 0, 1]}

    martini22_name2id = {"AC1": 0, "AC2": 1, "C3": 2, "C5": 3, "N0": 4, "Na": 5, "Nd": 6, "Nda": 7, "P1": 8,
                         # standard martini
                         "P4": 9, "P5": 10, "Qa": 11, "Qd": 12, "SC4": 13, "SC5": 14, "SNd": 15, "SP1": 16}
    elenedyn22_name2id = {"C1": 0, "C2": 1, "C3": 2, "C5": 3, "N0": 4, "Na": 5, "Nd": 6, "Nda": 7, "P1": 8,
                          # elastic network
                          "P4": 9, "P5": 10, "Qa": 11, "Qd": 12, "SC4": 13, "SC5": 14, "SNd": 15, "SP1": 16}
    martini22_bond2id = {'backbone_bonds': 0, 'sidechain_bonds': 1, 'sheet_bonds_3': 2, 'sheet_bonds_4': 3,
                         'constraints': 4}  # itp file bond types
    martini22_beadpos2id = {'BB': 0, 'SC1': 1, 'SC2': 2, 'SC3': 3, 'SC4': 4}  # bead position categories
    martini22_angletype2id = {'backbone_angles': 0, 'backbone_sidec_angles': 1, 'sidechain_angles': 2,
                              'backbone_dihedrals': 3}

    # bead_physchem
    num_beads = len(martini22_name2id)
    feat_dim = 1 + 4 + 1 + 1 + 1 + 1 + 1  # 10 维: polarity + one-hot(P/N/C/Q) + donor + acceptor + ring + mass + charge
    bead_physchem = torch.zeros((num_beads, feat_dim), dtype=torch.float32)

    for name, bead_id in martini22_name2id.items():
        polarity, btype, donor, acceptor, is_ring, mass, charge = bead_info[name]
        vec = [
            polarity / 5,
            *type2onehot[btype],
            donor,
            acceptor,
            is_ring,
            mass / 72,
            charge
        ]
        bead_physchem[bead_id] = torch.tensor(vec, dtype=torch.float32)

    id2martini22_name = {v: k for k, v in martini22_name2id.items()}
    id2elenedyn22_name = {v: k for k, v in elenedyn22_name2id.items()}
    id2martini22_bond = {v: k for k, v in martini22_bond2id.items()}
    id2martini22_beadpos = {v: k for k, v in martini22_beadpos2id.items()}
    id2martini22_angletype = {v: k for k, v in martini22_angletype2id.items()}

    def __init__(self, edge_list=None, bead_type=None, bead2residue=None, bond_type=None, residue_type=None,
                 aa_sequence=None, view=None, backbone_angles=None, backbone_sidec_angles=None,
                 sidechain_angles=None, backbone_dihedrals=None, bead2global=None, **kwargs):

        if 'atom_type' in kwargs.keys():
            # print(bead_type) # None
            bead_type = kwargs['atom_type']
            kwargs.pop('atom_type')
            super(CG22_Protein, self).__init__(edge_list, atom_type=bead_type, bond_type=bond_type, **kwargs)
        else:
            super(CG22_Protein, self).__init__(edge_list, atom_type=bead_type, bond_type=bond_type, **kwargs)

        residue_type, num_residue = self._standarize_num_residue(residue_type)
        self.num_residue = num_residue
        self.view = self._standarize_view(view)
        self.aa_sequence = aa_sequence

        # BBB (2nd as center)
        self.backbone_angles = self._standarize_angle(backbone_angles)
        # BBS (3rd as center)
        self.backbone_sidec_angles = self._standarize_angle(backbone_sidec_angles)
        # BSS (3rd as center)
        self.sidechain_angles = self._standarize_angle(sidechain_angles)
        # BBBB (2nd as center), it will only be provided for the consecutive four beads being the helix structure, which maintain the helix structure
        self.backbone_dihedrals = self._standarize_angle(backbone_dihedrals)

        # bead2residue index starts from 0
        bead2residue = self._standarize_attribute(bead2residue, self.num_node)
        bead2global = self._standarize_attribute(bead2global, self.num_node)

        with self.atom():
            self.bead2global = bead2global

        with self.atom():
            with self.residue_reference():
                self.bead2residue = bead2residue

        with self.residue():
            self.residue_type = residue_type  # tensor idx

    def residue(self):
        return self.context("residue")

    def residue_reference(self):
        return self.context("residue reference")

    @property
    def node_feature(self):
        if getattr(self, "view", "atom") == "atom":
            return self.atom_feature

    @node_feature.setter
    def node_feature(self, value):
        self.atom_feature = value

    @property
    def num_node(self):
        return self.num_atom

    @num_node.setter
    def num_node(self, value):
        self.num_atom = value

    def _check_attribute(self, key, value):
        super(CG22_Protein, self)._check_attribute(key, value)
        for type in self._meta_contexts:
            if type == "residue":
                if len(value) != self.num_residue:
                    raise ValueError("Expect residue attribute `%s` to have shape (%d, *), but found %s" %
                                     (key, self.num_residue, value.shape))
            elif type == "residue reference":
                is_valid = (value >= -1) & (value < self.num_residue)
                if not is_valid.all():
                    error_value = value[~is_valid]
                    raise ValueError(
                        "Expect residue reference in [-1, %d), but found %d" % (self.num_residue, error_value[0]))

    def _standarize_attribute(self, attribute, size, dtype=torch.long, default=0):
        if attribute is not None:
            attribute = torch.as_tensor(attribute, dtype=dtype, device=self.device)
        else:
            if isinstance(size, torch.Tensor):
                size = size.tolist()
            if not isinstance(size, Sequence):
                size = [size]
            attribute = torch.full(size, default, dtype=dtype, device=self.device)
        return attribute

    def _standarize_num_residue(self, residue_type):
        if residue_type is None:
            raise ValueError("`residue_type` should be provided")

        residue_type = torch.as_tensor(residue_type, dtype=torch.long, device=self.device)
        num_residue = torch.tensor(len(residue_type), device=self.device)
        return residue_type, num_residue

    def _standarize_angle(self, angle):
        if angle is None:
            return torch.zeros(0, device=self.device)
        else:
            return torch.as_tensor(angle, dtype=torch.long, device=self.device)

    def __setattr__(self, key, value):
        # https://www.runoob.com/python/python-func-setattr.html
        if key == "view" and value not in ["atom", "residue"]:
            raise ValueError("Expect `view` to be either `atom` or `residue`, but found `%s`" % value)
        return super(CG22_Protein, self).__setattr__(key, value)

    def _standarize_view(self, view):
        if view is None:
            if self.num_atom > 0:
                view = "atom"
            else:
                view = "residue"
        return view

    @classmethod
    # cgfile is the path storing the original generate CG files
    def from_cg_molecule(cls, cg_file, chains=None, chain_type=None, AA_num_threshold=3000):

        complete_check, cg_info = cls.cg_file_reader(cg_file, chains, chain_type, AA_num_threshold)
        # complete_check, cg_info = cls.cg_file_reader(cg_file, AA_num_threshold)
        if (not complete_check) and (isinstance(cg_info, str)):
            return complete_check, cg_info

        edge_list, bead_type, bead2residue, node_position, bead2global, bond_type, num_node, num_relation, residue_type, \
        aa_sequence, backbone_angles, backbone_sidec_angles, sidechain_angles, backbone_dihedrals = \
            cls.cg_feature_generator(cg_info, cg_file, chain_type, chains)

        return complete_check, \
               cls(edge_list, bead_type=bead_type, bead2residue=bead2residue, node_position=node_position,
                   bead2global=bead2global, bond_type=bond_type, num_node=num_node,
                   num_relation=num_relation, residue_type=residue_type,
                   aa_sequence=aa_sequence, backbone_angles=backbone_angles, backbone_sidec_angles=backbone_sidec_angles,
                   sidechain_angles=sidechain_angles, backbone_dihedrals=backbone_dihedrals)

    def clone(self):
        return type(self)(self.edge_list.clone(), bead_type=self.atom_type.clone(),
                          bead2residue=self.bead2residue.clone(), node_position=self.node_position.clone(),
                          bead2global=self.bead2global.clone(),
                          bond_type=self.bond_type.clone(), num_node=self.num_node, num_relation=self.num_relation,
                          residue_type=self.residue_type.clone(), aa_sequence=copy.copy(self.aa_sequence),
                          backbone_angles=self.backbone_angles.clone(),
                          backbone_sidec_angles=self.backbone_sidec_angles.clone(),
                          sidechain_angles=self.sidechain_angles.clone(),
                          backbone_dihedrals=self.backbone_dihedrals.clone())

    @classmethod
    def cg_file_reader(cls, cg_file, chains, chain_type, AA_num_threshold=3000):
        pdb = os.path.basename(cg_file)
        # topology files
        # itp_paths = sorted(glob.glob(os.path.join(cg_file, 'Protein_*.itp')))  # 'cg_*_M2.itp'  Protein_*
        itp_lines_dict = dict()
        for chain_type, chain_group in chains.items():
            for chain_id in chain_group:
                fname = f'Protein_{chain_id}'
                itp_path = f'{cg_file}/{fname}.itp'

                if not os.path.exists(itp_path):
                    raise FileNotFoundError(f"Missing ITP file: {itp_path}")

                with open(itp_path) as f:
                    itp_lines = f.readlines()
                itp_lines_dict[fname] = itp_lines  # keys: Protein_A, Protein_D (to identify chains)

        # CG pdb file
        cg_pdb_path = f'{cg_file}/cg_M2.pdb'
        # cg_pdb_path = f'{cg_file}/cg_{chain_type}_M2.pdb'
        if os.path.exists(cg_pdb_path):
            with open(cg_pdb_path) as f:
                cg_lines = f.readlines()
        else:
            raise FileNotFoundError(f"Error: Missing CG pdb file at {cg_pdb_path}")

        complete_check, cg_pdb_info = cleaning_cg_pdb(cg_lines, pdb, AA_num_threshold=AA_num_threshold)
        if not complete_check:  # not passing the check
            return complete_check, cg_pdb_info  # over-large protein info

        # 2. detach and classify the information contained in each itp chain file
        complete_check2 = True
        cg_itp_info = dict()
        for key in itp_lines_dict.keys():  # key: Protein_A
            chain_lines = itp_lines_dict[key]
            chain_dict_ = cleaning_cg_itp(chain_lines, pdb)
            complete_check2_, chain_dict = chain_dict_[0], chain_dict_[1]
            complete_check2 = complete_check2 & complete_check2_
            cg_itp_info[key] = chain_dict

        # complete check = True: passing the check, False: not passing the check
        return complete_check & complete_check2, {'cg_pdb_info': cg_pdb_info, 'cg_itp_info': cg_itp_info}

    @classmethod
    # adding the support to None bond info and angle info
    def cg_feature_generator(cls, cg_info, cg_file, chain_type, chains):
        global_chain_order = chains['pc1'] + chains['pc2']

        cg_pdb_info, cg_itp_info, pdb_name = cg_info['cg_pdb_info'], cg_info['cg_itp_info'], os.path.basename(cg_file)
        current_chain, chain_list = None, []
        residue2global, bead2global = {}, []
        cg_pdb_info_, node_position_, cb_token_list_ = defaultdict(list), defaultdict(list), defaultdict(list)

        chain_buckets = {c: [] for c in global_chain_order}
        for row in cg_pdb_info:
            # if row[0:4] == 'ATOM':
            if not row.startswith("ATOM"):
                continue

            chainid = row[21]
            chainid_ = f'Protein_{chainid}'
            cg_pdb_info_[chainid_].append(row)

            resid = row[22:30].strip()
            cb_token = f"{chainid}_{resid}"

            # cb_token = '{}_{}'.format(chainid, row[5:13].strip())
            cb_token_list_[chainid_].append(cb_token)
            # node_position is arranged based on the fixed 'BB'+'SC1'+'SC2'+'SC3' order
            node_position_[chainid_].append(get_coords(row))
            if current_chain != chainid:
                current_chain = chainid
                chain_list.append(chainid_)

            if chainid in chain_buckets:
                chain_buckets[chainid].append(row)

        pdb_chain_set, itp_chain_set = set(chain_list), set(cg_itp_info.keys())
        assert pdb_chain_set == itp_chain_set, "the chain identifiers contained in CG pdb file and itp file " \
                                               "are different for {}: {}, {} (should be consistent)". \
            format(pdb_name, pdb_chain_set, itp_chain_set)

        # ===== Step 1：计算每个 chain 的 residue 数（保留你原逻辑本质）=====
        chain_residue_count = {}
        for chain in global_chain_order:
            rows = chain_buckets.get(chain, [])
            tokens = [f"{chain}_{row[22:30].strip()}" for row in rows]
            chain_residue_count[chain] = len(set(tokens))

        # ===== Step 2：构建 offset =====
        offset_map = {}
        current_offset = 0
        for chain in global_chain_order:
            offset_map[chain] = current_offset
            current_offset += chain_residue_count[chain]

        chain_local_seen = {c: {} for c in global_chain_order}
        chain_local_count = {c: 0 for c in global_chain_order}

        for row in cg_pdb_info:
            if not row.startswith("ATOM"):
                continue

            chain = row[21]
            if chain not in offset_map:
                continue

            resid = row[22:30].strip()
            token = f"{chain}_{resid}"

            if token not in chain_local_seen[chain]:
                gid = offset_map[chain] + chain_local_count[chain]
                chain_local_seen[chain][token] = gid
                chain_local_count[chain] += 1
                residue2global[token] = gid

            bead2global.append(chain_local_seen[chain][token])

        # if chain_type == 'pc1':
        #     chain_type_map = {0: 0, 1: 1}
        # elif chain_type == 'pc2':
        #     chain_type_map = {0: 2}

        bead_type, edge_list, bead2residue, res_serial_list = [], [], [], []
        chain_bead_num, chain_aa_num, residue_type = [], [], []

        # chain_list: ['Protein_A', 'Protein_D']
        for chain_idx, chain in enumerate(chain_list):
            chain_bead_info = cg_itp_info[chain]['atom']
            chain_aa_info = cg_itp_info[chain]['sequence']
            chain_pdb_info = cg_pdb_info_[chain]

            pdb_bead_num = len(chain_pdb_info)
            itp_bead_num = len(chain_bead_info)

            assert pdb_bead_num == itp_bead_num, "the number of bead contained in CG pdb file and itp file " \
                                                 "are different for {}: {}, {} (should be consistent)". \
                format(pdb_name, pdb_bead_num, itp_bead_num)

            chain_bead_num.append(itp_bead_num)
            chain_aa_num.append(len(chain_aa_info))
            residue_type.append(chain_aa_info)

            current_res_serial = None
            for row in chain_bead_info:
                row = row.split()  # each row represents a new bead
                # bead name, residue serial number (re-numbered from 1),
                # residue name, bead position category (indicating BB/SC1/SC2/SC3): 12, 1, 4, 0
                bead, res_serial, res, bead_pos = cls.martini22_name2id[row[1]], int(row[2]), cls.residue2id[row[3]], \
                                                  cls.martini22_beadpos2id[row[4]]
                if current_res_serial != res_serial:
                    current_res_serial = res_serial
                    res_serial_list.append(
                        res_serial)  # record the aa serial numbers of each chain following the order of chain_list
                bead2residue.append(len(res_serial_list) - 1)
                bead_type.append([bead, res, bead_pos])
                # transform bead type and corresponding residue type and bead position category into idx
        # print(chain_bead_num, chain_aa_num) # [248, 191] [108, 87]

        node_position, cb_token_list = [], []
        for chain in chain_list:
            node_position.extend(node_position_[chain])
            cb_token_list.extend(cb_token_list_[chain])

        chain_bead_cumnum = np.cumsum([0] + chain_bead_num[:-1])  # [0 248]
        #  ['backbone_bonds', 'sidechain_bonds', 'sheet_bonds_3', 'sheet_bonds_4', 'constraints']
        bond_keys = list(cls.martini22_bond2id.keys())
        # ['backbone_angles', 'backbone_sidec_angles', 'sidechain_angles', 'backbone_dihedrals']
        backbone_angles, backbone_sidec_angles, sidechain_angles, backbone_dihedrals = [], [], [], []
        angle_keys = list(cls.martini22_angletype2id)

        for chain_id, chain in enumerate(chain_list):  # get one chain itp
            chain_itp_info = cg_itp_info[chain]
            cum_bead_id = chain_bead_cumnum[chain_id]  # cumulative bead serial number for current chain
            chain_length = chain_bead_num[chain_id]
            for bond_key in bond_keys:  # get one type of bond for current chain
                rows = chain_itp_info[bond_key]  # get rows for current type of bond
                current_type = cls.martini22_bond2id[bond_key]  # current bond type
                for row in rows:
                    row = row.split()
                    if int(row[0]) <= chain_length and int(row[1]) <= chain_length:
                        h, t = int(row[0]) + cum_bead_id - 1, int(
                            row[1]) + cum_bead_id - 1  # make edge_list index starting from 0
                        edge_list += [[h, t, current_type], [t, h, current_type]]

            for angle_key in angle_keys:
                rows = chain_itp_info[angle_key]
                for row in rows:
                    row = row.split()
                    if 'dihedral' in angle_key:
                        # print(pdb_name, chain, row)
                        if all(map(lambda x: int(x) <= chain_length, row[:4])):
                            _1, _2, _3, _4 = int(row[0]) + cum_bead_id - 1, int(row[1]) + cum_bead_id - 1, int(
                                row[2]) + cum_bead_id - 1, int(
                                row[3]) + cum_bead_id - 1  # make angle node index start from 0
                            locals()[angle_key] += [[_1, _2, _3, _4]]
                    else:
                        if all(map(lambda x: int(x) <= chain_length, row[:3])):
                            _1, _2, _3 = int(row[0]) + cum_bead_id - 1, int(row[1]) + cum_bead_id - 1, int(
                                row[2]) + cum_bead_id - 1
                            locals()[angle_key] += [[_1, _2, _3]]

        assert len(edge_list), "edge information provided in itp files of protein {} is empty".format(pdb_name)
        edge_list = torch.tensor(sorted(edge_list))  # sorted: fix the edge_list order
        bond_type = torch.tensor(edge_list)[:, -1]
        bead_type = torch.tensor(bead_type)
        bead2residue = torch.tensor(bead2residue)
        bead2global = torch.tensor(bead2global)

        num_node, num_relation = sum(chain_bead_num), len(cls.martini22_bond2id)
        aa_sequence = '.'.join(residue_type)
        residue_type = torch.tensor([cls.residue_symbol2id[i] for chain in residue_type for i in chain])

        node_position = np.array(node_position)
        bead2global = np.array(bead2global)
        backbone_angles = torch.tensor(backbone_angles)
        backbone_sidec_angles = torch.tensor(backbone_sidec_angles)
        sidechain_angles = torch.tensor(sidechain_angles)
        backbone_dihedrals = torch.tensor(backbone_dihedrals)

        chain_bead_num_ = [len(cg_pdb_info_[chain]) for chain in chain_list]
        assert chain_bead_num_ == chain_bead_num, "the bead number for each chain between CG pdb and itp files " \
                                                  "is different for {}, the number in itp files is {}, " \
                                                  "while the number in CG pdb is {}". \
            format(pdb_name, chain_bead_num, chain_bead_num_)

        return edge_list, bead_type, bead2residue, node_position, bead2global, bond_type, num_node, num_relation, residue_type, \
               aa_sequence, backbone_angles, backbone_sidec_angles, sidechain_angles, backbone_dihedrals

    def protein_cropping(self, cropping_threshold=10, contact_threshold=8.5, compact=True):
        # this function is performed prior to the 'transform' functions
        aa_num_chain = [len(chain) for chain in self.aa_sequence.split('.')]
        assert self.residue_type.size(0) == sum(aa_num_chain) == int(self.num_residue), \
            "the residue number in residue_type and aa_sequence should be the same"

        if len(aa_num_chain) >= 2:
            # default last chain name is antigen chain
            target_length = sum(aa_num_chain[:-1])

            BB_mask = (self.atom_type[:, 2] == self.martini22_beadpos2id['BB'])
            BB_position = self.node_position[BB_mask].unsqueeze(0)
            aa_square_distance = square_distance(BB_position, BB_position).squeeze(0)

            contact_matrix = aa_square_distance[:target_length, target_length:]
            closest_distance = torch.sqrt(torch.min(contact_matrix))

            contact_matrix = (contact_matrix <= contact_threshold ** 2)  # 5A**2
            contact_matrix = torch.nonzero(contact_matrix)

            contact_matrix[:, 1] += target_length
            contact_aa_index = torch.cat([contact_matrix[:, 0], contact_matrix[:, 1]])
            contact_aa_index = torch.unique(contact_aa_index, sorted=True)  # remove the duplicated AA indices

            retain_matrix = (aa_square_distance[contact_aa_index] <= cropping_threshold ** 2)

            retain_aa_index = torch.unique(torch.nonzero(retain_matrix)[:, 1], sorted=True)

            return self.residue_mask(retain_aa_index, compact=compact, intermol_mat=contact_matrix), closest_distance

        else:
            print('current protein cropping does not support proteins with chain number over two')
            raise NotImplementedError

    def residue_mask(self, index, compact=False, intermol_mat=None):
        """
        Return a masked protein based on the specified residues.
        Note the compact option is applied to both residue and atom ids.
        Parameters:
            index (array_like): residue index: mask[start:end] = True
            start and end represent the end point AAs to be retained
            compact (bool, optional): compact residue ids or not
        Returns:
            Protein
        """
        index = self._standarize_index(index, self.num_residue)
        if (torch.diff(index) <= 0).any():
            warnings.warn("`residue_mask()` is called to re-order the residues. This will change the protein sequence."
                          "If this is not desired, you might have passed a wrong index to this function.")

        residue_mapping = -torch.ones(self.num_residue, dtype=torch.long, device=self.device)
        residue_mapping[index] = torch.arange(len(index), device=self.device)

        node_index = residue_mapping[self.bead2residue] >= 0
        node_index = self._standarize_index(node_index, self.num_node)  # the node-based mask
        # energy_matrix = self.energy_matrix[node_index][:, node_index]

        mapping = -torch.ones(self.num_node, dtype=torch.long, device=self.device)
        if compact:
            mapping[node_index] = torch.arange(len(node_index), device=self.device)
            num_node = len(node_index)
        else:
            mapping[node_index] = node_index
            num_node = self.num_node

        edge_list = self.edge_list.clone()
        edge_list[:, :2] = mapping[edge_list[:, :2]]
        edge_index = (edge_list[:, :2] >= 0).all(dim=-1)
        edge_index = self._standarize_index(edge_index, self.num_edge)  # remove edges with '-1' nodes

        if compact:
            data_dict, meta_dict = self.data_mask(node_index, edge_index, residue_index=index)
        else:
            data_dict, meta_dict = self.data_mask(edge_index=edge_index)

        # truncate the angle information
        backbone_angles = self.angle_mapping(self.backbone_angles, mapping)
        backbone_sidec_angles = self.angle_mapping(self.backbone_sidec_angles, mapping)
        sidechain_angles = self.angle_mapping(self.sidechain_angles, mapping)
        backbone_dihedrals = self.angle_mapping(self.backbone_dihedrals, mapping)

        # retrieving the required information for initialing an on-the-fly truncated protein class
        bead_type, bead2residue, node_position, bond_type, residue_type, bead2global = \
            data_dict['atom_type'], data_dict['bead2residue'], data_dict['node_position'], data_dict['bond_type'], \
            data_dict['residue_type'], data_dict['bead2global']
        #
        # if hasattr(self, "node_chain_res"):
        #     node_chain_res = [self.node_chain_res[i] for i in node_index.tolist()]

        # do not need to return core region AA pair information
        if intermol_mat == None:
            return type(self)(edge_list[edge_index], bead_type=bead_type, bead2residue=bead2residue,
                              node_position=node_position, bead2global=bead2global, bond_type=bond_type,
                              num_node=num_node, num_relation=self.num_relation, residue_type=residue_type, view=self.view,
                              backbone_angles=backbone_angles, backbone_sidec_angles=backbone_sidec_angles,
                              sidechain_angles=sidechain_angles, backbone_dihedrals=backbone_dihedrals)
        else:
            assert torch.all(torch.isin(intermol_mat, index)), 'All elements in intermol_mat should in elements of ' \
                                                               'the index input of this cropping function.'
            if compact:
                intermol_mat = residue_mapping[intermol_mat]

            return type(self)(edge_list[edge_index], bead_type=bead_type, bead2residue=bead2residue,
                              node_position=node_position, bead2global=bead2global, bond_type=bond_type,
                              num_node=num_node, num_relation=self.num_relation, residue_type=residue_type, view=self.view,
                              backbone_angles=backbone_angles, backbone_sidec_angles=backbone_sidec_angles,
                              sidechain_angles=sidechain_angles, backbone_dihedrals=backbone_dihedrals,
                              intermol_mat=intermol_mat)

    def angle_mapping(self, angles, mapping):
        if angles.size(0) > 0:
            angle_info = angles.clone()
            angle_info = mapping[angle_info]
            angle_index = (angle_info >= 0).all(dim=-1)
            angle_index = self._standarize_index(angle_index, angles.size(0))
            return angle_info[angle_index]
        else:
            return angles.clone()

    def data_mask(self, node_index=None, edge_index=None, residue_index=None, graph_index=None, include=None,
                  exclude=None):
        data_dict, meta_dict = super(CG22_Protein, self).data_mask(node_index, edge_index, graph_index=graph_index,
                                                                   include=include, exclude=exclude)

        residue_mapping = None
        for k, v in data_dict.items():
            for type in meta_dict[k]:
                if type == "residue" and residue_index is not None:
                    if v.is_sparse:
                        v = v.to_dense()[residue_index].to_sparse()
                    else:
                        v = v[residue_index]
                elif type == "residue reference" and residue_index is not None:
                    if residue_mapping is None:
                        residue_mapping = self._get_mapping(residue_index, self.num_residue)
                    v = residue_mapping[v]
            data_dict[k] = v

        return data_dict, meta_dict

    # return a subgraph based on the specified residues
    def subresidue(self, index):
        return self.residue_mask(index, compact=True)

    @classmethod
    def pack(cls, graphs):
        # * adding the record of residue number and view information *
        # * adding the support of bead2residue cumulative sum *
        # * adding the support of angle information from CG itp files *
        # * adding the support of intermolcular matrix information from the AA-based distance matric from the cropping function *
        edge_list = []
        edge_weight = []
        num_nodes = []
        num_edges = []
        num_residues = []
        backbone_angles = []
        backbone_sidec_angles = []
        sidechain_angles = []
        backbone_dihedrals = []
        intermol_mat = []
        # pack the information the input graphs
        num_cum_node = 0
        num_cum_edge = 0
        num_cum_residue = 0
        num_graph = 0
        data_dict = defaultdict(list)
        meta_dict = graphs[0].meta_dict
        view = graphs[0].view
        for graph in graphs:
            edge_list.append(graph.edge_list)
            edge_weight.append(graph.edge_weight)
            num_nodes.append(graph.num_node)
            num_edges.append(graph.num_edge)
            num_residues.append(graph.num_residue)
            backbone_angles.append(graph.backbone_angles + num_cum_node)
            backbone_sidec_angles.append(graph.backbone_sidec_angles + num_cum_node)
            sidechain_angles.append(graph.sidechain_angles + num_cum_node)
            backbone_dihedrals.append(graph.backbone_dihedrals + num_cum_node)
            # ** note that the incremental information for angles is num_cum_node (bead node-based) **
            # ** while that for intermolecular matrix is num_cum_residue (because the distance calculation is AA-based) **
            intermol_mat.append(graph.intermol_mat + num_cum_residue)
            for k, v in graph.data_dict.items():
                for type in meta_dict[k]:
                    if type == "graph":
                        v = v.unsqueeze(0)
                    elif type == "node reference":
                        v = torch.where(v != -1, v + num_cum_node, -1)
                    elif type == "edge reference":
                        v = torch.where(v != -1, v + num_cum_edge, -1)
                    elif type == "residue reference":
                        # bead2residue cumulative sum
                        v = torch.where(v != -1, v + num_cum_residue, -1)
                    elif type == "graph reference":
                        v = torch.where(v != -1, v + num_graph, -1)
                data_dict[k].append(v)
            num_cum_node += graph.num_node  # one value
            num_cum_edge += graph.num_edge  # one value
            num_cum_residue += graph.num_residue  # one value
            num_graph += 1

        edge_list = torch.cat(edge_list)
        edge_weight = torch.cat(edge_weight)
        backbone_angles = torch.cat(backbone_angles)
        backbone_sidec_angles = torch.cat(backbone_sidec_angles)
        sidechain_angles = torch.cat(sidechain_angles)
        backbone_dihedrals = torch.cat(backbone_dihedrals)
        intermol_mat = torch.cat(intermol_mat)

        # data_dict.keys: dict_keys(['atom_type', 'formal_charge', 'explicit_hs', 'chiral_tag', 'radical_electrons',
        # 'atom_map', 'node_position', 'bond_type', 'bond_stereo', 'stereo_atoms', 'bead2residue', 'residue_type'])
        data_dict = {k: torch.cat(v) for k, v in data_dict.items()}

        return cls.packed_type(
            edge_list, edge_weight=edge_weight, num_relation=graphs[0].num_relation, num_nodes=num_nodes,
            num_edges=num_edges, num_residues=num_residues, view=view,
            backbone_angles=backbone_angles, backbone_sidec_angles=backbone_sidec_angles,
            sidechain_angles=sidechain_angles, backbone_dihedrals=backbone_dihedrals,
            intermol_mat=intermol_mat, meta_dict=meta_dict, **data_dict)

    def __repr__(self):
        fields = ["num_atom=%d" % self.num_node, "num_bond=%d" % self.num_edge,
                  "num_residue=%d" % self.num_residue]
        if self.device.type != "cpu":
            fields.append("device='%s'" % self.device)
        return "%s(%s)" % (self.__class__.__name__, ", ".join(fields))


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2, dim=-1) + sum(dst**2, dim=-1) - 2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def cleaning_cg_pdb(cg_lines, pdb, AA_num_threshold=3000):
    screened_list = []  # the list containing the cleaned CG lines
    bead_pos_list = []  # for recording the bead list for current CG file
    BB_num_threshold = AA_num_threshold  # threshold for screening out over-large proteins

    for row in cg_lines:
        resname = row[17:20].strip()  # residue name

        if row[0:4] == 'ATOM' and len(resname) == 3:
            # a temporary correcting of CG proteins (i.e., chain a/d to chain B)
            # if row[21].islower():
            #     row = row[:21] + 'B' + row[22:]
            screened_list.append(row)
            bead_pos_list.append(row[12:16].strip())
        # a check about the non-residue atoms
        elif row[0:4] == 'ATOM' and len(resname) != 3:
            print('Non-residue atoms:', pdb, row)
        elif row[0:3] == 'TER':
            screened_list.append(row[0:3] + '\n')
        else:
            continue

    # over-large protein check
    BB_num = np.sum(np.array(bead_pos_list) == 'BB')  # each residue only has one BB bead
    if BB_num > BB_num_threshold:
        return False, 'Over-large protein {} is ignored'.format(pdb)

    return True, screened_list


def cleaning_cg_itp(chain_lines, pdb):
    # * in some cases, sheet_bonds_3 and sheet_bonds_4 will not exist in itp along with the tag row, *
    # * for backbone dihedrals, sometimes the rows will not exist but the tag is remained, which needs to be considered *

    complete_check = True
    chain_dict = dict()
    tag = None
    # Note: within the types martini22_bond2id = {'backbone_bonds': 0, 'sidechain_bonds': 1, 'sheet_bonds_3': 2, 'sheet_bonds_4': 3, 'constraints': 4}
    # only type 0 will definitely exist for each protein in itp bond types
    # thus, we also need to consider the case that a part of bond types does not exist in itp files
    backbone_bonds, sidechain_bonds, sheet_bonds_3, sheet_bonds_4, constraints = [], [], [], [], []
    backbone_angles, backbone_sidec_angles, sidechain_angles, backbone_dihedrals = [], [], [], []

    for row_num, row in enumerate(chain_lines):
        # ** with current logic, each 'elif' branch will not conflict with each other **
        if row_num != len(chain_lines) - 1:
            next_row = chain_lines[row_num + 1].strip()  # last row of itp file: '#endif'
        else:  # the last row of current itp file
            next_row = ''
            # further check about the '#endif' tag for finding itp files which may be incomplete (should end with '#endif' tag)
            if row.strip() != '#endif':
                print('current itp files of protein {} may be incomplete'.format(pdb))
                complete_check = False

        # recording AA sequence
        if row.strip() == '; Sequence:':
            chain_dict['sequence'] = next_row[2:]

        # recording secondary structure
        elif row.strip() == '; Secondary Structure:':
            chain_dict['secondary_structure'] = next_row[2:]

        # recording CG beads
        elif row.strip() == '[ atoms ]':
            aa_record_tag = False
            tag = 'atom'
            atom = []
            if 'sequence' not in chain_dict.keys():
                aa_record_tag = True
                current_resid = None
                aa_sequence = []
        # considering the case that 'sequence' does not exist in itp files
        elif tag == 'atom':
            if next_row == '':
                row = row.strip()
                atom.append(row)
                tag = None
                # recording AA types if they are not provided in itp file
                if aa_record_tag == True:
                    row = row.split()
                    res_id, res_name = row[2], row[3]
                    if res_id != current_resid:
                        current_resid = res_id
                        aa_sequence.append(residue_symbol2abbr[res_name])
                    # this is the end of the 'atom' rows
                    chain_dict['sequence'] = ''.join(aa_sequence)
            else:
                row = row.strip()
                atom.append(row)
                # recording AA types if they are not provided in itp file
                if aa_record_tag == True:
                    row = row.split()
                    res_id, res_name = row[2], row[3]
                    if res_id != current_resid:
                        current_resid = res_id
                        aa_sequence.append(residue_symbol2abbr[res_name])

        # recording backbone bonds with flexible bond length (defined by martini22_aminoacid.itp)
        elif row.strip() == '; Backbone bonds':
            # skip the case that the topological tag exists but no content exists
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                tag = None
            else:
                tag = 'backbone_bonds'
        elif tag == 'backbone_bonds':
            # if chain_lines[row_num+1].strip() == '; Sidechain bonds':
            # 1. (len(next_row) > 0 and next_row[0] == ';') for handling cases like '; Sidechain bonds'
            # 2. next_row == '' for handling empty row cases
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                row = row.strip()
                backbone_bonds.append(row)
                tag = None
            else:
                backbone_bonds.append(row.strip())

        # recording bonds between backbone and side chain with flexible bond length (defined by martini22_aminoacid.itp)
        elif row.strip() == '; Sidechain bonds':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                tag = None
            else:
                tag = 'sidechain_bonds'
        elif tag == 'sidechain_bonds':
            # if chain_lines[row_num+1].strip() == '; Short elastic bonds for extended regions':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                row = row.strip()
                sidechain_bonds.append(row)
                tag = None
            else:
                sidechain_bonds.append(row.strip())

        # recording virtual bonds for sheet secondary structure (based on three AA distance)
        elif row.strip() == '; Short elastic bonds for extended regions':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                tag = None
            else:
                tag = 'sheet_bonds_3'
        elif tag == 'sheet_bonds_3':
            # if chain_lines[row_num+1].strip() == '; Long elastic bonds for extended regions':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                row = row.strip()
                sheet_bonds_3.append(row)
                tag = None
            else:
                sheet_bonds_3.append(row.strip())

        # recording virtual bonds for sheet secondary structure (based on four AA distance)
        elif row.strip() == '; Long elastic bonds for extended regions':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                tag = None
            else:
                tag = 'sheet_bonds_4'
        elif tag == 'sheet_bonds_4':
            # if chain_lines[row_num+1].strip() == '':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                row = row.strip()
                sheet_bonds_4.append(row)
                tag = None
            else:
                sheet_bonds_4.append(row.strip())

        # recording bonds between backbones, between backbone and side chain, and between side chains with fixed bond length (defined by martini22_aminoacid.itp)
        # for some types of residues, the 'constraints' could be empty, thus we also need to consider the case that the 'constraints' could be empty
        elif row.strip() == '[ constraints ]':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                tag = None
            else:
                tag = 'constraints'
        elif tag == 'constraints':
            # if chain_lines[row_num+1].strip() == '':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                row = row.strip()
                constraints.append(row)
                tag = None
            else:
                constraints.append(row.strip())

        # recording backbone angles
        elif row.strip() == '; Backbone angles':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                tag = None
            else:
                tag = 'backbone_angles'
        elif tag == 'backbone_angles':
            # if chain_lines[row_num+1].strip() == '; Backbone-sidechain angles':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                row = row.strip()
                backbone_angles.append(row)
                tag = None
            else:
                backbone_angles.append(row.strip())

        # recording backbone-sidechain angles
        elif row.strip() == '; Backbone-sidechain angles':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                tag = None
            else:
                tag = 'backbone_sidec_angles'
        elif tag == 'backbone_sidec_angles':
            # if chain_lines[row_num+1].strip() == '; Sidechain angles':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                row = row.strip()
                backbone_sidec_angles.append(row)
                tag = None
            else:
                backbone_sidec_angles.append(row.strip())

        # recording sidechain angles
        elif row.strip() == '; Sidechain angles':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                tag = None
            else:
                tag = 'sidechain_angles'
        elif tag == 'sidechain_angles':
            # if chain_lines[row_num+1].strip() == '':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                row = row.strip()
                sidechain_angles.append(row)
                tag = None
            else:
                sidechain_angles.append(row.strip())

        # recording backbone dihedrals
        # ** the side chain diredrals are not necessarily recorded, as only side chains with aromatic nucleus contain at least three side chain beads so that this dihedral can be calculated **
        # ** however, the aromatic nucleus side chain is the plane structure with this dehedral value 0, which cannot be used for distinguishing (the types of) each other **
        elif row.strip() == '; Backbone dihedrals':
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                tag = None
            else:
                tag = 'backbone_dihedrals'
        elif tag == 'backbone_dihedrals':
            # if chain_lines[row_num+1].strip() == '; Sidechain improper dihedrals':
            # print(len(next_row), next_row, next_row[0])
            if (len(next_row) > 0 and next_row[0] == ';') or (next_row == ''):
                row = row.strip()
                backbone_dihedrals.append(row)
                tag = None
            else:
                backbone_dihedrals.append(row.strip())

        else:
            continue

    # print(atom[0], '///' , atom[-1], '///' ,len(atom))
    # print(backbone_bonds[0], '///' ,backbone_bonds[-1], '///' ,len(backbone_bonds))
    # print(sidechain_bonds[0], '///', sidechain_bonds[-1], '///', len(sidechain_bonds))
    # print(sheet_bonds_3[0] if len(sheet_bonds_3) > 0 else sheet_bonds_3, '///', sheet_bonds_3[-1] if len(sheet_bonds_3) > 0 else sheet_bonds_3, '///', len(sheet_bonds_3))
    # print(sheet_bonds_4[0] if len(sheet_bonds_4) > 0 else sheet_bonds_4, '///', sheet_bonds_4[-1] if len(sheet_bonds_4) > 0 else sheet_bonds_4, '///', len(sheet_bonds_4))
    # print(constraints[0], '///', constraints[-1], '///', len(constraints))
    # print(backbone_angles[0], '///', backbone_angles[-1], '///', len(backbone_angles))
    # print(backbone_sidec_angles[0], '///', backbone_sidec_angles[-1], '///', len(backbone_sidec_angles))
    # print(sidechain_angles[0], '///', sidechain_angles[-1], '///', len(sidechain_angles))
    # print(backbone_dihedrals[0], '///', backbone_dihedrals[-1], '///', len(backbone_dihedrals))

    # (1) backbone_angles: BBB (2nd as center_pos, B)
    # (2) backbone_sidec_angles: BBS (3rd as center_pos, S)
    # (3) sidechain_angles: BSS (3rd as center_pos, S)
    # (4) backbone_dihedrals: BBBB (2nd as center_pos, B), it will only be provided for the consecutive four beads being the helix structure, which maintain the helix structure
    complete_check = (len(backbone_bonds) != 0) & (len(sidechain_bonds) != 0) & (len(backbone_angles) != 0) & (
            len(backbone_sidec_angles) != 0) & complete_check
    # (1) 'constraints' could be empty if some proteins are lack of specific kinds of residues
    # (2) 'backbone_bonds' could also be empty, in the case like residues of current whole chain belong the helix structure (these backbone bonds will be assigned to 'constraints')
    # (3) specifically, 'bonds' and 'constraints' can both be treated as the representation of bond length
    # (4) the bond length of backbone bonds will be assigned to 'backbone_bonds' or 'constraints' based on the secondary structure:
    # if one of two beads within a backbone bond is identified as the helix structure, this bond will be 'constraints' otherwise be 'backbone_bonds'
    # (5) the bond length of side chain bonds will be assigned to 'sidechain_bonds' or 'constraints' according to the AA type:
    # the specific definition of side chain bond assignment is based on martini22_aminoacid.itp file
    # (6) the bond length refers to the geometric distance between two beads

    # beads and bonds
    chain_dict['atom'], chain_dict['backbone_bonds'], chain_dict['sidechain_bonds'], chain_dict['sheet_bonds_3'], \
    chain_dict['sheet_bonds_4'], chain_dict['constraints'] = \
        atom, backbone_bonds, sidechain_bonds, sheet_bonds_3, sheet_bonds_4, constraints
    # angles and dihedrals
    chain_dict['backbone_angles'], chain_dict['backbone_sidec_angles'], chain_dict['sidechain_angles'], chain_dict[
        'backbone_dihedrals'] = \
        backbone_angles, backbone_sidec_angles, sidechain_angles, backbone_dihedrals

    return complete_check, chain_dict


def get_coords(line):
    if len(line[30:38].strip()) != 0:
        x = float(line[30:38].strip())
    else:
        x = float('nan')  # nan in math format
    if len(line[38:46].strip()) != 0:
        y = float(line[38:46].strip())
    else:
        y = float('nan')
    if len(line[46:54].strip()) != 0:
        z = float(line[46:54].strip())
    else:
        z = float('nan')
    return x, y, z
