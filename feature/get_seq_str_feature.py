import sys
import os
proj_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(proj_dir)
import torch
from utils.pdb_utils import *
from arg_parse import *
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
import scipy.stats as stats
import glob
# import ablang
import esm
import esm.inverse_folding
from sklearn.preprocessing import StandardScaler
import MDAnalysis as mda
# from Bio.Data.IUPACData import protein_letters_3to1
import joblib
import json
from Bio.Data.IUPACData import protein_letters_3to1


class RASA_feat(PDBfuc):
    def __init__(self, pdbfile=None, max_asa=None, dsspexe=None, dssp_file=None):
        super().__init__(pdbfile)
        self.dssp_file = dssp_file
        self.dsspexe = dsspexe
        self.res_max_asa = max_asa
        self.pdbfile = pdbfile
        # self.dssp(dssp_file, pdbfile)

        if dssp_file:
            if not os.path.exists(dssp_file):
                self.dssp(dssp_file, pdbfile)
        else:
            print("Warning: dssp_file not provided, skipping DSSP execution.")

    def dssp(self, dssp_file, pdbfile):
        """
        excute dssp program
        """
        if not pdbfile:
            print("Error: PDB file not provided.")
            return None

        if not self.dsspexe or not os.path.exists(self.dsspexe):
            print("Error: DSSP executable not found. Please provide a valid path.")
            return None

        result = os.system(f'{self.dsspexe} -i {pdbfile} -o {dssp_file}')
        if result == 0 and os.path.exists(dssp_file):
            self.dssp_file = dssp_file
            return dssp_file
        else:
            print("Error: DSSP execution failed or output file not created.")
            return None

    def rasa(self, dssp_file):
        """
        return: str_array col.1=>res col.2=>shellAcc col.3=>Rinacc col.4=>pocketness
        """
        res_rasa = []
        dssp_file = dssp_file if os.path.exists(dssp_file) else self.dssp(dssp_file, self.pdbfile)
        with open(dssp_file, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines[28:]):
                aa = line[13]
                if aa == "!" or aa == "*" or aa == "X":
                    continue
                res_rasa.append(float(line[34:38].strip()) / self.res_max_asa[aa])
        return np.array(res_rasa).reshape(-1, 1)

    def res_array(self):
        return self.rasa(self.dssp_file)


def dp_feat(pc):
    dpfile = f'./features/STR_feature/DP/{pc}.tbl'
    if not os.path.exists(dpfile):
        print("Error: Please ensure that the PSAIA software is "
              "installed and has been run to generate the required file.")
    out = np.loadtxt(dpfile, skiprows=16, usecols=(18, 24), dtype=float)
    return out


class DP_feat(PDBfuc):
    def __init__(self, pdbfile=None, pc=None, psaexe=None, psafile=None):
        super().__init__(pdbfile)
        self.psafile = psafile
        self.psaexe = psaexe
        self.pc = pc
        self.pdbfile = pdbfile
        # self.psa(pc, pdbfile)

    def config_set(self):
        config_path = f"{self.psaexe}/Examples/psa.cfg"
        with open(config_path, "r") as file:
            config_content = file.readlines()

        replacements = {
            "standard_asa:": f"standard_asa: {self.psaexe}/amac_data/natural_asa.asa\n",
            "hydro_file:": f"hydro_file: {self.psaexe}/amac_data/hydrophobicity.hpb\n",
            "radii_filename:": f"radii_filename: {self.psaexe}/amac_data/chothia.radii\n",
            "output_dir:": f"output_dir: {self.psaexe}/Examples/\n"
        }

        new_config_content = []
        for line in config_content:
            modified = False
            for key, replacement in replacements.items():
                if line.startswith(key):
                    new_config_content.append(replacement)
                    modified = True
                    break
            if not modified:
                new_config_content.append(line)
        with open(config_path, "w") as file:
            file.writelines(new_config_content)
        return config_path

    def pdblist_set(self, pdbfile):
        pdblist_path = f"{self.psaexe}/Examples/list.fls"
        with open(pdblist_path, "w") as file:
            file.write(pdbfile)
        return pdblist_path

    def get_psafile(self, pc, pdbfile=None):
        """
        excute psa program
        """
        if pdbfile is not None:
            psafile = f'{self.psaexe}/Examples/{pc}*unbound.tbl'
            if os.path.exists(self.psaexe):
                config_path = self.config_set()
                pdblist_path = self.pdblist_set(pdbfile)
                os.system(f'{self.psaexe}/psa {config_path} {pdblist_path}')
                unbound_files = glob.glob(psafile)
                if unbound_files:
                    self.psafile = unbound_files
                    return self.psafile
                else:
                    print(f"Error: Output file {psafile} not found in {self.psaexe}/Examples/")
            else:
                print("Error: PSAIA software is not installed. Please install!")

    def res_array(self, psafile=None):
        """
        Return PSAIA output as a numpy array, generating the file if necessary.
        """
        # Check if provided psafile exists; if not, generate it
        psafile = psafile if os.path.exists(psafile) else self.get_psafile(self.pc, self.pdbfile)
        psa_out = np.loadtxt(psafile, skiprows=16, usecols=(18, 24), dtype=float)
        return psa_out


class GE_feat(PDBfuc):
    def __init__(self, pdbfile=None, ghecomexe=None, ge_file=None):
        super().__init__(pdbfile)
        self.ge_file = ge_file
        self.ghecomexe = ghecomexe
        self.pdbfile = pdbfile
        # self.ghecom(ge_file, pdbfile)

        if ge_file:
            if not os.path.exists(ge_file):
                self.ghecom(ge_file, pdbfile)
        else:
            print("Warning: ge_file not provided, skipping ghecom execution.")

    def ghecom(self, ge_file, pdbfile):
        """
        excute ghecom program
        """
        if not pdbfile:
            print("Error: PDB file not provided.")
            return None

        if not self.ghecomexe or not os.path.exists(self.ghecomexe):
            print("Error: ghecom executable not found. Please provide a valid path.")
            return None

        os.system(f'{self.ghecomexe} -M M -atmhet B -hetpep2atm F -ipdb {pdbfile} -ores {ge_file}')
        self.ge_file = ge_file
        return ge_file

    def descriptor(self, ge_file):
        """
        return: str_array col.1=>res col.2=>shellAcc col.3=>Rinacc col.4=>pocketness
        """
        ghecom_res = ge_file if os.path.exists(ge_file) else self.ghecom(ge_file, self.pdbfile)
        out = np.loadtxt(ghecom_res, skiprows=43, usecols=(3, 4, 7), dtype=float)
        out[:, 0] = out[:, 0] / 100
        out[:, 1] = out[:, 1]
        out[:, 2] = out[:, 2] / 100
        return out

    def res_array(self):
        return self.descriptor(self.ge_file)


class CircularVariance(PDBfuc):
    """
    Reference: A new method for mapping macromolecular topography
    """

    def __init__(self, pdbfile, atom_distmap, r):
        """
        pdbfile: file
        r: int/float/list
        """
        super(CircularVariance, self).__init__(pdbfile)

        if isinstance(r, (int, float)):
            self.r = [r]
        elif isinstance(r, (list, tuple)):
            self.r = r
        else:
            raise ValueError('not accepted r')

        self.atom_distmap = atom_distmap
        self.atom_atom_vector = self._atom_atom_vector()

    def _atom_atom_vector(self):
        """
        return: dict {atom:{atom:vector},......}
        """
        atoms = self.atom_distmap.index.values
        atom_atom_vector = dict()
        for atom1 in atoms:
            atom_atom_vector[atom1] = \
                dict(map(lambda x: (x, np.array(self.coord['_'.join(atom1.split('_')[:-1])]
                                                [atom1.split('_')[-1]], dtype=float) -
                                    np.array(self.coord['_'.join(x.split('_')[:-1])]
                                             [x.split('_')[-1]], dtype=float)), atoms))
        return atom_atom_vector

    def _CVatom(self, r):
        """
        r: float
        return: dict {atom:cv,......}
        """

        def cv_calculate(atom, relative_atoms):
            vectors = np.array([self.atom_atom_vector[atom][x] for x in relative_atoms if x != atom], dtype=float)
            if len(vectors) == 0:
                return 0.0
            norms = np.linalg.norm(vectors, axis=1).reshape(-1, 1)
            norm_vectors = vectors / norms
            return 1 - np.linalg.norm(np.sum(norm_vectors, axis=0)) / len(vectors)

        atoms = self.atom_distmap.index.values
        cv_atomdict = {}

        # dup_counts = self.atom_distmap.index.value_counts()
        # dup_counts = dup_counts[dup_counts > 1]
        # print(f"rep:{dup_counts}")

        for atom in atoms:
            dists = self.atom_distmap.loc[atom].values
            relative_atoms = atoms[dists < r]
            cv_atomdict[atom] = cv_calculate(atom, relative_atoms)
        return cv_atomdict

    def _CVatom_array(self, r):
        """
        r: float
        return: ndarray (length of atoms*1)
        """
        cvatomdict = self._CVatom(r)
        return np.array(list(map(lambda x: cvatomdict['_'.join(x)], self.res_atom))).reshape(-1, 1)

    def _CVres(self, r):
        """
        r: float
        return: ndarray (length of res*1)
        """
        cvatomdict = self._CVatom(r)

        def cvcalculate(res):
            res_atoms = filter(lambda x: '_'.join(x.split('_')[:-1]) == res, cvatomdict)
            return np.mean([cvatomdict[x] for x in res_atoms])

        cv = np.array([cvcalculate(x) for _, _, x in self.res], dtype=float).reshape(-1, 1)
        return cv

    def CVatom(self):
        """
        return: ndarray (length of atoms*length of r)
        """
        return np.hstack(list(map(self._CVatom_array, self.r)))

    def CVres(self):
        """
        return: ndarray (length of res*length of r)
        """
        return np.hstack(list(map(self._CVres, self.r)))

    def atom_array(self):
        return self.CVatom()

    def res_array(self):
        return self.CVres()


class Graph(PDBfuc):
    def __init__(self, pdbfile, graph):
        super(Graph, self).__init__(pdbfile)
        """
        graph: edges (list/file)
        """
        if isinstance(graph, str):
            with open(graph, 'r') as f:
                edges = list(map(lambda x: x.strip().split(), f.readlines()))
        elif isinstance(graph, (list, np.ndarray)):
            edges = list(graph)
        else:
            raise ValueError('check edges in')

        self.g = nx.Graph()
        edges = list(map(lambda x: [self.res_index[y] for y in x], edges))
        self.g.add_edges_from(edges)


class RicciCurvature(Graph):
    def __init__(self, pdbfile, graph):
        """
        graph: edges (list/file)
        """
        super(RicciCurvature, self).__init__(pdbfile, graph)

        self.ORC = []
        self.FRC = []

    def ollivier_ricci(self, mode='sum'):
        """
        mode: str (sum/mean)
        return: dict
        """
        orc = OllivierRicci(self.g, alpha=0.5, verbose="INFO")
        orc.compute_ricci_curvature()
        g = orc.G
        aggregate = np.sum if mode == 'sum' else np.mean
        nodes = list(g.nodes)
        for n in self.index_res.keys():
            if n in nodes:
                curvature = list(map(lambda x: g[n][x]['ricciCurvature'], g[n]))
                curvature = aggregate(curvature)
                self.ORC.append(curvature)
            else:
                self.ORC.append(0)
        return np.array(self.ORC).reshape(-1, 1)

    def forman_ricci(self, mode='sum'):
        """
        mode: str (sum/mean)
        return: dict
        """
        frc = FormanRicci(self.g)
        frc.compute_ricci_curvature()
        g = frc.G
        aggregate = np.sum if mode == 'sum' else np.mean
        nodes = list(g.nodes)
        for n in self.index_res.keys():
            if n in nodes:
                curvature = list(map(lambda x: g[n][x]['formanCurvature'], g[n]))
                curvature = aggregate(curvature)
                self.FRC.append(curvature)
            else:
                self.FRC.append(0)
        return np.array(self.FRC).reshape(-1, 1)


class Topo(PDBfuc):
    def __init__(self, pdbfile, atom_contact):
        super(Topo, self).__init__(pdbfile)
        self.contact = atom_contact
        self.g = nx.Graph()

    def TPres(self):
        self.g.add_nodes_from(self.res_index.keys())
        self.g.add_edges_from(self.contact)
        degrees = np.array(self.g.degree())[:, -1].astype(int)
        close = np.array(list(nx.closeness_centrality(self.g).values()))
        between = np.array(list(nx.betweenness_centrality(self.g).values()))
        clusters = np.array(list(nx.algorithms.clustering(self.g).values()))
        return np.vstack((degrees, clusters, close, between)).T
        # return np.vstack((degrees, clusters, stats.zscore(close), stats.zscore(between))).T

    def res_array(self):
        return self.TPres()


class Laplacian(PDBfuc):
    def __init__(self, pdbfile, res_distmap):
        super(Laplacian, self).__init__(pdbfile)
        self.distance_map = res_distmap

    def _sigmas(self):
        distmap_lower = np.tril(self.distance_map, k=-1)
        flattened_distance = distmap_lower[distmap_lower > 1e-6]
        sigmas = np.percentile(flattened_distance, [0, 25, 50, 75, 100])
        sigmas = np.maximum(sigmas, 0.1)
        return sigmas

    def _omega(self, sigma, eps=1e-6):
        dist = self.distance_map
        w = np.exp(-dist ** 2 / sigma ** 2)
        i, j = np.indices(w.shape)
        w[abs(i - j) <= 1] = 0
        sum_omega = np.sum(w, axis=1)
        sum_omega = np.where(sum_omega == 0, eps, sum_omega)
        return w, sum_omega

    def cal_laps(self, sigma):
        omega, sum_omega = self._omega(sigma)
        coords = np.array(list(self.get_res_coord_dict().values()))
        weighted_coords = np.dot(omega, coords)
        averaged_coords = weighted_coords / sum_omega.reshape(-1, 1)
        LAPS = np.linalg.norm(coords - averaged_coords, axis=1)
        return LAPS

    def res_array(self):
        return np.array(list(map(lambda x: self.cal_laps(x), self._sigmas()))).T


class MultifractalDim(Graph):
    def __init__(self, pdbfile, graph):
        """
        graph: edges (list/file)
        """
        super(MultifractalDim, self).__init__(pdbfile, graph)

        self.MFD = list()

    def slope(self, weight=None):
        for index_res in self.index_res.keys():
            if index_res in self.g.nodes:
                self.MFD.append(self._slope(index_res, weight=weight))
            else:
                self.MFD.append(0)
        return np.array(self.MFD).reshape(-1, 1)

    def _slope(self, node, weight=None):
        m = nx.single_source_shortest_path_length if weight is None else nx.single_source_dijkstra_path_length
        spl = m(self.g, node)
        grow = [y for x, y in spl.items() if x != node]
        grow.sort()
        l_ml = [[x, y] for x, y in Counter(grow).items()]
        if len(l_ml) < 2:
            slope = 0
        else:
            l = np.log([x for x, y in l_ml])
            ml = np.log(np.cumsum([y for x, y in l_ml]))
            slope, intercept, r_value, p_value, std_err = stats.linregress(l, ml)
        return slope


def zscore_feat_per_protein(feat: np.ndarray) -> np.ndarray:
    """对每个蛋白 (N, D) 特征做图内 z-score"""
    scaler = StandardScaler()
    return scaler.fit_transform(feat)


def get_hand_str_feat(args, pdbfile, pdb_fuc, pc, data_type):
    max_asa = np.loadtxt(f'max_ASA.txt', dtype=str)
    res_max_asa = dict(zip(max_asa[:, 0], max_asa[:, 1].astype(int)))

    map_file = f'./dist_map/{data_type}/{pc}.pkl'
    dssp_file = f'./hand_str_feat/{data_type}/DSSP/{pc}.dssp'
    ge_file = f'./hand_str_feat/{data_type}/GE/{pc}-ghe.txt'

    for output_file in (map_file, dssp_file, ge_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    rasa_feat = RASA_feat(pdbfile, res_max_asa, args.dssp, dssp_file)
    rasa = rasa_feat.res_array()  # (N, 1)

    def compute_and_save_distmap():
        res_distmap, res_contact = pdb_fuc.contact(distmap_type='res', discut=7)
        atom_distmap, atom_contact = pdb_fuc.contact(discut=5)
        with open(map_file, 'wb') as fd:
            pickle.dump((res_distmap, res_contact, atom_distmap, atom_contact), fd)
        return res_distmap, res_contact, atom_distmap, atom_contact

    if os.path.exists(map_file):
        with open(map_file, 'rb') as fd:
            res_distmap, res_contact, atom_distmap, atom_contact = pickle.load(fd)
        current_residue_ids = set(pdb_fuc.res_index)
        cached_residue_ids = {
            residue_id
            for edge in res_contact
            for residue_id in edge
        }
        if (
            res_distmap.shape[0] != rasa.shape[0]
            or not cached_residue_ids.issubset(current_residue_ids)
        ):
            # print(f"[Warning] Shape mismatch for {pc}, regenerating dist map...")
            res_distmap, res_contact, atom_distmap, atom_contact = compute_and_save_distmap()
            # 删除旧的缓存文件
            if os.path.exists(ge_file):
                os.remove(ge_file)
            if os.path.exists(dssp_file):
                os.remove(dssp_file)
            rasa_feat = RASA_feat(pdbfile, res_max_asa, args.dssp, dssp_file)
            rasa = rasa_feat.res_array()  # (N, 1)
    else:
        res_distmap, res_contact, atom_distmap, atom_contact = compute_and_save_distmap()

    # dp = dp_feat(pc)
    # dp_feat = DP_feat(pdbfile, pc, args.psaia)
    # dp = dp_feat.res_array(f'/home/yyShen/Epitope_project/GraphBepi/feature/DP_pre/output/{pc}.tbl')
    # dp = dp_feat.res_array(f'./features/STR_feature/DP/{pc}.tbl')

    tp_feat = Topo(pdbfile, atom_contact)
    tp = tp_feat.res_array()  # (N, 4)

    mfd_feat = MultifractalDim(pdbfile, res_contact)
    mfd = mfd_feat.slope()  # (N, 1)

    rc_feat = RicciCurvature(pdbfile, res_contact)
    orc = rc_feat.ollivier_ricci()  # (N, 1)
    frc = rc_feat.forman_ricci()  # (N, 1)

    ghecom_feat = GE_feat(pdbfile, args.ghecom, ge_file)
    ge = ghecom_feat.res_array()  # (N, 3)

    cv_feat = CircularVariance(pdbfile, atom_distmap, [12, 100])
    cv = cv_feat.res_array()  # (N, 2)

    ln_feat = Laplacian(pdbfile, res_distmap)
    ln = ln_feat.res_array()  # (N, 5)

    local_feat = np.hstack((rasa, tp[:, :2], mfd, orc, frc, ge, cv[:, :1], ln[:, :-2]))  # (N, 12)
    local_feat_z = zscore_feat_per_protein(local_feat)

    global_feat = np.hstack((tp[:, 2:], cv[:, 1:], ln[:, -2:]))  # (N, 6)
    hand_str_feat = np.hstack((local_feat_z, global_feat))  # (N, 18)

    return hand_str_feat


class ESMIFFeatureExtractor:
    def __init__(self, model_path=None):
        super().__init__()
        if model_path is not None:
            self.model_path = model_path
        else:
            print("Please download ESM_IF1 model or path!")
        self.model, self.alphabet = self.load_model()

    def load_model(self):
        torch.hub.set_dir(self.model_path)
        # model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm_if1_gvp4_t16_142M_UR50")
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        model = model.eval()
        return model, alphabet

    def get_feat(self, pdbfile, chain_id):
        structure = esm.inverse_folding.util.load_structure(pdbfile, chain_id)
        coords, seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
        esmif_feat = esm.inverse_folding.util.get_encoder_output(self.model, self.alphabet, coords)
        return esmif_feat.detach().numpy()


def convert_3to1(resname):
    try:
        return protein_letters_3to1[resname.capitalize()]
    except KeyError:
        return 'X'  # 未知残基，替换为 X


def get_chain_seq(u, pc, chain, fasta_file, mut_info=None):
    """
    提取指定链的氨基酸序列并保存为 fasta
    命名规则与单链 pdb 一致:
        - 突变链: pdb_chain.mutation.fasta
        - 未突变链: pdb_chain.fasta
    """

    if chain is None:
        return ''

    # mut_chains = list({mut[1] for mut in mut_info.split('_')}) if mut_info else []

    chain_res = u.select_atoms(f"chainID {chain}").residues
    sequence = ''.join([convert_3to1(res.resname) for res in chain_res])

    # # 关键点：统一用 suffix
    # suffix = f".{mut_info}" if (mut_info and chain in mut_chains) else ""
    # fasta_file = os.path.join(out_dir, f"{pdb}_{chain}{suffix}.fasta")
    # fasta_header = f">{pdb}_{chain}{suffix}"

    if os.path.exists(fasta_file):
        with open(fasta_file, 'r') as fa:
            fasta_sequence = fa.readlines()[1].strip()
            if fasta_sequence != sequence:
                print(f"Warning: {fasta_file} mismatch, updating...")
                with open(fasta_file, 'w') as fa:
                    fa.write(f"{pc}\n{sequence}")
                return sequence
            else:
                return fasta_sequence
    else:
        os.makedirs(os.path.dirname(fasta_file), exist_ok=True)

        with open(fasta_file, 'w') as fa:
            fa.write(f"{pc}\n{sequence}")
        return sequence


class ESM2FeatureExtractor:
    def __init__(self, model_path=None):
        if model_path is not None:
            self.model_path = model_path
        else:
            print("Error: Please download ESM-2 model or change path!")
        self.model, self.alphabet, self.batch_converter = self.load_model()

    def load_model(self):
        torch.hub.set_dir(self.model_path)
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        return model, alphabet, batch_converter

    def get_feat(self, u, pc, chain, fasta_file, mode="rescoding"):
        # def save_seq_to_fasta(pc, sequence):
        #     with open(fasta_file, 'w') as fa:
        #         fa.write(f'>{pc}\n{sequence}')

        # fasta_path = f'../data/fasta/na_ag_fasta/{pc}.fasta'
        seq = get_chain_seq(u, pc, chain, fasta_file)
        data = [(pc, seq)]

        # if os.path.exists(fasta_file):
        #     with open(fasta_file, 'r') as fa:
        #         flist = fa.readlines()
        #     data = [(flist[0][1:].strip(), flist[1].strip())]
        # else:
        #     seq = get_chain_seq(u, pc, chain, fasta_file)
        #     save_seq_to_fasta(pc, seq)
        #     data = [(pc, seq)]

        _, _, batch_tokens = self.batch_converter(data)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33][0][1:-1, :].detach().numpy()  # 残基级特征

        if mode == "rescoding":
            return token_representations  # (L, D)

        elif mode == "seqcoding":
            cls_token = results["representations"][33][0][0, :].detach().numpy()  # CLS token
            return cls_token  # (D,)

        # else:
        #     # print(f'!check! File {fasta_file} not found.')
        #     seq = get_chain_seq(u, pdb, chain, fasta_file)
        #     return None

            # data = [(pc, ''.join(self.get_chain_seq()))]
            # save_seq_to_fasta(pc=data[0][0], sequence=data[0][1])


class AblangFeatureExtractor:
    def __init__(self, chain_type, model_path):
        if model_path is not None:
            self.model_path = model_path
        else:
            print("Error: Please download Ablang model or change path!")
        self.model = ablang.pretrained(chain_type, self.model_path)
        self.model.freeze()

        self.max_position = 128 if chain_type == 'heavy' else 127  #157

    def get_feat(self, fasta_file, mode="rescoding"):
        """ 提取特征，支持 'rescoding' (残基级) 和 'seqcoding' (序列级) """
        # def save_seq_to_fasta(pc, sequence):
        #     with open(fasta_file, 'w') as fa:
        #         fa.write(f'>{pc}\n{sequence}')
        if os.path.exists(fasta_file):
            with open(fasta_file, 'r') as f:
                flist = f.readlines()
            pc, sequence = flist[0][1:].strip(), flist[1].strip()
            seq_length = len(sequence)

            # seq_length < max_position，直接计算特征
            if seq_length > self.max_position:
                seq_inputs = [sequence[i:i + self.max_position] for i in range(0, seq_length, self.max_position)]
            else:
                seq_inputs = [sequence]
        else:
            print(f'!check! File {fasta_file} not found.')
            return None

        if mode == "seqcoding":
            # 计算所有片段的平均值
            seq_feat = np.mean(self.model(seq_inputs, mode=mode), axis=0)
            return seq_feat
        else:
            # 残基级（rescoding）直接拼接
            res_feat = np.concatenate(self.model(seq_inputs, mode=mode), axis=0)
            return res_feat


def get_seq_feats(fasta_H, fasta_L, fasta_Ag, esm2_extractor):
    # phy_che = np.loadtxt('phyche_property.txt', dtype=str)
    # res_phy_che = dict(zip(phy_che[1:, 0], phy_che[1:, 1:]))
    # phyche = phy_che_feat(pdbfile, res_phy_che)
    # phyche_feat = phyche.res_array()
    ablang_H = AblangFeatureExtractor(chain_type="heavy", model_path='/home/yyShen/Software/ablang/ablang-heavy/')
    ablang_L = AblangFeatureExtractor(chain_type="light", model_path='/home/yyShen/Software/ablang/ablang-light/')

    features = {}

    # 处理重链
    if os.path.exists(fasta_H):
        with open(fasta_H, 'r') as f:
            h_seq = f.readlines()[1].strip()
        h_res_feat = ablang_H.get_feat(h_seq, mode="rescoding")
        h_seq_feat = ablang_H.get_feat(h_seq, mode="seqcoding")  # 处理重链序列

    else:
        h_res_feat, h_seq_feat = None, None

    # 处理轻链
    if fasta_L and os.path.exists(fasta_L):
        with open(fasta_L, 'r') as f:
            l_seq = f.readlines()[1].strip()
        l_res_feat = ablang_L.get_feat(l_seq, mode="rescoding")
        l_seq_feat = ablang_L.get_feat(l_seq, mode="seqcoding")  # 处理轻链序列
    else:
        l_res_feat, l_seq_feat = None, None

    # 处理抗原
    features['Ag_rescoding'] = esm2_extractor.get_feat(fasta_Ag, mode="rescoding")
    features['Ag_seqcoding'] = esm2_extractor.get_feat(fasta_Ag, mode="seqcoding")

    # 组合抗体特征
    if h_res_feat is not None and l_res_feat is not None:
        features['Ab_rescoding'] = np.concatenate((h_res_feat, l_res_feat), axis=0)
    elif h_res_feat is not None:
        features['Ab_rescoding'] = h_res_feat
    elif l_res_feat is not None:
        features['Ab_rescoding'] = l_res_feat

    if h_seq_feat is not None and l_seq_feat is not None:
        features['Ab_seqcoding'] = np.concatenate((h_seq_feat, l_seq_feat), axis=0)
    elif h_seq_feat is not None:
        features['Ab_seqcoding'] = h_seq_feat
    elif l_seq_feat is not None:
        features['Ab_seqcoding'] = l_seq_feat

    return features


def get_str_feats(pdbfile, H_chain, L_chain, Ag_chain, esmif_extractor):
    esmif_H = esmif_extractor.get_feat(pdbfile, H_chain)
    esmif_Ag = esmif_extractor.get_feat(pdbfile, Ag_chain)

    features = {}
    if L_chain is not None:
        esmif_L = esmif_extractor.get_feat(pdbfile, L_chain)
        features['Ab_esmif'] = np.concatenate([esmif_H, esmif_L], axis=0)
    else:
        features['Ab_esmif'] = esmif_H

    features['Ag_esmif'] = esmif_Ag

    return features


def save_features(features, output_path):
    with open(output_path, 'wb') as fd:
        pickle.dump(features, fd)


def get_chain_filename(pdb, chain, mut_chain=None, mut_info=None):
    """根据是否为突变链生成结构文件名（用于避免覆盖）"""
    if chain == mut_chain and mut_info is not None:
        return f"{pdb}_{chain}.{mut_info}.pdb"
        # return f"{pdb}_{chain}.{mut_chain}.{mut_info}.pdb"
    else:
        return f"{pdb}_{chain}.pdb"


def has_nan_in_file(file_path):
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)  # 你 save_features 里面用的可能是 pickle
        # data 是 dict: { "chain1_hand_str": np.array(...), ... }
        for k, v in data.items():
            if isinstance(v, np.ndarray) and np.isnan(v).any():
                print(f"[Warning] NaN detected in {file_path} -> {k}")
                return True
        return False
    except Exception as e:
        print(f"[Error] Failed to check NaN in {file_path}: {e}")
        return True   # 如果文件坏了，也强制重跑


def normalize_global_feats(all_global_feats, file_cache, scaler_path, mode="fit"):
    """
    统一的全局标准化接口：
    - mode="fit":   在训练集上 fit scaler, 保存到 scaler_path，并更新文件
    - mode="transform": 在验证/测试集上用已有 scaler 更新文件
    """
    if all_global_feats and mode == "fit":
        print("Fitting global scaler ...")
        all_global_feats = np.vstack(all_global_feats)
        scaler = StandardScaler().fit(all_global_feats)
        joblib.dump(scaler, scaler_path)
    else:
        print("Loading saved scaler ...")
        scaler = joblib.load(scaler_path)

    # === 用 scaler 更新文件 ===
    print("Updating files with normalized global features ...")
    for fpath in file_cache:
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        for key, hand_feat in data.items():
            global_feat = hand_feat[:, -6:]
            global_feat_z = scaler.transform(global_feat)
            hand_feat_new = np.hstack((hand_feat[:, :-6], global_feat_z))
            data[key] = hand_feat_new

        new_fpath = fpath.replace("hand_str_feat", "hand_str_feat_norm")
        os.makedirs(os.path.dirname(new_fpath), exist_ok=True)
        with open(new_fpath, "wb") as f:
            pickle.dump(data, f)

    print(f"All files updated with global normalization ({mode})")


def hand_feat_scaler(data_dict, data_type=None):
    # pair_list = np.loadtxt(f'../data/split_data/cv/all_train.txt', dtype=str)
    # pair_list = np.loadtxt(f'../data/split_data/cv/test4_pair.txt', dtype=str)
    pair_list = np.atleast_1d(
        np.loadtxt(f'../data/PPB-Affinity/{data_type}_pair.txt', dtype=str)
    )
    all_global_feats = []  # 存整个数据集的 global feats
    file_cache = []  # 缓存每个pair的文件路径，等全局标准化后再写回

    for i, pair in enumerate(pair_list):
        # pair = '6ysq_AC_G'   '7bw4_A_BCD'
        print(i, pair)
        parts = pair.split('.', 1)
        complex_id = parts[0]
        mut_info = parts[1] if len(parts) > 1 else None
        wt_type = 'aai'
        # wt_type = data_dict[complex_id]['type']
        # dt = f'{wt_type}_mut' if mut_info else wt_type

        pdb, chain1, chain2 = complex_id.split('_')
        # chains_group = [list(chain1), list(chain2)]

        all_input_chains = list(chain1) + list(chain2)
        af3_chain_ids = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        chain_map = {
            old: new
            for old, new in zip(all_input_chains, af3_chain_ids)
        }
        chains_group = [
            [chain_map[c] for c in chain1],
            [chain_map[c] for c in chain2]
        ]

        # chains_group = [['A'], ['B']]

        hand_str_output_file = f'./hand_str_feat/{data_type}/{pair}.pkl'
        if (not os.path.exists(hand_str_output_file)
                or os.path.getsize(hand_str_output_file) <= 1024
                or has_nan_in_file(hand_str_output_file)):
            print(i, pair, f"{pair} hand_str_feat need generation")
            break
        else:
            with open(hand_str_output_file, "rb") as f:
                hand_str_feat = pickle.load(f)
            for j, chain_group in enumerate(chains_group, start=1):
                chain_prefix = f"chain{j}"  # chain1 / chain2 ...
                hand_feat = hand_str_feat[f"{chain_prefix}_hand_str"]
                all_global_feats.append(hand_feat[:, -6:])
            file_cache.append(hand_str_output_file)

    # === 全局标准化 ===
    # normalize_global_feats(all_global_feats, file_cache, f"tr_cv_global_scaler.pkl", mode="fit")
    normalize_global_feats(all_global_feats, file_cache, f"tr_cv_global_scaler.pkl", mode="transform")


def get_chain_dirs(pair_type, data_type):
    # if "aai" in pair_type:
    #     chain1_fasta_dir = f"../data/fasta/{data_type}_ab_fasta/"
    #     chain2_fasta_dir = f"../data/fasta/{data_type}_ag_fasta/"
    #     chain1_pdb_dir = f"../data/pdb_files/{data_type}_ab/"
    #     chain2_pdb_dir = f"../data/pdb_files/{data_type}_ag/"
    #
    # elif "tcr-pmhc" in pair_type:
    #     chain1_fasta_dir = f"../data/fasta/{data_type}_tcr_fasta/"
    #     chain2_fasta_dir = f"../data/fasta/{data_type}_pmhc_fasta/"
    #     chain1_pdb_dir = f"../data/pdb_files/{data_type}_tcr/"
    #     chain2_pdb_dir = f"../data/pdb_files/{data_type}_pmhc/"
    #
    # else:
    chain1_fasta_dir = chain2_fasta_dir = f"../data/fasta/{data_type}_pc_fasta/"
    chain1_pdb_dir = chain2_pdb_dir = f"../data/pdb_files/{data_type}_pc/"

    return (
        chain1_fasta_dir,
        chain2_fasta_dir,
        chain1_pdb_dir,
        chain2_pdb_dir,
    )


def main(data_dict, data_type):
    pair_list = np.atleast_1d(
        np.loadtxt(f'../data/PPB-Affinity/{data_type}_pair.txt', dtype=str)
    )

    # ablang_H = AblangFeatureExtractor(chain_type="heavy", model_path='/home/yyShen/Software/ablang/ablang-heavy/')
    # ablang_L = AblangFeatureExtractor(chain_type="light", model_path='/home/yyShen/Software/ablang/ablang-light/')
    esm2_extractor = ESM2FeatureExtractor(args.esm_path)
    esmif_extractor = ESMIFFeatureExtractor(args.esm_path)

    for i, pair in enumerate(pair_list):
        # pair = "6ysq_AC_G"
        parts = pair.split('.', 1)
        complex_id = parts[0]
        mut_info = parts[1] if len(parts) > 1 else None
        mut_chains = list({mut[1] for mut in mut_info.split('_')}) if mut_info else []
        # pair_type = data_dict[complex_id]["type"]
        pair_type = 'aai'

        (
            chain1_fasta_dir,
            chain2_fasta_dir,
            chain1_pdb_dir,
            chain2_pdb_dir,
        ) = get_chain_dirs(pair_type, data_type)

        pdb, chain1, chain2 = complex_id.split('_')

        all_input_chains = list(chain1) + list(chain2)
        af3_chain_ids = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        chain_map = {
            old: new
            for old, new in zip(all_input_chains, af3_chain_ids)
        }
        chains_group = [
            [chain_map[c] for c in chain1],
            [chain_map[c] for c in chain2]
        ]

        # chains_group = [['A'], ['B']]

        # chains_group = [list(chain1), list(chain2)]
        seq_features = {}
        str_features = {}
        hand_str_feat = {}

        seq_output_file = f'./seq_features/{data_type}/{pair}.pkl'
        str_output_file = f'./str_features/{data_type}/{pair}.pkl'
        hand_str_output_file = f'./hand_str_feat/{data_type}/{pair}.pkl'
        dist_map_file = f'./dist_map/{data_type}/{pair}.pkl'

        for path in [seq_output_file, str_output_file, hand_str_output_file, dist_map_file]:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        # flag = True
        # with open(seq_output_file, "rb") as f:
        #     seq_feat = pickle.load(f)
        # print(i, pair)
        # for j, chain_group in enumerate(chains_group, start=1):
        #     chain_prefix = f"chain{j}"  # chain1 / chain2
        #     key = f"{chain_prefix}_seqcoding"
        #
        #     if key not in seq_feat:
        #         flag = True
        #         break
        #
        #     seq_f = seq_feat[key]
        #     seq_f = torch.from_numpy(seq_f)
        #     if seq_f.dim() == 1:
        #         L = seq_f.numel()
        #         if L % 1280 == 0:
        #             continue
        #         else:
        #             flag = True
        #             break

        print(i, pair, "new_generation")
        for j, chain_group in enumerate(chains_group, start=1):
            chain_prefix = f"chain{j}"  # chain1 / chain2
            # seq_f = seq_feat[f"{chain_prefix}_seqcoding"]
            # seq_f = torch.from_numpy(seq_f)
            # if seq_f.dim() == 1:
            #     L = seq_f.numel()
            #     if L % 1280 == 0:
            #         continue
            seq_res_feats, seq_seq_feats = [], []
            str_feats, hand_feats = [], []

            # === 区分抗体/抗原组 ===
            if j == 1:
                pdb_dir = chain1_pdb_dir
                fasta_dir = chain1_fasta_dir
            else:
                pdb_dir = chain2_pdb_dir
                fasta_dir = chain2_fasta_dir

            for k, chain in enumerate(chain_group):
                # --- 突变链命名 ---
                suffix = f".{mut_info}" if (mut_info and chain in mut_chains) else ""
                pc_name = f"{pdb}_{chain}{suffix}"

                fasta_file = os.path.join(fasta_dir, f"{pc_name}.fasta")
                pdb_file = os.path.join(pdb_dir, f"{pc_name}.pdb")

                if os.path.exists(pdb_file):
                    u = mda.Universe(pdb_file)
                    # --- 序列特征 ---
                    # if (not os.path.exists(seq_output_file)
                    #         or os.path.getsize(seq_output_file) <= 1024
                    #         or has_nan_in_file(seq_output_file)):
                    if True:
                        # print("生成序列特征")
                        seq_res_feats.append(esm2_extractor.get_feat(u, pc_name, chain, fasta_file, mode="rescoding"))
                        seq_seq_feats.append(esm2_extractor.get_feat(u, pc_name, chain, fasta_file, mode="seqcoding"))

                            # if chain_prefix == "Ab":  # 抗体组：第一条为重链，其余为轻链
                            #     if k == 0:
                            #         res_feat = ablang_H.get_feat(fasta_file, mode="rescoding")
                            #         seq_feat = ablang_H.get_feat(fasta_file, mode="seqcoding")
                            #     else:  # 其余为轻链
                            #         res_feat = ablang_L.get_feat(fasta_file, mode="rescoding")
                            #         seq_feat = ablang_L.get_feat(fasta_file, mode="seqcoding")
                            #
                            # elif chain_prefix == "Ag":
                            #     res_feat = esm2_extractor.get_feat(fasta_file, mode="rescoding")
                            #     seq_feat = esm2_extractor.get_feat(fasta_file, mode="seqcoding")
                            #
                            # else:
                            #     raise ValueError(f"未知组别: {chain_prefix}")
                            #
                            # seq_res_feats.append(res_feat)
                            # seq_seq_feats.append(seq_feat)

                    # --- 结构特征 ---
                    # if (not os.path.exists(str_output_file)
                    #         or os.path.getsize(str_output_file) <= 1024
                    #         or has_nan_in_file(str_output_file)):
                    if True:
                        str_feats.append(esmif_extractor.get_feat(pdb_file, chain))

                        pdb_fuc = PDBfuc(pdb_file)
                        hand_feat = get_hand_str_feat(args, pdb_file, pdb_fuc, pc_name, data_type)
                        hand_feats.append(hand_feat)

                else:
                    print(i, pair)
                    continue

            # === 拼接同组多链特征 ===
            if seq_res_feats:
                arr = np.concatenate(seq_res_feats, axis=0)
                if arr.shape[1] != 1280:
                    raise ValueError(
                        f"[{chain_prefix}] seq_res_feats 维度错误: {arr.shape}, 期望 (n,1280)"
                    )
                seq_features[f"{chain_prefix}_rescoding"] = arr
            if seq_seq_feats:
                arr = np.concatenate(seq_seq_feats, axis=0)
                seq_features[f"{chain_prefix}_seqcoding"] = arr
            if str_feats:
                arr = np.concatenate(str_feats, axis=0)
                if arr.shape[1] != 512:
                    raise ValueError(
                        f"[{chain_prefix}] str_feats 维度错误: {arr.shape}, 期望 (n,512)"
                    )
                str_features[f"{chain_prefix}_esmif"] = arr
            if hand_feats:
                arr = np.concatenate(hand_feats, axis=0)
                if arr.shape[1] != 18:
                    raise ValueError(
                        f"[{chain_prefix}] hand_feats 维度错误: {arr.shape}, 期望 (n,18)"
                    )
                hand_str_feat[f"{chain_prefix}_hand_str"] = arr

        save_features(seq_features, seq_output_file)
        save_features(str_features, str_output_file)
        save_features(hand_str_feat, hand_str_output_file)


def rename_hand_str_keys(data_type):
    dir_path = f'./hand_str_feat_norm/{data_type}/'

    for fname in os.listdir(dir_path):
        if not fname.endswith(".pkl"):
            continue

        file_path = os.path.join(dir_path, fname)
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[跳过] 读取失败: {fname}, 错误: {e}")
            continue

        if not isinstance(data, dict):
            print(f"[跳过] 非字典类型: {fname}")
            continue

        new_data = {}
        modified = False

        for k, v in data.items():
            new_k = k
            if k.startswith("Ab_"):
                new_k = re.sub(r"^Ab_", "chain1_", k)
                modified = True
            elif k.startswith("Ag_"):
                new_k = re.sub(r"^Ag_", "chain2_", k)
                modified = True
            new_data[new_k] = v

        if modified:
            with open(file_path, "wb") as f:
                pickle.dump(new_data, f)
            print(f"[更新完成] {fname} ✅")
        else:
            print(f"[未修改] {fname}")


if __name__ == "__main__":
    raise SystemExit("Use feature/run_seq_str_features.py as the public entrypoint.")

    # import os
    # import json
    # import shutil
    # import numpy as np
    #
    # # --------------------------------------------------
    # # Config
    # # --------------------------------------------------
    # PAIR_FILE = "/home/yyShen/NAcontact/data/PPB-Affinity/CG_failed_42.txt"
    # SRC_ROOT = "/home/yyShen/NAcontact/feature/seq_features"
    # DST_ROOT = "/home/yyShen/NAcontact"
    #
    # # --------------------------------------------------
    # # Load pair list
    # # --------------------------------------------------
    # pairs = np.loadtxt(PAIR_FILE, dtype=str)
    # pairs = sorted(set(pairs))  # 去重，保证稳定
    #
    # # --------------------------------------------------
    # # Statistics
    # # --------------------------------------------------
    # copied = 0
    # missing = 0
    # skipped = 0
    #
    # # --------------------------------------------------
    # # Main loop
    # # --------------------------------------------------
    # for pair in pairs:
    #     pair_wt = pair.split('.')[0]
    #
    #     if pair_wt not in data_dict:
    #         print(f"[Skip] {pair_wt} not found in data_dict")
    #         skipped += 1
    #         continue
    #
    #     data_type = data_dict[pair_wt]["type"]
    #
    #     src_file = os.path.join(SRC_ROOT, data_type, f"{pair}.pkl")
    #     dst_dir = os.path.join(DST_ROOT, data_type)
    #     dst_file = os.path.join(dst_dir, f"{pair}.pkl")
    #
    #     if not os.path.exists(src_file):
    #         print(f"[Missing] {src_file}")
    #         missing += 1
    #         continue
    #
    #     os.makedirs(dst_dir, exist_ok=True)
    #
    #     if os.path.exists(dst_file):
    #         # 已存在则跳过，避免覆盖
    #         skipped += 1
    #         continue
    #
    #     shutil.copy2(src_file, dst_file)
    #     copied += 1
    #
    # # --------------------------------------------------
    # # Summary
    # # --------------------------------------------------
    # print("\n========== Copy Summary ==========")
    # print(f"Copied : {copied}")
    # print(f"Missing: {missing}")
    # print(f"Skipped: {skipped}")
    # print("=================================")

    # import os
    # import shutil
    #
    # #######################################################
    # # 用户配置
    # #######################################################
    #
    # pair_file = "/home/yyShen/NAcontact/data/split_data/cv_splits/test4_pair.txt"
    # output_dir = "/home/yyShen/NAcontact/compared_method/prodigy/input/test4/"
    #
    # search_dirs = [
    #     "/home/yyShen/NAcontact/data/pdb_files/ppi/",
    #     "/home/yyShen/NAcontact/data/pdb_files/aai/",
    #     "/home/yyShen/NAcontact/data/pdb_files/tcr-pmhc/",
    #     "/home/yyShen/NAcontact/data/pdb_files/ppi_mut/",
    #     "/home/yyShen/NAcontact/data/pdb_files/aai_mut/",
    #     "/home/yyShen/NAcontact/data/pdb_files/tcr-pmhc_mut/"
    # ]
    #
    # #######################################################
    # # Step 1: 读取 pair 列表
    # #######################################################
    #
    # pairs = np.loadtxt(pair_file, dtype=str)
    # pairs = sorted(set(pairs))
    #
    # print(f"pair 数量：{len(pairs)}")
    #
    # #######################################################
    # # Step 2: 搜索并复制 pair.pdb 文件
    # #######################################################
    #
    # os.makedirs(output_dir, exist_ok=True)
    #
    # found = []
    # not_found = []
    #
    # for pair in pairs:
    #     pdb_name = f"{pair}.pdb"
    #     copied = False
    #
    #     # 遍历三个目录
    #     for d in search_dirs:
    #         full_path = os.path.join(d, pdb_name)
    #         if os.path.isfile(full_path):
    #             shutil.copy(full_path, output_dir)
    #             found.append(pair)
    #             copied = True
    #             break
    #
    #     if not copied:
    #         not_found.append(pair)
    #
    # print("\n====== 搜索结果 ======")
    # print(f"成功复制：{len(found)}")
    # print(f"未找到：{len(not_found)}")

    # pair_list_file = "../data/PPB-Affinity/ppi_pair_2328.txt"
    # str_dir = f"/home/yyShen/NAcontact/feature/hand_str_feat/{data_type}/"
    #
    # # --- 1. 加载合法 pair 列表 ---
    # pair_list = np.loadtxt(pair_list_file, dtype=str)
    # pair_set = set(pair_list)
    #
    # # --- 2. 遍历目录，删除不在 pair_list 中的 .pkl 文件 ---
    # remove_count = 0
    # keep_count = 0
    #
    # for fname in os.listdir(str_dir):
    #     if not fname.endswith(".pkl"):
    #         continue
    #
    #     pair = fname.replace(".pkl", "")
    #     file_path = os.path.join(str_dir, fname)
    #
    #     if pair not in pair_set:
    #         os.remove(file_path)
    #         remove_count += 1
    #         print(f"删除: {fname}")
    #     else:
    #         keep_count += 1
    #
    # print("\n=== 完成 ===")
    # print(f"保留 {keep_count} 个 pair 文件")
    # print(f"删除 {remove_count} 个多余文件")

    # import os
    # import re
    #
    # # 路径设置
    # pkl_dir = "/home/yyShen/NAcontact/feature/dist_map/ppi_mut/"
    # ref_file = "/home/yyShen/NAcontact/data/PPB-Affinity/ppi_pdb.txt"
    #
    # # 读取参考列表（去除空行、重复）
    # with open(ref_file) as f:
    #     valid_pdbs = {line.strip().lower() for line in f if line.strip()}
    # print(f"📘 参考 PDB 数量: {len(valid_pdbs)}")
    #
    # # 遍历目录下的 .pkl 文件
    # deleted = []
    # for fname in os.listdir(pkl_dir):
    #     if not fname.endswith(".pkl"):
    #         continue
    #
    #     # 从文件名提取 pdb id（如 1a22_A.pkl → 1a22）
    #     match = re.match(r"([0-9a-zA-Z]{4})_", fname)
    #     if not match:
    #         continue
    #     pdb_id = match.group(1).lower()
    #
    #     # 检查是否在参考列表中
    #     if pdb_id not in valid_pdbs:
    #         full_path = os.path.join(pkl_dir, fname)
    #         os.remove(full_path)
    #         deleted.append(fname)
    #
    # # 输出结果
    # if deleted:
    #     print(f"🗑 删除了 {len(deleted)} 个无效文件：")
    #     for f in deleted:
    #         print(f"   - {f}")
    # else:
    #     print("✅ 所有 .pkl 文件均匹配参考列表，无需删除。")

    # import os
    # import json
    #
    # # === 配置路径 ===
    # base_dir = "/home/yyShen/NAcontact/feature/energy_features/"
    # mapping_file = "/absolute/path/to/pdb2clean_mapping.json"
    #
    # # 进入目标目录
    # os.chdir(base_dir)
    #
    # # === 读取 JSON 映射 ===
    # with open(mapping_file, "r") as f:
    #     rename_dict = json.load(f)
    #
    # # === 遍历目录并重命名 ===
    # for d in os.listdir("."):
    #     if d in rename_dict:
    #         new_name = rename_dict[d]
    #         print(f"🔄 Renaming {d} → {new_name}")
    #         os.rename(d, new_name)
    #     else:
    #         print(f"⚠️ 未在映射表中找到：{d}")
    #
    # print("✅ 重命名完成！")



# ls -l /home/yyShen/NAcontact/data/pdb_files/affinity_mut/ | grep "^-" | wc -l
# ls /home/yyShen/NAcontact/data/pdb_files/ppi_sig_mut/ | sed 's/\..*$//' > file_name.txt
# find /home/yyShen/NAcontact/data/pdb_files/foldx_multi_mut_file/ -maxdepth 1 -type d -not -name '.' | wc -l
