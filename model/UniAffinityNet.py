from torchdrug.data import constant
from torchdrug import layers
from model.STRNet import *
from model.SEQNet import *
import math


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, num_layers=2, dropout=0.0):
        super().__init__()

        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))

            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UniAffinityNet(nn.Module):
    num_class = constant.NUM_AMINO_ACID  # 20

    def __init__(self, seq_emb_dim=1280, seq_hd_dim=512, hd_dim=256,
                 res_node_dim=1813, res_edge_dim=20, res_hd_dims=[512],
                 cg_node_dim=35, cg_edge_dim=53, cg_hd_dims=[256] * 2,
                 cg_gh_builder=None, use_type=False, num_types=3, t_dim=16,
                 use_cg=False, use_res=False, use_str=True, use_seq=True, dropout=0.1,
                 task_mode='reg', ensemble_mode="repr_residual"):
        super().__init__()

        self.use_res = use_res
        self.use_cg = use_cg
        self.use_str = use_str
        self.use_seq = use_seq
        self.use_type = use_type

        self.task_mode = task_mode
        self.ensemble_mode = ensemble_mode  # ["moe", "repr", "fusion"]
        self.type2id = {"ppi": 0, "aai": 1, "tcr-pmhc": 2}
        gate_in_dim = 0

        if use_seq:
            self.seq_net = SeqEncoder(seq_emb_dim, seq_hd_dim)
            self.seq_proj = nn.Sequential(
                nn.Linear(seq_hd_dim, seq_hd_dim),
                nn.LayerNorm(seq_hd_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            gate_in_dim += seq_hd_dim

        if use_str:
            self.str_net = STRNet(res_node_dim, res_edge_dim, res_hd_dims,
                                  cg_node_dim, cg_edge_dim, cg_hd_dims,
                                  cg_gh_builder=cg_gh_builder, task_mode=task_mode)
            self.str_proj = nn.Sequential(
                nn.Linear(self.str_net.out_dim, seq_hd_dim),
                nn.LayerNorm(seq_hd_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            gate_in_dim += seq_hd_dim

        if use_type:
            # self.task_base_gate = nn.ParameterDict({
            #     "ppi": nn.Parameter(torch.tensor(0.10)),
            #     "aai": nn.Parameter(torch.tensor(0.90)),
            #     "tcr-pmhc": nn.Parameter(torch.tensor(0.90)),
            # })
            self.task_base_gate = nn.ParameterDict({
                "ppi": nn.Parameter(torch.tensor(0.05)),
                "aai": nn.Parameter(torch.tensor(0.90)),
                "tcr-pmhc": nn.Parameter(torch.tensor(0.95)),
            })

        if self.ensemble_mode in ["repr_residual"]:

            # self.seq_error_head = nn.Sequential(
            #     nn.Linear(seq_hd_dim + 1, hd_dim),
            #     nn.ReLU(),
            #     nn.Dropout(dropout),
            #     nn.Linear(hd_dim, 1)
            # )
            #
            # self.str_error_head = nn.Sequential(
            #     nn.Linear(seq_hd_dim + 1, hd_dim),
            #     nn.ReLU(),
            #     nn.Dropout(dropout),
            #     nn.Linear(hd_dim, 1)
            # )
            #
            # # -------------------------------------------------
            # # residual correction head
            # # -------------------------------------------------
            # self.delta_head = nn.Sequential(
            #     nn.Linear(seq_hd_dim + 2, hd_dim),
            #     nn.LayerNorm(hd_dim),
            #     nn.ReLU(),
            #     nn.Dropout(dropout),
            #     nn.Linear(hd_dim, 1)
            # )
            #
            # nn.init.zeros_(self.delta_head[-1].weight)
            # nn.init.zeros_(self.delta_head[-1].bias)
            self.routing_input_ln = nn.LayerNorm(seq_hd_dim * 4 + 2)

            self.routing_mlp = nn.Sequential(
                nn.Linear(seq_hd_dim * 4 + 2, hd_dim),
                nn.ReLU(),
                nn.Linear(hd_dim, 1)
            )

            def init_fn(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    nn.init.zeros_(m.bias)

            self.routing_mlp.apply(init_fn)

        # if self.ensemble_mode in ["moe"]:
        #     err_in_dim = 2 + 2 + 4 + t_dim  # = 8 + t_dim
        #
        #     self.err_seq = MLP(err_in_dim, 1, hd_dim)
        #     self.err_str = MLP(err_in_dim, 1, hd_dim)
        #
        #     # input = [y_avg, |diff|, rep_diff, t_emb]
        #     shared_in_dim = 1 + 1 + 1 + t_dim
        #     self.shared_mlp = MLP(shared_in_dim, 1, hd_dim)
        #
        #     # input = [y_seq, y_str, rep_diff, t_emb]
        #     alpha_in_dim = 2 + 1 + t_dim
        #     self.alpha_mlp = MLP(alpha_in_dim, 1, hd_dim)
        #
        #     self.temp = nn.Parameter(torch.tensor(1.0))
        #
        # if self.ensemble_mode in ["repr"]:
        #     self.node2seq_proj = nn.Linear(res_node_dim, seq_hd_dim)
        #     self.pool_gate = nn.Linear(seq_hd_dim, 1)
        #
        #     self.gate_mlp = nn.Sequential(
        #         nn.Linear(3 * seq_hd_dim + t_dim, hd_dim),
        #         nn.LayerNorm(hd_dim),
        #         nn.ReLU(),
        #         nn.Linear(hd_dim, seq_hd_dim)
        #     )
        #
        #     nn.init.zeros_(self.gate_mlp[-1].bias)
        #
        #     self.reg_head = nn.Sequential(
        #         nn.Linear(seq_hd_dim + t_dim + 1, hd_dim),
        #         nn.ReLU(),
        #         nn.Linear(hd_dim, 1)
        #     )
        #
        # if self.ensemble_mode == "fusion":
        #     shared_in_dim = 7 + t_dim
        #     self.corr_mlp = MLP(shared_in_dim, 1, hd_dim)
        #     self.type_gate = MLP(t_dim, 1, hidden_dim=32, num_layers=2)
        #
        #     nn.init.zeros_(self.corr_mlp.net[-1].weight)
        #     nn.init.zeros_(self.corr_mlp.net[-1].bias)
        #
        #     nn.init.zeros_(self.type_gate.net[-1].weight)
        #     nn.init.zeros_(self.type_gate.net[-1].bias)

    def forward(self, seq_input,
                res_graph=None,
                cg_graph=None,
                t_emb=None
                ):

        device = seq_input.device
        gate_inputs, expert_outputs, names = [], [], []
        # y_seq, h_seq = None, None
        # y_str, h_str = None, None
        data_type = res_graph.data_type

        y_seq, h_seq = self.seq_net(seq_input)  # [1,D]
        h_seq = self.seq_proj(h_seq)  # [1, D]
        gate_inputs.append(h_seq)
        expert_outputs.append(y_seq)
        names.append("seq")

        str_out = self.str_net(res_graph, cg_graph)
        y_str = str_out["pKd"]
        h_str = self.str_proj(str_out["graph_emb"])  # [1, D]
        gate_inputs.append(h_str)
        expert_outputs.append(y_str)
        names.append("str")
        base_gate = 0.5

        if self.use_type and data_type is not None:
            base_gate = 0.5

            # base_gate = self.task_base_gate[data_type]
            # type_id = torch.tensor([self.type2id[data_type]], device=device)
            # task_bias = self.task_bias[type_id]

        y_moe, weight_dict = None, None
        if self.ensemble_mode in ["repr_residual"]:
            signed_diff = (y_str - y_seq).view(-1, 1)
            disagreement = torch.abs(signed_diff)

            routing_input = torch.cat([
                h_seq,
                h_str,
                h_str - h_seq,
                torch.abs(h_str - h_seq),
                signed_diff,
                disagreement
            ], dim=-1)

            routing_input = self.routing_input_ln(routing_input)
            alpha = torch.sigmoid(self.routing_mlp(routing_input)).squeeze(-1)

            # small disagreement  -> conservative fusion
            # large disagreement  -> sharper routing
            tau = 0.5
            k = 3.0
            sharpness = torch.sigmoid(k * (disagreement.squeeze(-1) - tau))

            # low disagreement : ±0.15
            # high disagreement: ±0.50
            gate_scale = 0.15 + 0.35 * sharpness
            seq_gate = 0.5 + gate_scale * (alpha - 0.5)
            seq_gate = seq_gate.clamp(min=0.01, max=0.99)

            y_fuse = seq_gate * y_seq + (1 - seq_gate) * y_str

            return {
                "pKd": y_fuse,
                "y_seq": y_seq,
                "y_str": y_str,
                "seq_gate": seq_gate.squeeze(-1),
                "gate_scale": gate_scale,
                "disagreement": disagreement.squeeze(-1),
                "alpha": alpha,
                "pair_attn": str_out["pair_attn"],
            }

            # joint_repr = torch.cat([h_seq, h_str], dim=-1)
            # diff_vec = h_str - h_seq
            # diff_norm = torch.norm(diff_vec, dim=-1, keepdim=True)
            # signed_diff = (y_str - y_seq).view(-1, 1)
            # disagreement = torch.abs(signed_diff)
            #
            # seq_in = torch.cat([h_seq, signed_diff], dim=-1)
            # str_in = torch.cat([h_str, signed_diff], dim=-1)
            #
            # pred_seq_err = F.softplus(self.seq_error_head(seq_in)).squeeze(-1)
            # pred_str_err = F.softplus(self.str_error_head(str_in)).squeeze(-1)

            # tau = 0.9
            # k = 5.0
            # routing_strength = torch.sigmoid(k * (disagreement.squeeze(-1) - tau))
            # routing_strength = routing_strength ** 2
            # routing_strength = routing_strength.detach() * 0.8 + routing_strength * 0.2
            #
            # temperature = 1.0
            # err_diff = (pred_str_err - pred_seq_err) / temperature
            # adaptive_gate = torch.sigmoid(err_diff)
            # adaptive_gate = 0.5 + 0.15 * (adaptive_gate - 0.5)
            #
            # seq_gate = base_gate + routing_strength * (adaptive_gate - base_gate)
            # seq_gate = seq_gate.clamp(min=0.01, max=0.99)

            # tau = 0.5
            # k = 5.0
            # routing_strength = torch.sigmoid(k * (disagreement.squeeze(-1) - tau)) ** 2
            # routing_strength = routing_strength.detach() * 0.8 + routing_strength * 0.2
            #
            # err_seq_score = -pred_seq_err
            # err_str_score = -pred_str_err
            #
            # winner_gate = torch.softmax(
            #     torch.stack([err_seq_score, err_str_score], dim=-1),
            #     dim=-1)
            # adaptive_gate = winner_gate[..., 0]
            #
            # seq_gate = (1 - routing_strength) * base_gate + routing_strength * adaptive_gate
            # seq_gate = seq_gate.clamp(min=0.01, max=0.99)
            #
            # y_fuse = seq_gate * y_seq + (1 - seq_gate) * y_str
            #
            # delta_in = torch.cat([
            #     diff_vec,
            #     diff_norm,
            #     signed_diff,
            #
            # ], dim=-1)
            #
            # delta_raw = self.delta_head(delta_in).squeeze(-1)
            # delta_y = 0.05 * torch.tanh(delta_raw) * routing_strength.detach()
            #
            # pKd_pred = y_fuse + delta_y

            # return {
            #     "pKd": pKd_pred.squeeze(-1),
            #     "y_str": y_str.squeeze(-1),
            #     "y_seq": y_seq.squeeze(-1),
            #     "seq_gate": seq_gate.squeeze(-1),
                # "adapt_gate": adaptive_gate.squeeze(-1),
                # "pred_seq_err": pred_seq_err,
                # "pred_str_err": pred_str_err,
                # "delta_y": delta_y,
                # "corr_gate": corr_gate,
            # }

        if self.ensemble_mode in ["moe"]:
            y_avg = 0.5 * (y_seq + y_str)
            diff_y = y_seq - y_str

            rep_diff = torch.norm(h_seq - h_str, dim=-1, keepdim=True)
            seq_norm = torch.norm(h_seq, dim=-1, keepdim=True)
            str_norm = torch.norm(h_str, dim=-1, keepdim=True)
            cos_sim = F.cosine_similarity(h_seq, h_str, dim=-1).unsqueeze(-1)

            # =========================================================
            # relative prediction（关键新增）
            # =========================================================
            rel_seq = y_seq - y_avg
            rel_str = y_str - y_avg

            # =========================================================
            # error predictor input
            # =========================================================
            err_in = torch.cat([
                y_seq, y_str,
                rel_seq, rel_str,
                rep_diff,
                seq_norm, str_norm,
                cos_sim,
                t_emb
            ], dim=-1)

            # =========================================================
            # 1️⃣ 预测误差（核心）
            # =========================================================
            e_seq = F.softplus(self.err_seq(err_in))  # 保证 >0
            e_str = F.softplus(self.err_str(err_in))

            # =========================================================
            # 2️⃣ soft oracle 权重
            # =========================================================
            w_seq = torch.exp(-e_seq / self.temp)
            w_str = torch.exp(-e_str / self.temp)

            w_sum = w_seq + w_str + 1e-6
            w_seq = w_seq / w_sum
            w_str = w_str / w_sum

            y_unc = w_seq * y_seq + w_str * y_str

            # =========================================================
            # confidence（解释用）
            # =========================================================
            # confidence = torch.abs(w_seq - w_str)  # fusion1
            # confidence = torch.exp(-torch.abs(y_unc - y_avg))  # fusion
            overall_unc = w_seq * e_seq + w_str * e_str
            moe_conf = torch.exp(-overall_unc)  # fusion2

            # =========================================================
            # 3️⃣ shared correction（只在不确定时启用）
            # =========================================================
            shared_in = torch.cat([
                y_avg,
                torch.abs(diff_y),
                rep_diff,
                t_emb
            ], dim=-1)

            gamma = torch.tanh(self.shared_mlp(shared_in))

            # =========================================================
            # final（关键：confidence gating）
            # =========================================================
            y_moe = y_unc + (1.0 - moe_conf) * gamma

        y_repr, h_fuse = None, None
        if self.ensemble_mode in ["repr"]:

            h_node = res_graph.x.to(device)  # [N, res_node_dim]
            h_node_proj = self.node2seq_proj(h_node)  # [N, seq_hd_dim]
            score = self.pool_gate(h_node_proj)  # [N, 1]
            attn = torch.softmax(score, dim=0)
            g_graph = (attn * h_node_proj).sum(dim=0, keepdim=True)  # [1, seq_hd_dim]

            h_base = 0.5 * (h_seq + h_str)
            diff_vec = h_str - h_seq
            diff = torch.norm(diff_vec, dim=-1, keepdim=True)   # [1,1]
            diff = diff / (diff.mean().detach() + 1e-6)

            gate_in = torch.cat([
                h_base,
                diff_vec,
                g_graph,
                t_emb
            ], dim=-1)  # [1, 3D+t_dim]

            # alpha = torch.sigmoid(self.gate_mlp(gate_in))  # [1, 1]
            # h_fuse = h_base + alpha * (h_other - h_base)

            alpha = 0.5 * torch.tanh(self.gate_mlp(gate_in))  # [1, D]
            h_fuse = h_base + diff * alpha  # [1, D]

            h_ht = torch.cat([h_fuse, t_emb, diff], dim=-1)

            y_repr = self.reg_head(h_ht)

        if self.ensemble_mode in ["fusion"]:
            y_mean = 0.5 * (y_seq + y_str)
            diff_y = y_seq - y_str

            rep_diff = torch.norm(h_seq - h_str, dim=-1, keepdim=True)
            seq_norm = torch.norm(h_seq, dim=-1, keepdim=True)
            str_norm = torch.norm(h_str, dim=-1, keepdim=True)
            cos_sim = F.cosine_similarity(h_seq, h_str, dim=-1).unsqueeze(-1)

            shared_in = torch.cat([
                y_mean,
                diff_y,
                torch.abs(diff_y),
                rep_diff,
                seq_norm,
                str_norm,
                cos_sim,
                t_emb
            ], dim=-1)

            delta_raw = torch.tanh(self.corr_mlp(shared_in))

            # =========================================================
            # 2️⃣ type-aware correction gate
            # =========================================================
            type_gate = torch.sigmoid(self.type_gate(t_emb))
            diff_gate = torch.sigmoid(2.0 * torch.abs(diff_y))
            delta = (
                    0.15
                    * type_gate
                    * diff_gate
                    * delta_raw
                    * diff_y
            )

            y_pred = y_mean + delta

            return {
                "pKd": y_pred,
                "y_str": y_str,
                "y_seq": y_seq,
                "y_mean": y_mean,
                "delta": delta,
                "delta_raw": delta_raw,
                "type_gate": type_gate,
                "diff_gate": diff_gate,
            }

        if self.ensemble_mode == "moe":

            return {
                "pKd": y_moe,
                "y_seq": y_seq,
                "y_str": y_str,
                "w_seq": w_seq,
                "w_str": w_str,
                "e_seq": e_seq,
                "e_str": e_str,
                "moe_conf": moe_conf
            }

        elif self.ensemble_mode == "repr":

            return {
                "pKd": y_repr,
                "y_str": y_str,
                "y_seq": y_seq,
                "alpha": alpha,
                "h_fuse": h_fuse,
                "h_seq": h_seq,
                "h_str": h_str
            }
        #
        # elif self.ensemble_mode == "fusion":
        #
        #     return {
        #         "pKd": y_pred,
        #         "y_str": y_str,
        #         "y_seq": y_seq,
        #         "gate": gate,
        #         "logits": logits,
        #         "h_fuse": h_fuse,
        #         "h_seq": h_seq,
        #         "h_str": h_str,
        #     }

    def attention(self, q, k, v):
        attn_logits = (q @ k.transpose(-1, -2)) * self.attn_scale
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)
        return attn @ v

    def _process_intra_feat(self, seq_feat):
        if seq_feat.dim() == 1:
            L = seq_feat.numel()
            assert L % 1280 == 0, f"Invalid seq_feat length: {L}"

            n_chain = L // 1280
            seq_feat = seq_feat.view(n_chain, 1280)  # [n_chain, 1280]

        seq_emb = self.seq_proj(seq_feat)  # [n_chain, hidden]
        seq_repr = seq_emb.mean(dim=0)  # [hidden]
        return seq_repr

    def _encode_intra_graph(self, graph, bead_emb=None):
        res_emb, bead_emb = self._process_intra_graph(graph)
        struct_repr = pool_node_emb(bead_emb)  # [hidden]
        seq_feat = graph.seq_feat.to(self.device)
        seq_repr = self._process_intra_feat(seq_feat)
        return struct_repr + seq_repr

    def _process_inter_graph(self, graph, device):
        graph = graph.to(device)

        pc1_x, pc2_x, node_type = graph.pc1_node_attr, graph.pc2_node_attr, graph.node_type
        edge_index, edge_attr = graph.edge_index, graph.edge_attr
        pc1_mask = (node_type == 0)
        pc2_mask = (node_type == 1)

        # === inter节点特征投影 ===
        inter_pc1_emb = self.pc1_node_proj(pc1_x)  # (N1,hidden_dims)
        inter_pc2_emb = self.pc2_node_proj(pc2_x)
        inter_res_emb = torch.zeros((node_type.size(0), inter_pc1_emb.size(-1)), device=device)
        inter_res_emb[pc1_mask] = inter_pc1_emb
        inter_res_emb[pc2_mask] = inter_pc2_emb

        # === inter交互嵌入 ===
        pair_emb = self.pair_encoder(inter_res_emb, edge_index)  # [N_pair, conv_out_dim]

        # === 图注意力传播 ===
        pair_attn_current = pair_emb
        for i, layer in enumerate(self.graph_attn):
            pair_emb_new, inter_res_emb = layer(
                inter_res_emb, edge_index, edge_attr, pair_attn_current, node_type=node_type
            )
            pair_attn_current = pair_emb_new

        return inter_res_emb, pair_attn_current

        # pair_repr, graph_repr, res_repr = self._get_weighted_repr(pair_attn_current, inter_res_emb)
        # return graph_repr, pair_repr, res_repr

    def _get_weighted_repr(self, pair_emb, res_emb, edge_weight=None, node_weight=None):
        # ==========================
        # Edge-level aggregation
        # ==========================
        pair_scores = self.pair_scorer(pair_emb)  # [E, 1]

        if edge_weight is not None:
            pair_scores = pair_scores * edge_weight

        weighted_pair = pair_emb * pair_scores  # [E, D]
        # pair_attn = torch.softmax(pair_scores, dim=0)

        pair_sum = weighted_pair.sum(dim=0)  # ∑ g_ij E_ij
        pair_max = weighted_pair.max(dim=0).values  # strongest contact

        pair_repr = torch.cat([pair_sum, pair_max], dim=-1)
        pair_repr = self.ln_pair(pair_repr)

        # ==========================
        # Node-level aggregation
        # ==========================
        if node_weight is not None:
            w = node_weight / (node_weight.sum() + 1e-6)
            node_mean = (res_emb * w).sum(0)
        else:
            node_mean = res_emb.mean(0)

        node_max = res_emb.max(0).values
        graph_repr = torch.cat([node_mean, node_max], dim=-1)
        graph_repr = self.ln_graph(graph_repr)

        res_repr = self.ln_res(res_emb)

        return pair_repr, graph_repr, res_repr

    def _encode_chain(self, graph):
        res_emb, bead_emb = self._process_intra_graph(graph)
        struct_repr = pool_node_emb(bead_emb)  # [hidden]
        seq_feat = graph.seq_feat.to(self.device)
        if seq_feat.dim() == 1:
            L = seq_feat.numel()
            assert L % 1280 == 0, f"Invalid seq_feat length: {L}"

            n_chain = L // 1280
            seq_feat = seq_feat.view(n_chain, 1280)  # [n_chain, 1280]

        seq_emb = self.seq_proj(seq_feat)  # [n_chain, hidden]
        seq_repr = seq_emb.mean(dim=0)  # [hidden]

        return struct_repr + seq_repr

    def _process_intra_graph(self, graph):
        def bead_to_res(bead_emb, bead2residue, method='max'):
            """
            bead_emb: [N_bead, D]
            bead2residue: [N_bead]，表示每个 bead 属于哪个残基（编号从0开始）
            return: [N_residue, D]
            """
            num_residues = int(bead2residue.max().item()) + 1
            if method == 'mean':
                return scatter_mean(bead_emb, bead2residue, dim=0, dim_size=num_residues)
            elif method == 'max':
                return scatter_max(bead_emb, bead2residue, dim=0, dim_size=num_residues)[0]
            elif method == 'meanmax':
                mean = scatter_mean(bead_emb, bead2residue, dim=0, dim_size=num_residues)
                maxv = scatter_max(bead_emb, bead2residue, dim=0, dim_size=num_residues)[0]
                return torch.cat([mean, maxv], dim=-1)
            else:
                raise ValueError("Unsupported pooling method")

        # def bead_edge_to_res_edge(graph, edge_input):
        #     device = edge_input.device
        #     bead2res = graph.bead2residue
        #     bead_edge_index = graph.edge_list[:, :2].T  # [2, E_bead]
        #
        #     # bead → residue
        #     bead_src = bead_edge_index[0]
        #     bead_dst = bead_edge_index[1]
        #     res_src = bead2res[bead_src]  # [E_bead]
        #     res_dst = bead2res[bead_dst]  # [E_bead]
        #
        #     res_edge_pairs = torch.stack([res_src, res_dst], dim=1)
        #     unique_pairs, inv = torch.unique(res_edge_pairs, dim=0, return_inverse=True)
        #     res_edge_feat = scatter_mean(edge_input, inv.to(device), dim=0)  # [E_res, D_edge]
        #     res_edge_index = unique_pairs.t()  # [2, E_res]
        #     return res_edge_index, res_edge_feat

        if self.graph_construction_model:
            # currently no graph_construction_model.apply_node_layer function is used here (None)
            graph = self.graph_construction_model.apply_node_layer(graph)
            # AdvSpatialEdge in cg_transform script
            graph = self.graph_construction_model.apply_edge_layer(graph)

        hiddens = []
        x = graph.atom_feature.float().to(self.device)  # (num_node, input_dim=25)

        if self.num_angle_bin:
            line_graph = self.spatial_line_graph(graph)
            edge_input = line_graph.node_feature.float().to(self.device)  # (num_edge, edge_input_dim=51)

        for i in range(len(self.layers)):
            h = self.layers[i](graph, x)  # (num_node, hidden_dim=256)
            if self.short_cut and h.shape == x.shape:
                h = h + x
            if self.num_angle_bin:
                edge_hidden = self.edge_layers[i](line_graph, edge_input)  # (num_edge, hidden_dim=256)
                edge_weight = graph.edge_weight.unsqueeze(-1).to(self.device)
                node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
                update = scatter_add(edge_hidden * edge_weight, node_out.to(self.device), dim=0,
                                     dim_size=graph.num_node * self.num_relation)  # (dim_size, input_dim=25))
                update = update.view(graph.num_node, self.num_relation * edge_hidden.shape[1])
                update = self.layers[i].linear(update)  # (num_node, hidden_dim=256)
                update = self.layers[i].activation(update)
                h = h + update
                edge_input = edge_hidden

            if self.batch_norm:
                h = self.batch_norms[i](h)
            hiddens.append(h)
            x = h

        if self.concat_hidden:
            bead_emb = torch.cat(hiddens, dim=-1)
        else:
            bead_emb = hiddens[-1]

        res_emb = bead_to_res(bead_emb, graph.bead2residue.to(self.device), 'meanmax')
        # res_edge_index, res_edge_emb = bead_edge_to_res_edge(graph, edge_input)
        return res_emb, bead_emb  # , res_edge_index, res_edge_emb

    def set_requires_grad(self, keys, requires_grad: bool):
        """
        keys: set or list of parameter names
        requires_grad: True or False
        """
        for name, param in self.named_parameters():
            if name in keys:
                param.requires_grad = requires_grad
