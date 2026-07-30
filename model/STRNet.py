from torchdrug.data import constant
from torchdrug import layers
from model.RESNet import *
from model.CGNet import *
from model.ATOMNET import *
from model.SEQNet import *
# from torch.utils.checkpoint import checkpoint
import math


class CGGuidedAttentionPooling(nn.Module):

    def __init__(self, dim, dropout=0.1):
        super().__init__()

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.bias_scale = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cg_ctx, inter_pair, raw_score=None):
        """
        cg_ctx:(d,)
        inter_pair:(N_pair, d)
        """
        inter_pair = self.norm(inter_pair)

        Q = self.q_proj(cg_ctx).unsqueeze(0)      # (1,d)
        K = self.k_proj(inter_pair)               # (N,d)
        V = self.v_proj(inter_pair)               # (N,d)

        attn_bias = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(Q.size(-1))  # (1,N)
        if raw_score is not None:
            raw_score = raw_score.squeeze(-1).unsqueeze(0)  # (1,N)
            score = raw_score + self.bias_scale * attn_bias
        else:
            score = attn_bias

        attn = torch.softmax(score, dim=-1)
        attn = self.dropout(attn)

        pooled = torch.matmul(attn, V)     # (d,)

        return pooled.squeeze(0), attn.squeeze(0)


class CGGuidedPooling(nn.Module):

    """
    CG-guided interaction aggregation

    CG only controls:
        1. interaction entropy (temperature)
        2. mean/max aggregation balance

    raw_score remains the dominant interaction ranking signal.
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()

        # -----------------------------------------
        # temperature controller
        # -----------------------------------------
        self.temp_mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )

        self.cg_res_proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.LayerNorm(dim * 2)
        )

        # -----------------------------------------
        # mean/max balance controller
        # -----------------------------------------
        # self.pool_gate = nn.Sequential(
        #     nn.Linear(dim, dim // 2),
        #     nn.GELU(),
        #     nn.Linear(dim // 2, 1)
        # )

        # self.mean_gate = nn.Sequential(
        #     nn.Linear(dim, dim // 2),
        #     nn.GELU(),
        #     nn.Linear(dim // 2, 1)
        # )
        #
        # self.max_gate = nn.Sequential(
        #     nn.Linear(dim, dim // 2),
        #     nn.GELU(),
        #     nn.Linear(dim // 2, 1)
        # )
        #
        # self.std_gate = nn.Sequential(
        #     nn.Linear(dim, dim // 2),
        #     nn.GELU(),
        #     nn.Linear(dim // 2, 1)
        # )

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cg_ctx, inter_pair, raw_score):
        """
        cg_ctx:     (d,)
        inter_pair: (N,d)
        raw_score:  (N,1)

        return:
            pair_repr: (2d,)
            attn:      (N,)
        """

        inter_pair = self.norm(inter_pair)
        raw_score = raw_score.squeeze(-1)          # (N,)

        # -----------------------------------------
        # CG-conditioned entropy control
        # -----------------------------------------
        delta_tem = torch.tanh(self.temp_mlp(cg_ctx)).squeeze(-1)
        temperature = 1.0 + 0.2 * delta_tem
        score = raw_score / temperature

        attn = torch.softmax(score, dim=0)         # (N,)
        attn = self.dropout(attn)
        weighted_pair = (inter_pair * attn.unsqueeze(-1)).sum(dim=0)   # (d,)
        pair_repr = self.cg_res_proj(weighted_pair)   # (2d,)

        return pair_repr, attn
        # # weighted mean interaction
        # mean_pair = (inter_pair * attn.unsqueeze(-1)).sum(dim=0)   # (d,)
        # # strongest interaction
        # max_pair = weighted_pair.max(dim=0).values    # (d,)
        # weighted_max = (inter_pair * attn.unsqueeze(-1)).max(dim=0).values  # (d,)
        # # interaction diversity
        # std_pair = inter_pair.std(dim=0)  # (d,)
        # pair_repr = torch.cat([weighted_pair, weighted_max], dim=-1)

        # -----------------------------------------
        # CG-guided aggregation style
        # -----------------------------------------
        # alpha = torch.sigmoid(self.pool_gate(cg_ctx)).squeeze(-1)
        # pair_repr = torch.cat([alpha * weighted_pair, (1.0 - alpha) * weighted_max], dim=-1)  # (2d,)

        # pair_repr = torch.cat([alpha * mean_pair, (1.0 - alpha) * max_pair], dim=-1)  # (2d,)

        # mean_gate = torch.sigmoid(self.mean_gate(cg_ctx)).squeeze(-1)
        # max_gate = torch.sigmoid(self.max_gate(cg_ctx)).squeeze(-1)
        # std_gate = torch.sigmoid(self.std_gate(cg_ctx)).squeeze(-1)
        #
        # mean_pair = mean_gate * mean_pair + std_gate * std_pair
        # max_pair = max_gate * max_pair
        #
        # pair_repr = torch.cat([mean_pair, max_pair], dim=-1)



class STRNet(nn.Module):
    num_class = constant.NUM_AMINO_ACID  # 20

    def __init__(self, res_node_dim=1813, res_edge_dim=20, res_hd_dims=[512],
                 cg_node_dim=35, cg_edge_dim=51, cg_hd_dims=[256] * 2, num_relation=5, num_angle_bin=6,
                 cg_gh_builder=None, use_cg=True, use_res=True, use_str=False, use_seq=False,
                 n_energy_dim=2, num_edge_types=3, num_layers=1, task_mode='reg', ensemble_mode="repr"):
        super().__init__()

        self.use_res = use_res
        self.use_cg = use_cg
        self.use_str = use_str
        self.use_seq = use_seq
        self.task_mode = task_mode
        self.ensemble_mode = ensemble_mode  # ["output", "repr", "conditional"]
        self.dropout = 0.1

        self.hidden_dim = res_hd_dims[-1]
        self.cg_dim = self.hidden_dim * 2
        self.out_dim = self.hidden_dim * 3

        if use_cg:
            self.cg_net = CGIntraNet(cg_node_dim, cg_edge_dim, cg_hd_dims,
                                     num_relation, num_angle_bin,
                                     cg_gh_builder=cg_gh_builder, task_mode=task_mode)

            self.cg_proj = nn.Sequential(
                nn.Linear(self.cg_dim, self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim)
            )

            self.res_proj = nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_dim * 3),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim * 3)
            )

            self.cg_pair_pool = CGGuidedPooling(
                dim=self.hidden_dim,
                dropout=self.dropout
            )

            self.delta_scale = nn.Parameter(torch.tensor(0.01))

            self.gate_mlp = nn.Sequential(
                nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim * 2, 1)
            )

            self.corr_mlp = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 1)
            )

        if self.task_mode in ["reg", "both"]:
            self.reg_head = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim // 2),
                nn.ReLU(),
                nn.Linear(self.out_dim // 2, self.out_dim // 4),
                nn.ReLU(),
                nn.Linear(self.out_dim // 4, 1)
            )

        if self.task_mode in ["cla", "both"]:
            self.cla_head = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim // 2),
                nn.ReLU(),
                nn.Linear(self.out_dim // 2, self.out_dim // 4),
                nn.ReLU(),
                nn.Linear(self.out_dim // 4, 3)
            )

        if use_res:
            self.res_net = ResInterNet(res_node_dim, res_edge_dim, res_hd_dims, task_mode=task_mode)

    def forward(self, res_graph=None, cg_graph=None, cla_pred=None, pKd_pred=None):
        device = next(self.parameters()).device
        y_res, y_cg = None, None

        res_graph = res_graph.to(device)
        data_type = res_graph.data_type
        # if self.use_res:
        with torch.no_grad():
            res_out = self.res_net(res_graph)

        res_gh_repr = res_out["graph_repr"]   # 3d
        pair_repr = res_out["pair_repr"]  # 2d
        node_repr = res_out["node_repr"]   # d
        y_res = res_out["pKd"].squeeze(-1)

        pair_emb = res_out["pair_emb"]
        res_pair_weight = res_out["pair_weight"]  # (E,1)
        inter_mask = (res_graph.edge_type == 2)
        inter_pair = pair_emb[inter_mask]
        raw_score = res_pair_weight[inter_mask]
        pair_attn = raw_score
        fused_repr = res_gh_repr
        refined_pair_repr = pair_repr
        cg_gate = torch.tensor(0.0, device=device)

        if cg_graph is not None and self.use_cg:
            with torch.no_grad():
                cg_out = self.cg_net(cg_graph)
            cg_ctx = self.cg_proj(cg_out["graph_repr"])  # d
            y_cg = cg_out["pKd"].squeeze(-1)

            gate_input = torch.cat([res_gh_repr, cg_ctx], dim=-1)
            gate_logit = self.gate_mlp(gate_input).squeeze(-1)
            # if data_type is not None:
            #     gate_logit = gate_logit + self.type_gate_bias[data_type]
            cg_gate = torch.sigmoid(gate_logit)
            y_base = cg_gate * y_cg + (1 - cg_gate) * y_res

            # -------------------------------------------------
            # CG-guided attention pooling
            # ------------------------------------------------
            cg_pair_repr, pair_attn = self.cg_pair_pool(cg_ctx, inter_pair, raw_score)
            delta_scale = torch.clamp(self.delta_scale, min=0.0, max=0.5)
            refined_pair_repr = pair_repr + delta_scale * cg_pair_repr

            corr_gate_input = torch.cat([cg_ctx, node_repr], dim=-1)
            corr_gate = 0.15 * torch.sigmoid(self.corr_mlp(corr_gate_input)).squeeze(-1)

            fused_repr = torch.cat([refined_pair_repr, node_repr], dim=-1)
            fused_repr = self.res_proj(fused_repr)

            if self.task_mode in ["reg", "both"]:
                delta_y = self.reg_head(fused_repr).squeeze(-1)
                pKd_pred = y_base + corr_gate * delta_y
            if self.task_mode in ["cla", "both"]:
                cla_pred = self.cla_head(fused_repr).unsqueeze(0)
        else:
            pKd_pred = y_res
            # corr_gate = torch.tensor(0.0, device=device)

        return {
            "pKd": pKd_pred,  # scalar prediction (optional use)
            "cla": cla_pred,  # classification logits
            "graph_emb": fused_repr.unsqueeze(0),
            "cg_gate": cg_gate,
            "pair_attn": pair_attn,
            "y_res": y_res,
            "y_cg": y_cg,
        }

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

    # def _process_inter_graph(self, res_graph, cg_graph=None, cg_out=None, device=None):
    #     res_graph = res_graph.to(device)
    #
    #     node_feat = res_graph.x  # [N, node_in_dim]
    #     edge_attr = res_graph.edge_attr
    #     edge_index = res_graph.edge_index
    #     # node_type = res_graph.node_type
    #
    #     res_global = res_graph.global_index  # [N_res]
    #
    #     res_emb = self.node_proj(node_feat)  # [N, hidden_dim]
    #
    #     if cg_graph is not None:
    #         cg_res_emb = cg_out["res_emb"]  # [N_cg_res, hidden_dim]
    #         cg_res_emb_proj = self.cg_fusion_proj(cg_res_emb)
    #
    #         bead2residue = cg_graph.bead2residue
    #         bead2global = torch.as_tensor(cg_graph.bead2global)  # [N_bead]
    #         num_cg_res = cg_res_emb.size(0)
    #
    #         max_gid = max(
    #             int(res_global.max()),
    #             int(bead2global.max())
    #         ) + 1
    #
    #         global2res = torch.full(
    #             (max_gid,),
    #             -1,
    #             dtype=torch.long,
    #             device=device
    #         )
    #         global2res[res_global] = torch.arange(len(res_global), device=device)
    #
    #         # 每个 residue 对应一个 global residue
    #         residue2global = scatter_mean(
    #             bead2global.float(),
    #             bead2residue,
    #             dim=0,
    #             dim_size=num_cg_res
    #         ).long()  # [N_cg_res]
    #
    #         # ===== 5. 对齐到 res_graph =====
    #         cg_res2res = global2res[residue2global]  # [N_cg_res]
    #
    #         mask = cg_res2res >= 0
    #
    #         cg2res_emb = torch.zeros(
    #             res_emb.size(0),
    #             cg_res_emb.size(-1),
    #             device=device
    #         )
    #
    #         valid_index = cg_res2res[mask]
    #         assert valid_index.max() < res_emb.size(0)
    #         cg2res_emb[valid_index] = cg_res_emb_proj[mask]
    #
    #         has_cg = (cg2res_emb.abs().sum(dim=-1) > 0).float().unsqueeze(-1)
    #         fusion_input = torch.cat([res_emb, cg2res_emb, has_cg], dim=-1)
    #         gate = torch.sigmoid(self.cg_gate(fusion_input))
    #
    #         res_emb = (1 - gate) * res_emb + gate * cg2res_emb
    #
    #     else:
    #         res_emb = res_emb
    #
    #     pair_emb = self.pair_init(
    #         node_emb=res_emb,
    #         edge_index=edge_index,
    #         edge_attr=edge_attr,
    #     )  # [E, hidden_dim]
    #
    #     for i, conv in enumerate(self.pair_convs):
    #         pair_emb = conv(
    #             pair_emb=pair_emb,
    #             node_emb=res_emb,
    #             edge_attr=edge_attr,
    #             edge_index=edge_index,
    #             graph=res_graph
    #         )
    #
    #     return res_emb, pair_emb

    def _process_inter_graph(self, res_graph, res_out=None, cg_graph=None, cg_out=None, device=None):

        # node_feat = res_graph.x
        edge_attr = res_graph.edge_attr
        edge_index = res_graph.edge_index
        res_global = res_graph.global_index
        src, dst = edge_index

        # res_emb = res_out["res_emb"]
        # node_feat = node_feat + 0.3 * res_emb
        # res_emb = self.node_proj(node_feat)

        # res_emb = self.node_proj(node_feat)
        node_emb = res_out["res_emb"]
        pair_emb = res_out["pair_emb"]
        # fusion = torch.cat([node_base, res_emb], dim=-1)
        # gate = torch.sigmoid(self.gate_mlp(fusion))
        # delta = self.delta_proj(res_emb)
        # res_emb = node_base + gate * delta

        # fusion = torch.cat([node_feat, res_emb], dim=-1)
        # gate = torch.sigmoid(self.node_proj(fusion))
        # res_emb = gate * res_emb + (1 - gate) * node_feat

        # pair_emb = self.pair_init(
        #     node_emb=res_emb,
        #     edge_index=edge_index,
        #     edge_attr=edge_attr,
        # )

        # =========================================================
        # 🔴 Stage 2: CG fusion
        # =========================================================
        if cg_graph is not None:
            cg_global = cg_out["graph_emb"] .to(device)
            cg_context = cg_global.unsqueeze(0).expand(
                pair_emb.size(0), -1
            )
            cg_scale = self.cg_scale_proj(cg_context)
            # cg_shift = self.cg_shift_proj(cg_context)
            cg_scale = 0.1 * torch.tanh(cg_scale)

            # =================================================
            # Feature-wise Linear Modulation
            # =================================================
            pair_emb = (
                    pair_emb * (1.0 + cg_scale)
                    # + cg_shift
            )

            pair_emb = self.pair_norm(pair_emb)

        pair_emb = F.relu(pair_emb)

        pair_emb = F.dropout(
            pair_emb,
            p=0.1,
            training=self.training
        )

        return node_emb, pair_emb

    def _get_weighted_repr(self, pair_emb, node_emb, graph, INTER_EDGE_ID=2):
        edge_index = graph.edge_index
        node_prior = graph.node_type.float()
        edge_type = graph.edge_type
        src, dst = edge_index

        inter_mask = (edge_type == INTER_EDGE_ID)

        edge_weight = self.prior_attn(
            pair_emb,
            edge_index,
            node_prior
        )  # [E,1]

        inter_pair = pair_emb[inter_mask]  # [E_inter, D]
        inter_score = edge_weight[inter_mask]  # [E_inter, 1]
        pair_attn = torch.softmax(inter_score, dim=0)  # [E_inter,1]

        attn_pair = (inter_pair * pair_attn).sum(dim=0)
        max_pair = inter_pair.max(dim=0).values
        pair_repr = torch.cat([attn_pair, max_pair], dim=-1)

        # node_weight = torch.sigmoid(node_imp)  # [N,1]
        node_weight = (
                              scatter_mean(edge_weight, src, dim=0, dim_size=node_emb.size(0)) +
                              scatter_mean(edge_weight, dst, dim=0, dim_size=node_emb.size(0))
                      ) / 2
        node_weight = node_weight / (node_weight.sum() + 1e-6)
        graph_repr = (node_emb * node_weight).sum(dim=0)

        return pair_repr, graph_repr, edge_weight, node_weight

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
