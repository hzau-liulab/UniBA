import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_softmax, scatter_add, scatter_mean, scatter_sum


class NodeUpdate(nn.Module):
    def __init__(self, node_dim, pair_dim, n_energy_dim=2, dropout=0.1):
        super().__init__()

        self.n_energy_dim = n_energy_dim

        # ===== energy -> attention =====
        self.energy_proj = nn.Linear(n_energy_dim, 1, bias=False)

        # ===== pair aggregation projection =====
        self.agg_proj = nn.Sequential(
            nn.LayerNorm(pair_dim * 2),
            nn.Linear(pair_dim * 2, node_dim),
            nn.ReLU(),
        )

        # ===== node role encoder（融合 prior）=====
        self.node_role_mlp = nn.Sequential(
            nn.LayerNorm(node_dim + 3),  # node_prior dim = 3
            nn.Linear(node_dim + 3, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )

        # ===== fusion =====
        self.update_mlp = nn.Sequential(
            nn.LayerNorm(node_dim * 2),
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim)
        )

        # ===== gating（控制更新强度）=====
        self.gate = nn.Sequential(
            nn.LayerNorm(node_dim + 3),
            nn.Linear(node_dim + 3, 1),
            nn.Sigmoid()
        )

        # ===== residual scale =====
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, node_emb, pair_emb, edge_attr, edge_index, graph):
        """
        node_emb: [N, node_dim]
        pair_emb: [E, pair_dim]
        edge_attr: [E, edge_dim]
        edge_index: [2, E]
        graph.node_type: [N,3]
        """

        src, dst = edge_index
        N = node_emb.size(0)

        # ======================
        # 1. energy-aware weighting
        # ======================
        energy_feat = edge_attr[:, :self.n_energy_dim]  # [E, n_energy]
        energy_score = self.energy_proj(energy_feat)  # [E,1]

        weight_src = scatter_softmax(energy_score, src, dim=0)
        weight_dst = scatter_softmax(energy_score, dst, dim=0)

        # ======================
        # 2. pair → node aggregation（保留方向）
        # ======================
        agg_src = scatter_sum(pair_emb * weight_src, src, dim=0, dim_size=N)
        agg_dst = scatter_sum(pair_emb * weight_dst, dst, dim=0, dim_size=N)

        pair_info = torch.cat([agg_src, agg_dst], dim=-1)  # [N, 2*pair_dim]
        pair_info = self.agg_proj(pair_info)  # [N, node_dim]

        # ======================
        # 3. node role encoding
        # ======================
        node_prior = graph.node_type.float()  # [N,3]

        node_role = self.node_role_mlp(
            torch.cat([node_emb, node_prior], dim=-1)
        )  # [N, node_dim]

        # ======================
        # 4. fusion（node + interaction）
        # ======================
        fused = torch.cat([node_role, pair_info], dim=-1)
        delta = self.update_mlp(fused)  # [N, node_dim]

        # ======================
        # 5. gating
        # ======================
        conf = self.gate(
            torch.cat([node_emb, node_prior], dim=-1)
        )  # [N,1]
        node_emb_out = node_emb + self.res_scale * conf * delta

        return node_emb_out


class PriorAwareAttention(nn.Module):
    def __init__(self, pair_dim, prior_dim, hidden_dim=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = pair_dim // 2

        self.score_mlp = nn.Sequential(
            nn.LayerNorm(pair_dim + prior_dim),
            nn.Linear(pair_dim + prior_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.score_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, pair_emb, edge_index, node_prior):
        src, dst = edge_index

        # ===== pair prior =====
        pair_prior = torch.cat([node_prior[src], node_prior[dst]], dim=-1)

        # ===== attention input =====
        attn_input = torch.cat([pair_emb, pair_prior], dim=-1)

        score = self.score_mlp(attn_input)  # [E,1]
        # score = score.masked_fill((edge_type != 2).unsqueeze(-1), float('-inf'))
        #
        # weight = torch.softmax(score, dim=0)  # [E,1]

        return score


class NodeImportance(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.ReLU(),
            nn.Linear(node_dim // 2, 1)
        )

    def forward(self, node_emb):
        return torch.sigmoid(self.mlp(node_emb))  # [N, 1]


def build_mlp(input_dim, hidden_dims=None, output_dim=None, act=nn.ReLU, dropout=0.0):
    if hidden_dims is None:
        hidden_dims = [128]
    layers = []
    dims = [input_dim] + list(hidden_dims)
    n_hidden = len(hidden_dims)

    for i in range(n_hidden):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        # 当没有输出层时，最后一个隐藏层后不加激活
        if i < n_hidden - 1 or output_dim is not None:
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

    if output_dim is not None:
        layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


# class GlobalPairAttention(nn.Module):
#     """
#     对每个 pair 计算全局重要性分数，用于全局 pooling attention。
#     """
#
#     def __init__(self, input_dim, hidden_dim=128):
#         super().__init__()
#         self.query_proj = nn.Linear(input_dim, hidden_dim)
#         self.gate_proj = nn.Linear(input_dim, hidden_dim)
#         self.score_proj = nn.Linear(hidden_dim, 1)
#
#     def forward(self, x):
#         """
#         x: [E, D]，边特征
#         return: [E, 1] 全局 pair 分数
#         """
#         query = torch.tanh(self.query_proj(x))  # [E, hidden]
#         gate = torch.sigmoid(self.gate_proj(x))  # [E, hidden]
#         gated_feat = query * gate  # [E, hidden]
#         scores = self.score_proj(gated_feat)  # [E, 1]
#         return scores


class GlobalPairScorer(nn.Module):
    """
    Produce unnormalized importance scores for each pair (edge),
    used ONLY for softmax pooling.
    """

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.score_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, pair_emb):
        """
        pair_emb: [E, D]
        return:   [E, 1] (unnormalized scores)
        """
        return self.score_mlp(pair_emb)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, conv_out_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, conv_out_dim)

    def forward(self, src_feat, dst_feat):
        """
        src_feat: [N_pair, D]
        dst_feat: [N_pair, D]
        return: [N_pair, conv_out_dim]
        """
        q = src_feat.unsqueeze(1)
        k = v = dst_feat.unsqueeze(1)
        attn_out, _ = self.attn(q, k, v)  # [N_pair, 1, D]
        attn_out = self.ln(attn_out.squeeze(1))  # [N_pair, D]
        return self.proj(attn_out)  # [N_pair, conv_out_dim]


# class PairEncoder(nn.Module):
#     def __init__(self, embed_dim, conv_out_dim=64, n_heads=4, dropout=0.1):
#         super().__init__()
#         self.attn = CrossAttention(embed_dim, conv_out_dim, n_heads, dropout)
#
#     def forward(self, node_feats, edge_index):
#         """
#         node_feats: [N_total, D]，可以是 inter_emb（ab+ag 拼接）
#         edge_index: [2, N_edge]，全局节点索引
#         edge_type: [N_edge] 可选，指示边是同链/跨链
#         return: [N_edge, conv_out_dim]
#         """
#         src_idx, dst_idx = edge_index  # 两端节点全局索引
#         src_feat = node_feats[src_idx]
#         dst_feat = node_feats[dst_idx]
#         pair_emb = self.attn(src_feat, dst_feat)
#
#         return pair_emb


class PairEncoder(nn.Module):
    def __init__(self,
                 node_dim,
                 pair_dim,
                 num_edge_types=3,
                 edge_type_dim=16,
                 n_heads=4,
                 dropout=0.1):
        super().__init__()

        self.edge_type_emb = nn.Embedding(num_edge_types, edge_type_dim)

        self.attn = CrossAttention(
            node_dim + edge_type_dim,
            pair_dim,
            n_heads,
            dropout
        )

    def forward(self, node_feats, edge_index, edge_type):
        """
        node_feats: [N, D]
        edge_index: [2, E]
        edge_type:  [E]  (0=intra_pc1, 1=intra_pc2, 2=inter)
        """
        src, dst = edge_index

        etype_emb = self.edge_type_emb(edge_type)  # [E, T]

        src_feat = torch.cat([node_feats[src], etype_emb], dim=-1)
        dst_feat = torch.cat([node_feats[dst], etype_emb], dim=-1)

        pair_emb = self.attn(src_feat, dst_feat)
        return pair_emb


class CrossFusionLayer(nn.Module):
    """
    残基级融合：inter_res_emb + intra_res_emb
    轻量调制
    """

    def __init__(self, inter_dim, intra_dim):
        super().__init__()
        self.proj_intra = nn.Linear(intra_dim, inter_dim)
        self.gate = nn.Linear(inter_dim * 2, inter_dim)

    def forward(self, inter_res_emb, intra_res_emb):
        intra_proj = self.proj_intra(intra_res_emb)
        fused = torch.cat([inter_res_emb, intra_proj], dim=-1)
        gate = torch.sigmoid(self.gate(fused))
        return inter_res_emb + gate * intra_proj  # residual style


class PairConditionedNodeConv(nn.Module):
    def __init__(self, node_dim, pair_dim):
        super().__init__()
        self.msg = nn.Linear(pair_dim, node_dim, bias=False)
        self.update = nn.Sequential(
            nn.LayerNorm(node_dim * 2),
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
        )

    def forward(self, node_emb, edge_index, pair_emb):
        src, dst = edge_index
        msg = self.msg(pair_emb)  # [E, node_dim]

        agg = scatter_add(msg, dst, dim=0, dim_size=node_emb.size(0))  # [N, node_dim]
        node_new = self.update(torch.cat([node_emb, agg], dim=-1))
        return node_new


class NodeConditionedPairInit(nn.Module):
    def __init__(self, pair_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, node_emb, edge_index, edge_attr):
        src, dst = edge_index
        pair_input = torch.cat(
            [node_emb[src], node_emb[dst], edge_attr],
            dim=-1
        )
        pair_emb = self.proj(pair_input)
        return pair_emb


class RelationAwarePairConv(nn.Module):
    def __init__(
            self,
            pair_dim,
            node_dim,
            edge_dim,
            hidden_dim=128,
            num_edge_types=3,
            edge_type_dim=16,
            n_energy_dim=2,
            dropout=0.1,
    ):
        super().__init__()

        self.pair_dim = pair_dim
        self.n_energy_dim = n_energy_dim

        self.edge_type_emb = nn.Embedding(num_edge_types, edge_type_dim)
        self.dir_emb = nn.Embedding(4, 8)

        self.energy_proj = nn.Linear(n_energy_dim, 1, bias=False)

        rel_in_dim = edge_dim - n_energy_dim + edge_type_dim + 8

        self.relation_gate = nn.Sequential(
            nn.LayerNorm(rel_in_dim),
            nn.Linear(rel_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pair_dim),
            nn.Sigmoid()
        )

        self.pair_update = nn.Sequential(
            nn.LayerNorm(pair_dim * 3 + rel_in_dim),
            nn.Linear(pair_dim * 3 + rel_in_dim, pair_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_dim * 2, pair_dim),
        )

        # ===== node-conditioned scaling（加入 prior）=====
        self.scale_mlp = nn.Sequential(
            nn.LayerNorm(node_dim),  # + prior
            nn.Linear(node_dim, 1),
            nn.Sigmoid()
        )

        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, pair_emb, node_emb, edge_attr, edge_index, graph):
        edge_type = graph.edge_type
        node_chain = graph.node_chain
        src, dst = edge_index

        # ===== energy =====
        energy_feat = edge_attr[:, :self.n_energy_dim]  # [E, n_energy]
        non_energy = edge_attr[:, self.n_energy_dim:]  # [E, edge_dim - n_energy]

        # ===== direction =====
        direction = node_chain[src] * 2 + node_chain[dst]  # 0~3

        # mask intra（edge_type==2 是 inter）
        is_inter = (edge_type == 2)
        direction = direction * is_inter.long()
        dir_feat = self.dir_emb(direction)

        # ===== relation =====
        etype_emb = self.edge_type_emb(edge_type)  # [E, edge_type_dim]
        rel_feat = torch.cat([non_energy, etype_emb, dir_feat], dim=-1)  # [E, rel_in_dim]

        # # ===== energy bias =====
        energy_score = self.energy_proj(energy_feat)
        energy_attn = torch.tanh(energy_score)  # [E,1]

        # ===== gate =====
        rel_gate = self.relation_gate(rel_feat)
        gated_pair = pair_emb * rel_gate * (1 + energy_attn)

        agg = scatter_mean(gated_pair, src, dim=0, dim_size=node_emb.size(0))
        pair_context = agg[src]

        # ===== update =====
        update_input = torch.cat([
            pair_emb,
            gated_pair,
            pair_context,
            rel_feat
        ], dim=-1)

        delta = self.pair_update(update_input)

        # ===== node-conditioned =====
        node_ctx = node_emb[src] + node_emb[dst]
        is_inter = (edge_type == 2).float().unsqueeze(-1)
        conf = self.scale_mlp(node_ctx) * (0.5 + 0.5 * is_inter)
        # node_prior = graph.node_type.float()  # [N,3]
        # prior_ctx = node_prior[src] + node_prior[dst]
        # conf = self.scale_mlp(torch.cat([node_ctx, prior_ctx], dim=-1))

        return pair_emb + conf * self.res_scale * delta


class NodePairBlock(nn.Module):
    def __init__(self, node_dim, pair_dim, edge_dim):
        super().__init__()
        self.node_conv = PairConditionedNodeConv(node_dim, pair_dim)
        self.pair_conv = RelationAwarePairConv(pair_dim, node_dim, edge_dim)

    def forward(self, node_emb, pair_emb, edge_index, edge_type, edge_attr):
        node_emb = self.node_conv(node_emb, edge_index, pair_emb)
        pair_emb = self.pair_conv(pair_emb, edge_attr, edge_type, node_emb, edge_index)
        return node_emb, pair_emb


class ResInterNet(nn.Module):
    def __init__(
            self,
            node_in_dim,
            edge_in_dim,
            hidden_dims,
            n_energy_dim=2,
            num_edge_types=3,
            num_layers=2,
            task_mode='reg'
    ):
        super().__init__()
        self.task_mode = task_mode
        self.hidden_dim = hidden_dims[-1]

        self.node_proj = build_mlp(node_in_dim, hidden_dims)

        self.pair_dim = self.hidden_dim * 2 + edge_in_dim
        self.pair_init = NodeConditionedPairInit(self.pair_dim, self.hidden_dim)

        self.pair_convs = nn.ModuleList([
            RelationAwarePairConv(
                pair_dim=self.hidden_dim,
                node_dim=self.hidden_dim,
                edge_dim=edge_in_dim,
                hidden_dim=128,
                num_edge_types=num_edge_types,
                n_energy_dim=n_energy_dim,
            )
            for _ in range(num_layers)
        ])

        # # ===== Node update（新增关键）=====
        # self.node_updates = nn.ModuleList([
        #     NodeUpdate(self.hidden_dim, self.hidden_dim)
        #     for _ in range(num_layers)
        # ])

        # ===== Prior-aware attention =====
        self.prior_attn = PriorAwareAttention(
            pair_dim=self.hidden_dim,
            prior_dim=6  # 两端节点各3维
        )

        # ===== Node importance（解释性）=====
        # self.node_importance = NodeImportance(self.hidden_dim)

        # self.pair_scorer = GlobalPairScorer(self.hidden_dim, 1)

        self.out_dim = self.hidden_dim * 3

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

    def forward(self, graph, cla_pred=None, pKd_pred=None):
        device = next(self.parameters()).device

        node_emb, pair_emb = self._process_inter_graph(graph, device)
        # node_imp = self.node_importance(node_emb)

        pair_repr, node_repr, pair_weight, node_weight = self._get_weighted_repr(
            pair_emb, node_emb, graph
        )

        # res_gh_repr = pair_repr
        res_gh_repr = torch.cat([pair_repr, node_repr], dim=-1)

        if self.task_mode in ["reg", "both"]:
            # res_gh_repr = res_gh_repr + self.res_adapter(res_gh_repr)
            pKd_pred = self.reg_head(res_gh_repr)
        if self.task_mode in ["cla", "both"]:
            cla_pred = self.cla_head(res_gh_repr).unsqueeze(0)

        return {
            "pKd": pKd_pred,  # scalar prediction (optional use)
            "cla": cla_pred,  # classification logits
            "graph_repr": res_gh_repr,  # pair-level interaction embedding
            "pair_repr": pair_repr,
            "node_repr": node_repr,
            "res_emb": node_emb,  # residue-level embedding (for seq gating / pooling)
            "pair_emb": pair_emb,
            "pair_weight": pair_weight,
            "node_weight": node_weight
        }

    def _process_inter_graph(self, graph, device):
        graph = graph.to(device)

        # node_feat = torch.cat([graph.pc1_node_attr, graph.pc2_node_attr], dim=0)  # [N, node_in_dim]
        node_feat = graph.x  # [N, node_in_dim]
        # node_feat = node_feat[:, :1280]
        # node_feat = torch.cat([node_feat[:, :1280], node_feat[:, -21:]], dim=-1)
        edge_attr = graph.edge_attr
        edge_index = graph.edge_index
        # node_type = graph.node_type

        node_emb = self.node_proj(node_feat)  # [N, hidden_dim]

        pair_emb = self.pair_init(
            node_emb=node_emb,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )  # [E, hidden_dim]

        for i, conv in enumerate(self.pair_convs):
            pair_emb = conv(
                pair_emb=pair_emb,
                node_emb=node_emb,
                edge_attr=edge_attr,
                edge_index=edge_index,
                graph=graph
            )

        return node_emb, pair_emb

    # ======================
    # 加权聚合（核心解释性）
    # ======================
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

        mean_pair = (inter_pair * pair_attn).sum(dim=0)
        max_pair = inter_pair.max(dim=0).values
        pair_repr = torch.cat([mean_pair, max_pair], dim=-1)

        # node_weight = torch.sigmoid(node_imp)  # [N,1]
        node_weight = (
                              scatter_mean(edge_weight, src, dim=0, dim_size=node_emb.size(0)) +
                              scatter_mean(edge_weight, dst, dim=0, dim_size=node_emb.size(0))
                      ) / 2
        node_weight = node_weight / (node_weight.sum() + 1e-6)
        node_repr = (node_emb * node_weight).sum(dim=0)

        return pair_repr, node_repr, edge_weight, node_weight

    # def _get_weighted_repr(self, pair_emb, node_emb, edge_type, INTER_EDGE_ID=2):
    #
    #     # -------- pair-level readout (inter only) --------
    #     inter_mask = (edge_type == INTER_EDGE_ID)
    #     inter_pair = pair_emb[inter_mask]
    #
    #     pair_scores = self.pair_scorer(inter_pair)  # [E_inter, 1]
    #     pair_attn = torch.softmax(pair_scores, dim=0)
    #
    #     mean_pair = (inter_pair * pair_attn).sum(0)
    #     max_pair = inter_pair.max(0).values
    #     pair_repr = torch.cat([mean_pair, max_pair], dim=-1)
    #
    #     # -------- node-level readout --------
    #     graph_repr = node_emb.mean(0)
    #     # node_max = node_emb.max(0).values
    #     # graph_repr = torch.cat([node_mean, node_max], dim=-1)
    #
    #     return pair_repr, graph_repr

    def set_requires_grad(self, keys, requires_grad: bool):
        """
        keys: set or list of parameter names
        requires_grad: True or False
        """
        for name, param in self.named_parameters():
            if name in keys:
                param.requires_grad = requires_grad

    def _compute_node_scores(self, epoch, pc1_x, pc2_x, pc1_mask=None, pc2_mask=None,
                             ab_sim=None, ag_sim=None):
        """
        ab_x, ag_x: 原始节点特征 (未投影的)
        ab_mask, ag_mask: bool tensor (全图大小)
        返回: [N_total, 1] logits
        """
        device = pc1_mask.device
        N = pc1_mask.size(0)
        logits = torch.zeros((N, 1), device=device)

        # --- MLP 打分 ---
        if pc1_mask is not None and pc1_mask.any() and pc1_x.size(0) > 0:
            pc1_logits = torch.sigmoid(self.pc1_node_scorer(pc1_x))  # [N_ab, 1]
            logits[pc1_mask] = pc1_logits

        if pc2_mask is not None and pc2_mask.any() and pc2_x.size(0) > 0:
            pc2_logits = torch.sigmoid(self.pc2_node_scorer(pc2_x))  # [N_ag, 1]
            logits[pc2_mask] = pc2_logits

        # --- 冻结前直接返回 ---
        if epoch is not None and epoch < self.freeze_epoch:
            return logits

        # --- 融合结构启发 ---
        score_alpha = torch.sigmoid(self.node_score_alpha)
        if pc1_mask is not None and pc1_mask.any() and ab_sim is not None:
            ab_diff = 1 - ab_sim
            fused_ab = (1 - score_alpha) * logits[pc1_mask] + score_alpha * ab_diff.unsqueeze(-1)
            logits[pc1_mask] = fused_ab

        if pc2_mask is not None and pc2_mask.any() and ag_sim is not None:
            ag_diff = 1 - ag_sim
            fused_ag = (1 - score_alpha) * logits[pc2_mask] + score_alpha * ag_diff.unsqueeze(-1)
            logits[pc2_mask] = fused_ag

        return logits
