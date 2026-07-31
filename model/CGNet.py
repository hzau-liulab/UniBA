from collections.abc import Sequence
from torchdrug.data import constant
from torchdrug import layers
from utils.cg_graphconstruct import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv


def pool_node_emb(X: Tensor, pooling: str = 'meanmax', score_fn=None) -> Tensor:
    if pooling == 'mean':
        return X.mean(dim=0)
    if pooling == 'max':
        return X.max(dim=0).values
    if pooling == 'meanmax':
        mean = X.mean(dim=0)
        maxv = X.max(dim=0).values
        return torch.cat([mean, maxv], dim=0)
    if pooling == 'attn':
        assert score_fn is not None, \
            "score_fn must be provided when pooling='attn'"
        score = score_fn(X).squeeze(-1)
        alpha = torch.softmax(score, dim=0)
        return (X * alpha.unsqueeze(-1)).sum(dim=0)

    raise ValueError(f"Unknown pooling mode: {pooling}")


class _BaseAttentionPool(nn.Module):
    def __init__(self, dim, use_ln=True):
        super().__init__()
        self.score_fn = nn.Linear(dim, 1)
        self.ln = nn.LayerNorm(dim) if use_ln else None

    def compute_score(self, x):
        if self.ln is not None:
            x = self.ln(x)
        return self.score_fn(x).squeeze(-1), x


class ResidueAttentionPool(_BaseAttentionPool):
    def __init__(self, bead_dim, method='attn', use_ln=False):
        super().__init__(bead_dim, use_ln)
        self.method = method

    def forward(self, bead_emb, bead2residue):
        num_res = int(bead2residue.max().item()) + 1

        if self.method == 'mean':
            return scatter_mean(bead_emb, bead2residue, dim=0, dim_size=num_res)

        if self.method == 'max':
            return scatter_max(bead_emb, bead2residue, dim=0, dim_size=num_res)[0]

        # score, x = self.compute_score(bead_emb)
        # alpha = scatter_softmax(score, bead2residue, dim=0)
        #
        # return scatter_add(
        #     x * alpha.unsqueeze(-1),
        #     bead2residue,
        #     dim=0,
        #     dim_size=num_res
        # )


class GraphAttentionPool(_BaseAttentionPool):
    def __init__(self, node_dim, method='attn', use_ln=False):
        super().__init__(node_dim, use_ln)
        self.method = method

        if method == 'attn':
            # residual attention strength
            self.gamma = nn.Parameter(torch.tensor(0.0))
        else:
            self.gamma = None

    def forward(self, X):
        mean = X.mean(dim=0)

        if self.method == 'mean':
            return mean

        if self.method == 'max':
            return X.max(dim=0).values

        if self.method == 'meanmax':
            maxv = X.max(dim=0).values
            return torch.cat([mean, maxv], dim=0)

        score, x = self.compute_score(X)
        alpha = torch.softmax(score, dim=0)
        attn = torch.sum(alpha.unsqueeze(-1) * x, dim=0)

        return mean + self.gamma * attn


class CrossAttentionPooling(nn.Module):
    """
    通用 Cross-Attention Pooling 模块
    - 支持任意两组节点嵌入 (X1, X2)
    - 双向 cross-attention 聚合
    - 支持 mean / max / mean+max 池化
    """

    def __init__(self, embed_dim, dropout=0.1, pooling='meanmax'):
        super().__init__()
        self.embed_dim = embed_dim
        self.pooling = pooling

        # QKV & 输出映射（共享权重）
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # 初始化
        for m in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, X1, X2, mask1=None, mask2=None):
        """双向 cross-attention 聚合"""
        Q1, K1, V1 = self.q_proj(X1), self.k_proj(X1), self.v_proj(X1)
        Q2, K2, V2 = self.q_proj(X2), self.k_proj(X2), self.v_proj(X2)

        # X1 ↔ X2 注意力
        attn12 = torch.matmul(Q1, K2.T) / (self.embed_dim ** 0.5)
        attn21 = torch.matmul(Q2, K1.T) / (self.embed_dim ** 0.5)

        if mask2 is not None:
            attn12.masked_fill_(~mask2.unsqueeze(0), float('-inf'))
        if mask1 is not None:
            attn21.masked_fill_(~mask1.unsqueeze(0), float('-inf'))

        attn12 = F.softmax(attn12, dim=-1)
        attn21 = F.softmax(attn21, dim=-1)

        X1_out = self.out_proj(torch.matmul(self.dropout(attn12), V2))
        X2_out = self.out_proj(torch.matmul(self.dropout(attn21), V1))

        return pool_node_emb(X1_out, self.pooling), pool_node_emb(X2_out, self.pooling)


class CGIntraNet(nn.Module):
    num_class = constant.NUM_AMINO_ACID  # 20

    def __init__(self, input_dim, edge_input_dim, hidden_dims, num_relation, num_angle_bin,
                 short_cut=True, batch_norm=False, activation="relu", concat_hidden=False,
                 num_mlp_layer=2, cg_gh_builder=None, task_mode='reg'):
        super().__init__()
        self.finetune = False
        self.task_mode = task_mode

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.hd_out_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.edge_dims = [edge_input_dim] + self.dims[:-1]
        self.num_relation = num_relation
        self.num_angle_bin = num_angle_bin
        self.num_mlp_layer = num_mlp_layer
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.batch_norm = batch_norm
        self.cg_gh_builder = cg_gh_builder
        self.layers = nn.ModuleList()

        for i in range(len(self.dims) - 1):
            self.layers.append(layers.GeometricRelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                   edge_input_dim, batch_norm, activation))
        if num_angle_bin:
            self.spatial_line_graph = layers.SpatialLineGraph(num_angle_bin)
            self.edge_layers = nn.ModuleList()
            for i in range(len(self.edge_dims) - 1):
                self.edge_layers.append(layers.GeometricRelationalGraphConv(
                    self.edge_dims[i], self.edge_dims[i + 1], num_angle_bin, None, batch_norm, activation))

        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))

        # self.residue_pool = ResidueAttentionPool(self.hd_out_dim, method='mean')

        # self.seq_proj = nn.Sequential(
        #     nn.Linear(1280, self.hd_out_dim*2),
        #     nn.ReLU(),
        #     nn.LayerNorm(self.hd_out_dim*2)
        # )

        # self.cross_pool_res = CrossAttentionPooling(self.output_dim, pooling='meanmax')
        # self.cross_pool_bead = CrossAttentionPooling(self.output_dim, pooling='meanmax')

        self.out_dim = self.hd_out_dim*4

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hd_out_dim,
            num_heads=2,
            dropout=0.1,
            batch_first=True
        )

        # residue-level topology refinement
        # ---------------------------------------------------------
        # self.res_layers = nn.ModuleList([
        #     GATConv(self.out_dim, self.out_dim // 2, heads=2, concat=True, edge_dim=1, dropout=0.1),
        # ])
        #
        # self.graph_pool_gate = nn.Sequential(
        #     nn.Linear(self.out_dim, self.out_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(self.out_dim // 2, 1)
        # )

        # self.graph_pool = GraphAttentionPool(self.out_dim, method='attn', use_ln=True)
        if self.task_mode in ["reg", "both"]:
            self.reg_head = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim // 2),
                nn.ReLU(),
                nn.Linear(self.out_dim // 2, 1)
            )

        if self.task_mode in ["cla", "both"]:
            self.cla_head = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim // 2),
                nn.ReLU(),
                nn.Linear(self.out_dim // 2, self.out_dim // 4),
                nn.ReLU(),
                nn.Linear(self.out_dim // 4, 2)
            )

    def forward(self, intra_graphs, cla_pred=None, pKd_pred=None):
        graph_repr, bead_emb = self.encode_intra(intra_graphs)

        # res_emb, bead_emb = self._process_cg_graph(cg_graph)
        # cg_gh_emb = pool_node_emb(bead_emb)

        # score = self.graph_pool_gate(res_emb)
        # attn = torch.softmax(score, dim=0)
        # cg_gh_emb = torch.sum(attn * res_emb, dim=0, keepdim=True)

        if self.task_mode in ["reg", "both"]:
            # pKd_pred = self.reg_head(cg_gh_emb)
            pKd_pred = self.reg_head(graph_repr)
        if self.task_mode in ["cla", "both"]:
            cla_pred = self.cla_head(graph_repr) #.unsqueeze(0)

        return {
            "pKd": pKd_pred,  # scalar prediction
            "cla": cla_pred,  # classification logits
            "graph_repr": graph_repr,  # pair-level CG embedding
            # "res_emb": res_emb,  # residue-level CG embedding
            "bead_emb": bead_emb  # bead-level embedding (optional, future use)
        }

    def forward_ddg(self, wt_graphs, mut_graphs, cla_pred=None, ddg_pred=None):
        wt_emb = self.encode_intra(wt_graphs)
        mut_emb = self.encode_intra(mut_graphs)
        diff_emb = mut_emb - wt_emb
        if self.task_mode in ["cla", "both"]:
            cla_pred = self.cla_head(diff_emb).unsqueeze(0)
        if self.task_mode in ["reg", "both"]:
            ddg_pred = self.intra_head(diff_emb)
        return ddg_pred, cla_pred

    def encode_intra(self, graphs):
        device = next(self.parameters()).device

        if isinstance(graphs[0], torch.Tensor):
            pc1_seq_feat, pc2_seq_feat = (x.to(device) for x in graphs)
            pc1_repr = self._process_intra_feat(pc1_seq_feat)
            pc2_repr = self._process_intra_feat(pc2_seq_feat)
            emb = torch.cat([pc1_repr, pc2_repr], dim=-1)
        else:
            # emb = torch.cat([self._encode_chain(g) for g in graphs], dim=-1)
            chain_A = self._encode_chain(graphs[0])
            chain_B = self._encode_chain(graphs[1])

            bead_A = chain_A["bead_emb"]
            bead_B = chain_B["bead_emb"]

            cross_A, _ = self.cross_attn(
                query=bead_A.unsqueeze(0),
                key=bead_B.unsqueeze(0),
                value=bead_B.unsqueeze(0)
            )

            cross_B, _ = self.cross_attn(
                query=bead_B.unsqueeze(0),
                key=bead_A.unsqueeze(0),
                value=bead_A.unsqueeze(0)
            )

            cross_A = cross_A.squeeze(0)
            cross_B = cross_B.squeeze(0)

            bead_A = bead_A + 0.1 * cross_A
            bead_B = bead_B + 0.1 * cross_B

            graph_A = pool_node_emb(bead_A)
            graph_B = pool_node_emb(bead_B)

            graph_emb = torch.cat([
                graph_A * graph_B,
                torch.abs(graph_A - graph_B)
            ], dim=-1)

            bead_emb = torch.cat([
                bead_A,
                bead_B
            ], dim=0)

        return graph_emb, bead_emb  # [D]

    def _process_intra_feat(self, seq_feat):
        if seq_feat.dim() == 1:
            L = seq_feat.numel()
            assert L % 1280 == 0, f"Invalid seq_feat length: {L}"

            n_chain = L // 1280
            seq_feat = seq_feat.view(n_chain, 1280)  # [n_chain, 1280]

        seq_emb = self.seq_proj(seq_feat)  # [n_chain, hidden]
        seq_repr = seq_emb.mean(dim=0)  # [hidden]
        return seq_repr

    def _encode_chain(self, graph):

        res_emb, bead_emb = self._process_cg_graph(graph)

        return {
            "res_emb": res_emb,
            "bead_emb": bead_emb
        }

    # def _encode_chain(self, graph):
    #     res_emb, bead_emb = self._process_cg_graph(graph)
    #     struct_repr = pool_node_emb(res_emb)  # [hidden]
    #     # seq_feat = graph.seq_feat.to(self.device)
    #     # seq_repr = self._process_intra_feat(seq_feat)
    #     return struct_repr #+ seq_repr

    # def _encode_chain(self, graph):
    #     res_emb, bead_emb = self._process_intra_graph(graph)
    #     struct_repr = pool_node_emb(bead_emb)  # [hidden]
    #     seq_feat = graph.seq_feat.to(self.device)
    #     if seq_feat.dim() == 1:
    #         L = seq_feat.numel()
    #         assert L % 1280 == 0, f"Invalid seq_feat length: {L}"
    #
    #         n_chain = L // 1280
    #         seq_feat = seq_feat.view(n_chain, 1280)  # [n_chain, 1280]
    #
    #     seq_emb = self.seq_proj(seq_feat)  # [n_chain, hidden]
    #     seq_repr = seq_emb.mean(dim=0)  # [hidden]
    #
    #     return struct_repr + seq_repr

    def _build_res_top_graph(self, graph, bead2residue):
        """
        Build residue-level topology graph
        induced from CG bead graph.

        Returns
        -------
        res_edge_index : [2, E_res]
        edge_attr      : [E_res, 2]

        edge_attr:
            [:, 0] = normalized bead contact count
        """

        device = self.device

        # =====================================================
        # bead edges
        # =====================================================
        bead_edge_index = graph.edge_list[:, :2]

        src_bead = bead_edge_index[:, 0]
        dst_bead = bead_edge_index[:, 1]

        # =====================================================
        # bead -> residue
        # =====================================================
        src_res = bead2residue[src_bead]
        dst_res = bead2residue[dst_bead]

        # remove same residue edges
        mask = src_res != dst_res

        src_res = src_res[mask]
        dst_res = dst_res[mask]

        # =====================================================
        # aggregate bead contacts
        # =====================================================
        edge_dict = {}

        for i in range(src_res.shape[0]):

            r1 = int(src_res[i].item())
            r2 = int(dst_res[i].item())

            key = (r1, r2)

            if key not in edge_dict:
                edge_dict[key] = 0

            edge_dict[key] += 1

        # =====================================================
        # build residue edges
        # =====================================================
        res_src, res_dst, edge_attr = [], [], []
        max_count = max(edge_dict.values())

        for (r1, r2), count in edge_dict.items():
            res_src.append(r1)
            res_dst.append(r2)

            # normalized contact count
            count_feat = count / max_count

            edge_attr.append([count_feat])

        res_edge_index = torch.tensor(
            [res_src, res_dst],
            dtype=torch.long,
            device=device
        )

        edge_attr = torch.tensor(
            edge_attr,
            dtype=torch.float,
            device=device
        )

        return res_edge_index, edge_attr

    def _res_gat(self, res_emb, res_edge_index, edge_attr):
        x_res = res_emb

        for i in range(len(self.res_layers)):
            h_res = self.res_layers[i](x_res, res_edge_index, edge_attr)
            # # residual
            # if self.short_cut and h_res.shape == x_res.shape:
            #     h_res = h_res + x_res
            #
            # # batch norm
            # if self.batch_norm:
            #     h_res = self.res_batch_norms[i](h_res)

            h_res = F.relu(h_res)
            x_res = x_res + h_res

        return x_res

    def _process_cg_graph(self, graph):
        def pool_by_residue(bead_emb, bead2residue, method='meanmax'):
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

        if self.cg_gh_builder:
            # currently no cg_gh_builder.apply_node_layer function is used here (None)
            graph = self.cg_gh_builder.apply_node_layer(graph)
            # AdvSpatialEdge in cg_transform script
            graph = self.cg_gh_builder.apply_edge_layer(graph)

        x = graph.atom_feature.float().to(self.device)  # (num_node, input_dim=25)
        hiddens = []

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

        bead2residue = graph.bead2residue.to(self.device)
        # res_emb = self.residue_pool(bead_emb, graph.bead2residue.to(self.device))
        res_emb = pool_by_residue(bead_emb, bead2residue, 'meanmax')

        # res_edge_index, edge_attr = self._build_res_top_graph(graph, bead2residue)
        # res_emb = self._res_gat(res_emb, res_edge_index, edge_attr)

        return res_emb, bead_emb
