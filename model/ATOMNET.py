import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add, scatter_mean


class EGNNLayer(MessagePassing):
    def __init__(self, hidden_dim, edge_feat_dim):
        super().__init__(aggr="add")

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feat_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, x, pos, edge_index, edge_feat):
        row, col = edge_index
        rel = pos[row] - pos[col]
        dist2 = (rel ** 2).sum(dim=-1, keepdim=True)

        m = self.edge_mlp(
            torch.cat([x[row], x[col], edge_feat, dist2], dim=-1)
        )

        pos = pos + scatter_add(
            self.coord_mlp(m) * rel,
            row, dim=0, dim_size=pos.size(0)
        )

        agg = scatter_add(m, row, dim=0, dim_size=x.size(0))
        x = self.node_mlp(torch.cat([x, agg], dim=-1))

        return x, pos


class AtomInterNet(nn.Module):
    def __init__(
        self,
        node_dim=14,
        edge_dim=21,
        edge_attr_dim=5,      # Vina 5 项
        hidden_dim=256,
        num_layers=3,
        task_mode='reg'
    ):
        super().__init__()
        self.task_mode = task_mode
        self.node_emb = nn.Linear(node_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_dim, hidden_dim)
        self.out_dim = hidden_dim * 2 #+ edge_attr_dim

        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # pair 表征（用于 affinity）
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # vina 权重（依赖 pair）
        self.edge_weight_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_attr_dim)   # 对应 vina 5 项
        )
        self.energy_scale = nn.Parameter(torch.ones(1, 5))

        # if self.task_mode in ["reg", "both"]:
        # self.atom_head = nn.Sequential(
        #     nn.LayerNorm(self.out_dim),
        #     nn.Linear(self.out_dim, self.out_dim // 2),
        #     nn.SiLU(),
        #     nn.Linear(self.out_dim // 2, 1)
        # )
        self.phys_head = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, 8),
            nn.SiLU(),
            nn.Linear(8, 1)
        )
        self.struct_head = nn.Sequential(
            nn.LayerNorm(self.out_dim),
            nn.Linear(self.out_dim, self.out_dim // 2),
            nn.SiLU(),
            nn.Linear(self.out_dim // 2, 1)
        )
        self.conf_head = nn.Sequential(
            nn.LayerNorm(self.out_dim),
            nn.Linear(self.out_dim, self.out_dim // 2),
            nn.SiLU(),
            nn.Linear(self.out_dim // 2, 1)
        )

    def forward(self, data):
        """
        data.x         [N, node_dim]
        data.pos       [N, 3]
        data.edge_index[2, E]
        data.edge_attr [E, 5]   # 已算好的 vina 五项
        """
        device = next(self.parameters()).device
        data = data.to(device)

        pos = data.coords
        edge_index = data.edge_index
        edge_dist = data.edge_dist
        row, col = edge_index

        x = self.node_emb(data.x)
        edge_feat = self.edge_emb(data.edge_feat)

        # ---- EGNN 更新 atom embedding ----
        for layer in self.layers:
            x, pos = layer(x, pos, edge_index, edge_feat)

        res_id = data.atom2residue  # 全局 residue id
        _, inv = torch.unique(res_id, return_inverse=True)
        x_res = scatter_mean(x, inv, dim=0)

        # ---- 计算 edge-level vina 权重 ----
        h_ij = torch.cat([x[row], x[col], edge_dist.unsqueeze(-1)], dim=-1)   # [E, 2H + 5 + 1]
        pair_emb = self.pair_mlp(h_ij)  # [E, H]
        # w = torch.sigmoid(self.edge_weight_mlp(pair_emb)) + 0.5  # w: [E, 5]
        w = 1.0 + 0.2 * torch.tanh(self.edge_weight_mlp(pair_emb))

        # ---- Vina energy 加权 ----
        edge_E = w * data.edge_attr    # [E, 5]
        mask = (edge_dist < 4.0) & (data.edge_feat[:, -1] > 0.5)
        pair_emb_i = pair_emb[mask]
        edge_E_i = edge_E[mask]
        pair_gh = torch.cat([
            pair_emb_i.mean(dim=0),
            pair_emb_i.max(dim=0).values
        ], dim=-1).unsqueeze(0)
        # pair_gh = pair_emb_i.mean(dim=0, keepdim=True)
        E_terms = edge_E_i.sum(dim=0, keepdim=True) * 0.5
        E_terms = E_terms * self.energy_scale

        pc_flag = data.x[:, 0].long()  # [N], 0 or 1
        x_pc = scatter_mean(x, pc_flag, dim=0)  # [2, hidden]
        pc1_node_gh = x_pc[0:1]  # pc_flag == 0
        pc2_node_gh = x_pc[1:2]  # pc_flag == 1

        phys = self.phys_head(E_terms)  # 小 MLP
        struct = self.struct_head(pair_gh)  # 小 MLP
        conf = self.conf_head(torch.cat([pc1_node_gh, pc2_node_gh], dim=-1))

        pKd_pred = phys + struct + conf

        return pKd_pred, x_res

        # atom_gh_repr = torch.cat([E_terms, pair_gh, pc1_node_gh, pc2_node_gh], dim=-1)
        # if self.task_mode in ["reg", "both"]:
        # pKd_pred = self.atom_head(atom_gh_repr)

        # # 聚合到残基层
        # row_res = inv[row]  # 源原子对应残基索引
        # col_res = inv[col]  # 目标原子对应残基索引
        # # 残基间能量 = sum(edge_E) 按残基对聚合
        # res_edge_sum = torch.zeros((x_res.size(0), 5), device=device)
        # res_edge_sum.index_add_(0, row_res, edge_E)
        # res_edge_sum.index_add_(0, col_res, edge_E)
        # res_edge_sum = res_edge_sum * 0.5  # 防止重复计算
        #
        # # ---- 全局图表示 ----
        # node_graph = x_res.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        # global_energy = res_edge_sum.mean(dim=0, keepdim=True)  # [1, 5]
        # atom_gh_repr = torch.cat([global_energy, node_graph], dim=-1)  # [1, hidden+5]

