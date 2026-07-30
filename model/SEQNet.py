import torch
import torch.nn as nn


class SeqEncoder(nn.Module):
    def __init__(
        self,
        emb_dim=1280,
        hidden_dim=512,
        dropout=0.1,
        task_mode='reg'
    ):
        super().__init__()

        self.pre = nn.Linear(emb_dim, hidden_dim)

        self.pair_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.reg_head = nn.Linear(hidden_dim, 1)

    def forward(self, seq_data):
        emb_A, emb_B = seq_data.chunk(2, dim=-1)
        a = self.pre(emb_A)
        b = self.pre(emb_B)
        # pair_emb = torch.cat([a, b, torch.abs(a - b), a * b], dim=-1)

        pair_emb = torch.cat([
            torch.abs(a - b),
            a * b,
            (a + b) / 2
        ], dim=-1)

        pair = self.pair_proj(pair_emb)
        pair = self.trunk(pair)
        pKd_pred = self.reg_head(pair)

        return pKd_pred.squeeze(0), pair.unsqueeze(0)
