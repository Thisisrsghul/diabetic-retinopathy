import torch
import torch.nn as nn
from config import Config


class CrossFeatureAttention(nn.Module):
    """Multi-head self-attention over the 1310-D fused vector."""

    def __init__(self, dim=Config.FUSED_DIM,
                 heads=Config.ATTN_HEADS, dropout=Config.DROPOUT):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = x.unsqueeze(1)                    # (B,1,D)
        a, _ = self.attn(h, h, h)
        x = x + self.dropout(a.squeeze(1))
        x = self.norm1(x)
        x = x + self.dropout(self.ffn(x))
        x = self.norm2(x)
        return x


class GatedFusionAttention(nn.Module):
    """Channel-wise gating (squeeze-excite style)."""

    def __init__(self, dim=Config.FUSED_DIM,
                 reduction=16, dropout=Config.DROPOUT):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, max(dim // reduction, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(dim // reduction, 1), dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x * self.gate(x))