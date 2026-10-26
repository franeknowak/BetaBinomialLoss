"""K independent heads on a [B, D] embedding -> [B, K, out_dim].

MLP head: D -> 256 -> out_dim, GELU, dropout 0.1. Linear head: D -> out_dim.
"""
import torch
import torch.nn as nn


def _head(in_dim, out_dim, head_type, hidden, dropout):
    if head_type == "mlp":
        return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim))
    if head_type == "linear":
        return nn.Linear(in_dim, out_dim)
    raise ValueError(f"unknown head type {head_type}")


class MultiHead(nn.Module):
    def __init__(self, in_dim: int, num_heads: int, out_dim: int,
                 head_type: str = "mlp", hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleList(_head(in_dim, out_dim, head_type, hidden, dropout) for _ in range(num_heads))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.stack([h(z) for h in self.heads], dim=1)
