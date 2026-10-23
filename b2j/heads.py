"""Module a: K independent heads on a [B, D] embedding -> [B, K, out_dim].

MLP head per plan §2: D -> 256 -> out_dim, GELU, dropout 0.1. Linear head: D -> out_dim.
"""
import torch
import torch.nn as nn


def _head(in_dim, out_dim, head_type, hidden, dropout):
    if head_type == "mlp":
        return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim))
    if head_type == "linear":
        return nn.Linear(in_dim, out_dim)
    raise ValueError(f"unknown head type {head_type}")


class JointHead(nn.Module):
    """One MLP for all K labels (round D, D6) -> [B, K, out_dim].

    shared_concentration=False: D -> 256 -> K·out_dim (e.g. B²-r's per-label alphas from a shared trunk).
    shared_concentration=True (Dirichlet-Multinomial): D -> 256 -> K + 1 = K class logits and one log concentration per
    image, returned as [B, K, 2] with the log concentration repeated in [..., 1].
    """

    def __init__(self, in_dim: int, num_heads: int, out_dim: int, shared_concentration: bool = False,
                 hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        if shared_concentration and out_dim != 2:
            raise ValueError("a shared concentration needs out_dim 2")
        self.K, self.out_dim, self.shared = num_heads, out_dim, shared_concentration
        self.net = _head(in_dim, num_heads + 1 if shared_concentration else num_heads * out_dim, "mlp", hidden, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        o = self.net(z)
        if self.shared:
            return torch.stack([o[:, :self.K], o[:, self.K:].expand(-1, self.K)], -1)
        return o.view(len(z), self.K, self.out_dim)


class MultiHead(nn.Module):
    def __init__(self, in_dim: int, num_heads: int, out_dim: int,
                 head_type: str = "mlp", hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleList(_head(in_dim, out_dim, head_type, hidden, dropout) for _ in range(num_heads))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.stack([h(z) for h in self.heads], dim=1)


class RaterOffsets(nn.Module):
    """Round D, D8: a head plus one learned bias per rater and label, β [K, R], centred over the raters so the panel
    average is the 'typical rater'. Output [B, K, d + R]: the head's d outputs, then β repeated on every row (so the
    saved predictions carry β and the loss sees it without a second argument)."""

    def __init__(self, head: nn.Module, num_heads: int, num_raters: int):
        super().__init__()
        self.head = head
        self.bias = nn.Parameter(torch.zeros(num_heads, num_raters))

    def beta(self) -> torch.Tensor:
        return self.bias - self.bias.mean(1, keepdim=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        o = self.head(z)
        return torch.cat([o, self.beta().to(o.dtype)[None].expand(len(z), -1, -1)], -1)

