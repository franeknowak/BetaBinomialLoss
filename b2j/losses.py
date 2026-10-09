"""Module a: per-sample (k, n) losses for K independent binary heads.

Every function works on whatever device its inputs live on (no CPU round trip).
Shapes: k, n are float tensors [B, K]; head outputs are [B, K, d] with d = 2 for
B² heads (logit for alpha_0, logit for alpha_1) and d = 1 for BCE/DUP heads.
See journal_plan/specs/a_heads_losses.md.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

STUCK_LOGIT = -7.0


# ---------------------------------------------------------------- label helpers
def majority(k: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Majority positive = strictly more than half the panel voted positive."""
    return (2 * k > n).to(k.dtype)


def split_target(k: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Unbiased P(split) among 3 raters drawn without replacement from the n votes.

    Equals the indicator 1[0 < k < 3] when n = 3. Requires n >= 3.
    """
    if bool((n < 3).any()):
        raise ValueError("split_target needs n >= 3")
    denom = n * (n - 1) * (n - 2)
    m = n - k
    unanimous = (k * (k - 1) * (k - 2) + m * (m - 1) * (m - 2)) / denom
    return 1.0 - unanimous


def class_weights(k: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Majority-vote weights w± = N / (2 N±) per head, from training (k, n). Returns [K, 2] = (w_neg, w_pos)."""
    y = majority(k, n)
    N = y.shape[0]
    n_pos = y.sum(0).clamp(min=1)
    n_neg = (N - y.sum(0)).clamp(min=1)
    return torch.stack([N / (2 * n_neg), N / (2 * n_pos)], dim=-1)


def moment_matched_prior(k: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Corrected KL prior per head: pi = mean vote fraction, nu = method-of-moments Beta-Binomial S.

    Uses Var(k/n) = pi(1-pi) [rho + (1-rho) E(1/n)], rho = 1/(S+1). Returns [K, 2] = (a0, a1).
    """
    f = k / n
    pi = f.mean(0)
    v = f.var(0, unbiased=True)
    h = (1.0 / n).mean(0)
    rho = ((v / (pi * (1 - pi))) - h) / (1 - h)
    rho = rho.clamp(1e-6, 1 - 1e-6)
    S = 1.0 / rho - 1.0
    return torch.stack([(1 - pi) * S, pi * S], dim=-1)


def miccai_prior(k: torch.Tensor, n: torch.Tensor, nu: float = 2.0) -> torch.Tensor:
    """MICCAI prior: pi = majority-positive prevalence, nu = 2. Returns [K, 2] = (a0, a1)."""
    pi = majority(k, n).mean(0)
    return torch.stack([(1 - pi) * nu, pi * nu], dim=-1)


# ---------------------------------------------------------------- B² pieces
def alphas(logits: torch.Tensor, eps: float, plus_one: bool = False):
    """[..., 2] logits -> (a0, a1). B²-r: softplus + eps. B²-M: softplus + 1e-6 + 1."""
    a = F.softplus(logits) + eps
    if plus_one:
        a = a + 1.0
    return a[..., 0], a[..., 1]


def _log_beta(x, y):
    return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)


def beta_binomial_nll(a0, a1, k, n):
    """Full Beta-Binomial NLL of k positives out of n under Beta(a1, a0), including log C(n, k)."""
    log_comb = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
    return -(log_comb + _log_beta(k + a1, n - k + a0) - _log_beta(a1, a0))


def beta_kl(a0, a1, p0, p1):
    """KL( Beta(a1, a0) || Beta(p1, p0) ), analytic."""
    s, ps = a0 + a1, p0 + p1
    return (_log_beta(p1, p0) - _log_beta(a1, a0)
            + (a1 - p1) * torch.digamma(a1) + (a0 - p0) * torch.digamma(a0)
            + (ps - s) * torch.digamma(s))


def stuck_share(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Per head: share of samples with k >= 1 whose alpha_1 logit is below -7. Returns [K]."""
    has_pos = k >= 1
    stuck = (logits[..., 1] < STUCK_LOGIT) & has_pos
    return stuck.sum(0) / has_pos.sum(0).clamp(min=1)


# ---------------------------------------------------------------- the loss
@dataclass
class LossConfig:
    kind: str                       # 'hard_bce' | 'soft_bce' | 'b2' | 'dup'
    eps: float = 1e-3               # B² positivity; fallback 0.1 (R0 smoke test rule)
    plus_one: bool = False          # legacy B²-M only
    weighted: bool = False          # majority-vote per-sample weights w±
    kl_lambda: float = 0.0          # 0 = no KL

    def __post_init__(self):
        if self.kind not in ("hard_bce", "soft_bce", "b2", "dup"):
            raise ValueError(f"unknown loss kind {self.kind}")
        if self.kl_lambda and self.kind != "b2":
            raise ValueError("KL only applies to B² heads")
        if self.plus_one and self.kind != "b2":
            raise ValueError("plus_one only applies to B² heads")
        if self.weighted and self.kind not in ("hard_bce", "b2"):
            raise ValueError("weighting is defined for hard BCE and B² only")

    @property
    def out_dim(self) -> int:
        return 2 if self.kind == "b2" else 1


class PanelLoss(nn.Module):
    """Sum over heads of the batch-mean per-sample loss.

    weights: [K, 2] (w_neg, w_pos) from class_weights(), required iff cfg.weighted.
    prior:   [K, 2] (a0, a1), required iff cfg.kl_lambda > 0.
    """

    def __init__(self, cfg: LossConfig, weights=None, prior=None):
        super().__init__()
        self.cfg = cfg
        if cfg.weighted and weights is None:
            raise ValueError("weighted loss needs class weights")
        if cfg.kl_lambda and prior is None:
            raise ValueError("KL needs a prior")
        self.register_buffer("weights", None if weights is None else torch.as_tensor(weights, dtype=torch.float32))
        self.register_buffer("prior", None if prior is None else torch.as_tensor(prior, dtype=torch.float32))

    def per_sample(self, out: torch.Tensor, k: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg.kind == "b2":
            a0, a1 = alphas(out, cfg.eps, cfg.plus_one)
            loss = beta_binomial_nll(a0, a1, k, n)
        else:
            z = out[..., 0]
            target = {"hard_bce": lambda: majority(k, n),
                      "soft_bce": lambda: k / n,
                      "dup": lambda: split_target(k, n)}[cfg.kind]()
            loss = F.binary_cross_entropy_with_logits(z, target, reduction="none")
        if cfg.weighted:
            y = majority(k, n)
            loss = loss * (y * self.weights[:, 1] + (1 - y) * self.weights[:, 0])
        if cfg.kl_lambda:
            loss = loss + cfg.kl_lambda * beta_kl(a0, a1, self.prior[:, 0], self.prior[:, 1])
        return loss                                     # [B, K]

    def forward(self, out, k, n):
        return self.per_sample(out, k, n).mean(0).sum()
