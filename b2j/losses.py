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


LOG_S_BOUNDS = (-10.0, 15.0)        # mean-spread heads: log S clamp (S from 4.5e-5 to 3.3e6)
LOGIT_M_BOUND = 30.0                 # |logit m| clamp: m in [1e-13, 1 - 1e-13], logs stay finite in float64


def ms_alphas(out: torch.Tensor):
    """Mean-spread head (round D, D1): [..., 2] = (logit m, log S) -> (a0, a1, log S) with a1 = m S, a0 = (1 - m) S.

    Built in log space (log a1 = log sigmoid(l) + log S) so neither alpha underflows to 0.
    """
    lm = out[..., 0].clamp(-LOGIT_M_BOUND, LOGIT_M_BOUND)
    log_s = out[..., 1].clamp(*LOG_S_BOUNDS)
    return torch.exp(F.logsigmoid(-lm) + log_s), torch.exp(F.logsigmoid(lm) + log_s), log_s


def _log_beta(x, y):
    return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)


def _on_cpu(f, *xs):
    """Evaluate f on CPU copies and move the result back (autograd flows through the transfers).

    torch.lgamma / digamma on CUDA fail on some nodes of the UCL cluster (PyTorch 2.5.1+cu118, A6000): SIGILL in
    the MICCAI work (GitHub issue #22) and 'CUDA driver error: invalid argument' in job 7522356. Workaround from #22.
    """
    dev = xs[0].device
    return f(*(x.cpu() for x in xs)).to(dev)


def beta_binomial_nll(a0, a1, k, n):
    """Full Beta-Binomial NLL of k positives out of n under Beta(a1, a0), including log C(n, k).

    k and n are integer-valued, so B(k + a1, n - k + a0) / B(a1, a0) is evaluated as finite products
    Π_{j<k}(a1 + j) Π_{j<n-k}(a0 + j) / Π_{j<n}(a0 + a1 + j): exact, only `log` on the GPU (no lgamma, see
    _on_cpu), and stable at large S. log C(n, k) has no parameters and is computed on the CPU.
    """
    log_comb = _on_cpu(lambda n_, k_: torch.lgamma(n_ + 1) - torch.lgamma(k_ + 1) - torch.lgamma(n_ - k_ + 1), n, k)
    s = a0 + a1
    ll = torch.zeros_like(s)
    for j in range(int(n.max().item())):
        ll = ll + torch.where(j < k, torch.log(a1 + j), 0.0) + torch.where(j < n - k, torch.log(a0 + j), 0.0) \
            - torch.where(j < n, torch.log(s + j), 0.0)
    return -(log_comb + ll)


def beta_kl(a0, a1, p0, p1):
    """KL( Beta(a1, a0) || Beta(p1, p0) ), analytic; evaluated on the CPU (lgamma/digamma, see _on_cpu)."""
    def kl(a0, a1, p0, p1):
        s, ps = a0 + a1, p0 + p1
        return (_log_beta(p1, p0) - _log_beta(a1, a0)
                + (a1 - p1) * torch.digamma(a1) + (a0 - p0) * torch.digamma(a0)
                + (ps - s) * torch.digamma(s))
    return _on_cpu(kl, a0, a1, *torch.broadcast_tensors(p0, p1))


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
    param: str = "alpha"            # B² head output: 'alpha' (two softplus alphas) | 'ms' (logit m, log S; round D)
    s_shrink: float = 0.0           # 'ms' only: lambda * (log S - target)^2 (round D, D2)
    s_shrink_target: str = "prior"  # 'prior': log S of the moment-matched prior = spread ACROSS images (D2, invalid:
                                    #   ignores x, ~0.5 on CIFAR-10H); 'batch': the batch mean of log S (D2b), i.e. a
                                    #   penalty on the spread's variation between images; its level is learned
    n_weight: bool = False          # soft BCE only (round D, D5): per-sample weight n / batch mean of n, i.e. the
                                    #   Binomial NLL (= n × soft BCE + const) rescaled to the soft-BCE loss level

    def __post_init__(self):
        if self.kind not in ("hard_bce", "soft_bce", "b2", "dup"):
            raise ValueError(f"unknown loss kind {self.kind}")
        if self.kl_lambda and self.kind != "b2":
            raise ValueError("KL only applies to B² heads")
        if self.plus_one and self.kind != "b2":
            raise ValueError("plus_one only applies to B² heads")
        if self.weighted and self.kind not in ("hard_bce", "b2"):
            raise ValueError("weighting is defined for hard BCE and B² only")
        if self.param not in ("alpha", "ms") or (self.param == "ms" and (self.kind != "b2" or self.plus_one)):
            raise ValueError("param 'ms' is a B²-r head option (not B²-M, not BCE)")
        if self.s_shrink and self.param != "ms":
            raise ValueError("s_shrink needs the mean-spread head")
        if self.n_weight and self.kind != "soft_bce":
            raise ValueError("n_weight is a soft-BCE option")
        if self.s_shrink_target not in ("prior", "batch"):
            raise ValueError(f"unknown s_shrink_target {self.s_shrink_target}")

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
        if (cfg.kl_lambda or (cfg.s_shrink and cfg.s_shrink_target == "prior")) and prior is None:
            raise ValueError("KL and spread shrinkage need a prior")
        self.register_buffer("weights", None if weights is None else torch.as_tensor(weights, dtype=torch.float32))
        self.register_buffer("prior", None if prior is None else torch.as_tensor(prior, dtype=torch.float32))

    def per_sample(self, out: torch.Tensor, k: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg.kind == "b2":
            # float64: at large S the log terms differ by tiny amounts that float32 loses
            if cfg.param == "ms":
                a0, a1, log_s = ms_alphas(out.double())
            else:
                a0, a1 = alphas(out.double(), cfg.eps, cfg.plus_one)
            loss = beta_binomial_nll(a0, a1, k.double(), n.double())
            if cfg.s_shrink:
                if cfg.s_shrink_target == "batch":
                    log_s0 = log_s.mean(0, keepdim=True)     # not detached: the penalty is λ × batch variance of log S
                else:
                    log_s0 = torch.log(self.prior[:, 0] + self.prior[:, 1]).double()
                loss = loss + cfg.s_shrink * (log_s - log_s0) ** 2
        else:
            z = out[..., 0]
            target = {"hard_bce": lambda: majority(k, n),
                      "soft_bce": lambda: k / n,
                      "dup": lambda: split_target(k, n)}[cfg.kind]()
            loss = F.binary_cross_entropy_with_logits(z, target, reduction="none")
        if cfg.n_weight:
            loss = loss * (n / n.mean(0, keepdim=True))
        if cfg.weighted:
            y = majority(k, n)
            loss = loss * (y * self.weights[:, 1] + (1 - y) * self.weights[:, 0])
        if cfg.kl_lambda:
            loss = loss + cfg.kl_lambda * beta_kl(a0, a1, self.prior[:, 0].double(), self.prior[:, 1].double())
        return loss.to(out.dtype)                       # [B, K]

    def forward(self, out, k, n):
        return self.per_sample(out, k, n).mean(0).sum()
