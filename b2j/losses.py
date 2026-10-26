"""The Beta-Binomial (B²) formulation for K independent binary heads.

Each head outputs two logits per image; softplus + eps maps them to (alpha_0, alpha_1), the parameters of
θ ~ Beta(alpha_1, alpha_0), and the k positive votes out of n are k ~ Binomial(n, θ). The loss is the
Beta-Binomial NLL of the observed (k, n).
Shapes: k, n are float tensors [B, K]; head outputs are [B, K, 2] (logit for alpha_0, logit for alpha_1).
Every function works on whatever device its inputs live on (no CPU round trip except lgamma/digamma, see _on_cpu).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def alphas(logits: torch.Tensor, eps: float = 1e-3):
    """[..., 2] logits -> (a0, a1) = softplus(logits) + eps."""
    a = F.softplus(logits) + eps
    return a[..., 0], a[..., 1]


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


class B2Loss(nn.Module):
    """Sum over heads of the batch-mean Beta-Binomial NLL. Computed in float64: at large S the log terms differ by
    tiny amounts that float32 loses."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def per_sample(self, out: torch.Tensor, k: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        a0, a1 = alphas(out.double(), self.eps)
        return beta_binomial_nll(a0, a1, k.double(), n.double()).to(out.dtype)     # [B, K]

    def forward(self, out, k, n):
        return self.per_sample(out, k, n).mean(0).sum()
