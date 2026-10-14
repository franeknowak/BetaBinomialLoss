"""Post-hoc quantities on validation (module d; started in module c because the trainer needs them every epoch).

Every variant's raw head output [N, K, d] maps to a panel prediction (m, S): mean vote fraction m and
Beta spread S (S = inf is the Binomial plug-in). The spread step fits one scale per head on validation
by vote likelihood:
  - B² variants (except legacy B²-M): S = c · (α₀ + α₁)   (c, plan §2)
  - soft and hard BCE:                S = S*              (Global-S, one value per head; the plug-in
                                                          S = inf is a supplementary row, module d)
  - B²-M:                             no fit (native spread, legacy row)
DUP has no panel prediction; its score is P(split).
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, gammaln

from b2j.variants import VARIANTS

LOG_SCALE_BOUNDS = (-9.0, 9.0)


def spread_rule(name: str) -> str:
    """'c' (scale the B² spread) | 'global' (fit S* on a plug-in mean) | 'none' | 'dup'."""
    cfg = VARIANTS[name].loss
    if cfg.kind == "b2":
        return "none" if cfg.plus_one else "c"
    return {"soft_bce": "global", "hard_bce": "global", "dup": "dup"}[cfg.kind]


def mean_spread(name: str, out: np.ndarray, eps: float | None = None):
    """Raw head output [N, K, d] -> (m, S0) [N, K]. S0 is the unscaled spread (inf for BCE heads).

    eps: the run's B²-r positivity override (R0 fallback), as in variants.build; never applied to B²-M.
    """
    cfg = VARIANTS[name].loss
    out = np.asarray(out, dtype=np.float64)
    if cfg.kind == "b2":
        e = eps if (eps is not None and not cfg.plus_one) else cfg.eps
        a = np.logaddexp(0.0, out) + e + (1.0 if cfg.plus_one else 0.0)
        a0, a1 = a[..., 0], a[..., 1]
        return a1 / (a0 + a1), a0 + a1
    p = expit(out[..., 0])
    return p, np.full_like(p, np.inf)


def vote_nll(m, S, k, n):
    """Beta-Binomial NLL of k | n under Beta(m S, (1 - m) S), elementwise; S = inf gives the Binomial.

    Votes are integers, so B(k + a1, n - k + a0) / B(a1, a0) is evaluated as finite products
    Π_{j<k}(a1 + j) Π_{j<n-k}(a0 + j) / Π_{j<n}(S + j): exact at any S (the lgamma difference
    loses ~6 digits at S ~ 1e9).
    """
    m = np.clip(np.asarray(m, np.float64), 1e-12, 1 - 1e-12)
    S, k, n = (np.asarray(x, np.float64) for x in (S, k, n))
    S, m, k, n = (np.array(x) for x in np.broadcast_arrays(S, m, k, n))
    log_comb = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    fin = np.isfinite(S)
    Sf = np.where(fin, S, 1.0)
    a1, a0 = m * Sf, (1 - m) * Sf
    ll = np.zeros_like(m)
    for j in range(int(n.max()) if n.size else 0):
        ll += np.where(j < k, np.log(np.where(fin, a1 + j, m)), 0.0)
        ll += np.where(j < n - k, np.log(np.where(fin, a0 + j, 1 - m)), 0.0)
        ll -= np.where((j < n) & fin, np.log(Sf + j), 0.0)
    return -(log_comb + ll)


def fit_scale(m, S0, k, n):
    """Per head j, the scale s_j minimising Σ_i NLL(k | n, m, s_j · S0) with S0 = 1 for a global S*. Returns [K].

    The search is on log s within LOG_SCALE_BOUNDS. Heads with n = 1 everywhere carry no information on S:
    tie-break s = inf (plan §3, H4 anchor).
    """
    m, S0, k, n = (np.asarray(x, np.float64) for x in (m, S0, k, n))
    out = np.empty(m.shape[1])
    for j in range(m.shape[1]):
        if (n[:, j] <= 1).all():
            out[j] = np.inf
            continue
        f = lambda t: vote_nll(m[:, j], np.exp(t) * S0[:, j], k[:, j], n[:, j]).sum()
        out[j] = np.exp(minimize_scalar(f, bounds=LOG_SCALE_BOUNDS, method="bounded",
                                        options={"xatol": 1e-4}).x)
    return out


def panel(name: str, out, scale=None, eps=None):
    """(m, S) after the spread step with a given per-head scale [K] (None for rules 'none' / 'dup')."""
    m, S0 = mean_spread(name, out, eps)
    rule = spread_rule(name)
    if rule == "c":
        return m, S0 * scale
    if rule == "global":
        return m, np.broadcast_to(scale, m.shape).copy()
    return m, S0


def fit_spread(name: str, out, k, n, eps=None):
    """Fit the variant's spread scale on (validation) labels. Returns [K] or None."""
    rule = spread_rule(name)
    if rule in ("none", "dup"):
        return None
    m, S0 = mean_spread(name, out, eps)
    return fit_scale(m, S0 if rule == "c" else np.ones_like(m), k, n)
