"""Module d: post-hoc quantities, all fitted on validation only (plan §2 "Post-hoc steps"); the spread step was
started in module c because the trainer needs it at every evaluation. See journal_plan/specs/d_posthoc.md.

Every variant's raw head output [N, K, d] maps to a panel prediction (m, S): mean vote fraction m and
Beta spread S (S = inf is the Binomial plug-in). The spread step fits one scale per head on validation
by vote likelihood:
  - B² variants (except legacy B²-M): S = c · (α₀ + α₁)   (c, plan §2)
  - soft and hard BCE:                S = S*              (Global-S, one value per head; the plug-in
                                                          S = inf is a supplementary row, module d)
  - B²-M:                             no fit (native spread, legacy row)
DUP has no panel prediction; its score is P(split).

Also here: S*(m) (a spread that depends only on m), the panel probabilities P_maj, P(split), P(unanimous)
from the vote pmf, the soft-BCE ensemble mixture, and the validation thresholds (max BACC; recall 0.90).
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, gammaln, logit

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


# ---------------------------------------------------------------- S*(m): spread as a function of m only
def fit_s_of_m(m, k, n, bins=5):
    """Per head: 5 equal-mass bins of m (by rank), MLE of S per bin, knots at each bin's median logit m.

    Returns (knots [K, bins], S [K, bins]). S*(m) is piecewise linear in logit m (plan §2), flat outside
    the outermost knots. Interpolating S itself (not log S) follows the plan and matters: on the plan's
    steep example (S 1 -> 30 between m 0.15 and 0.3) log-S interpolation would hide the non-monotone
    P_maj that R3 must check (pmaj_monotone). A bin whose MLE is the Binomial limit sits at the search
    bound e^9 (fit_scale).
    """
    m, k, n = (np.asarray(x, np.float64) for x in (m, k, n))
    K = m.shape[1]
    knots, S = np.empty((K, bins)), np.empty((K, bins))
    for j in range(K):
        for b, idx in enumerate(np.array_split(np.argsort(m[:, j], kind="stable"), bins)):
            mb = m[idx, j:j + 1]
            knots[j, b] = np.median(logit(np.clip(mb, 1e-12, 1 - 1e-12)))
            S[j, b] = fit_scale(mb, np.ones_like(mb), k[idx, j:j + 1], n[idx, j:j + 1])[0]
    return knots, S


def s_of_m(fit, m):
    """Evaluate a fitted S*(m) at m [N, K]."""
    knots, S = fit
    x = logit(np.clip(np.asarray(m, np.float64), 1e-12, 1 - 1e-12))
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        if np.isinf(S[j]).any():             # n = 1 everywhere: tie-break S = inf
            out[:, j] = np.inf
            continue
        order = np.argsort(knots[j])
        out[:, j] = np.interp(x[:, j], knots[j][order], S[j][order])
    return out


# ---------------------------------------------------------------- vote pmf and panel probabilities
def vote_pmf(m, S, n):
    """P(k | n, m, S) for k = 0..max(n): [N, K, max n + 1]; entries with k > n are 0. n may vary per sample."""
    m, S, n = np.broadcast_arrays(*(np.asarray(x, np.float64) for x in (m, S, n)))
    kmax = int(n.max())
    ks = np.arange(kmax + 1)
    shape = m.shape + (kmax + 1,)
    kk = np.broadcast_to(ks, shape)
    valid = kk <= n[..., None]
    nll = vote_nll(m[..., None], S[..., None], np.where(valid, kk, 0), n[..., None])
    return np.where(valid, np.exp(-nll), 0.0)


def p_majority(pmf, n):
    """P(strict yes-majority, k > n/2) from a pmf [..., n_max + 1]; at even n a tie counts as 'not yes'."""
    ks = np.arange(pmf.shape[-1])
    return (pmf * (2 * ks > np.asarray(n, np.float64)[..., None])).sum(-1)


def p_unanimous(pmf, n):
    """P(k = 0) + P(k = n) from a pmf [..., n_max + 1]."""
    n = np.asarray(n).astype(int)
    return pmf[..., 0] + np.take_along_axis(pmf, n[..., None], -1)[..., 0]


def p_split3(m, S):
    """P(a 3-rater panel is not unanimous) = 3 m (1 - m) S / (S + 1)  (plan §3 point 5; S = inf: 3 m (1 - m))."""
    m, S = np.asarray(m, np.float64), np.asarray(S, np.float64)
    return 3 * m * (1 - m) / (1 + 1 / S)               # S / (S + 1), = 1 at S = inf


def agree_with_decision(p_maj, decision):
    """P(the panel majority agrees with a binary decision): P_maj if decision = 1, else 1 - P_maj."""
    return np.where(np.asarray(decision, bool), p_maj, 1 - p_maj)


def mixture_pmf(members, n):
    """Ensemble: the mean of the members' vote pmfs. members: list of (m, S) [N, K] each."""
    return np.mean([vote_pmf(m, S, n) for m, S in members], 0)


def mixture_p_split3(members):
    """Ensemble P(split) for 3 raters = mean of the members' P(split)."""
    return np.mean([p_split3(m, S) for m, S in members], 0)


# ---------------------------------------------------------------- validation thresholds
def threshold_max_bacc(score, y):
    """Per head, the threshold maximising validation BACC of (score >= t) against binary y. Returns [K].

    Candidates are midpoints between consecutive distinct scores (plus both ends). Ties: the lowest such
    threshold (favours sensitivity). A head with one class only returns 0.5.
    """
    score, y = np.asarray(score, np.float64), np.asarray(y, bool)
    out = np.empty(score.shape[1])
    for j in range(score.shape[1]):
        s, t = score[:, j], y[:, j]
        P, N = t.sum(), (~t).sum()
        if P == 0 or N == 0:
            out[j] = 0.5
            continue
        u = np.unique(s)
        cand = np.concatenate([[u[0] - 1e-12], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-12]])
        order = np.argsort(s, kind="stable")
        s_sorted, t_sorted = s[order], t[order]
        idx = np.searchsorted(s_sorted, cand, side="left")      # number of samples below each threshold
        tp = P - np.concatenate([[0], np.cumsum(t_sorted)])[idx]
        tn = np.concatenate([[0], np.cumsum(~t_sorted)])[idx]
        bacc = 0.5 * (tp / P + tn / N)
        out[j] = cand[int(np.argmax(bacc))]
    return out


def threshold_at_recall(score, y, recall=0.90):
    """Per head, the largest threshold whose validation recall of (score >= t) is >= recall. Returns [K]."""
    score, y = np.asarray(score, np.float64), np.asarray(y, bool)
    out = np.empty(score.shape[1])
    for j in range(score.shape[1]):
        pos = np.sort(score[y[:, j], j])[::-1]
        if len(pos) == 0:
            out[j] = np.nan
            continue
        need = int(np.ceil(recall * len(pos)))                  # positives that must be at or above t
        out[j] = pos[need - 1]
    return out


def pmaj_monotone(fit, n, grid=2001):
    """Plan §3 point 2: is P_maj(m, S*(m)) non-decreasing in m on (0, 1)? Per head [K] bool (checked on a grid)."""
    m = np.linspace(1e-4, 1 - 1e-4, grid)[:, None].repeat(fit[0].shape[0], 1)
    pm = p_majority(vote_pmf(m, s_of_m(fit, m), np.full(m.shape, n)), np.full(m.shape, n))
    return (np.diff(pm, axis=0) >= -1e-12).all(0)
