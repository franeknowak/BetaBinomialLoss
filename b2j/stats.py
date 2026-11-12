"""Statistics for the phase-1 tables (WRITEUP §2).

Table 1: one-tailed Welch t-test over the 5 seeds, H1 = B² better than the other loss.
Table 2: paired cluster bootstrap over test images: the same resampled clusters (Endoscapes videos, AMi-Br slides,
otherwise single images) for every loss and seed; per resample each metric is computed per seed and averaged over
the seeds; reported is the 95% percentile interval of B²'s improvement over each other loss.

Improvement is signed so that positive = B² better: higher-better metrics B² − other, lower-better other − B², spread
ratio |other − 1| − |B² − 1|.

Works on saved outputs (B²: α₀, α₁; BCE: logit), test split. Metric definitions as in b2j/metrics.py (PLAN §1.10);
resampled AP and AUROC use image weights (= how often an image is drawn), matching sklearn with sample_weight.
"""
import numpy as np
from scipy import stats
from scipy.stats import betabinom, binom

from b2j.metrics import LEVELS, central_interval, rps_over_n

METRICS = ("ap", "auroc", "mse", "vote_nll", "rps", "spread", "cov_gap")
HIGHER, LOWER = {"ap", "auroc"}, {"mse", "vote_nll", "rps", "cov_gap"}


def entries(loss, out, k, n):
    """Per (image, label) quantities from saved outputs. out [N, K, w] numpy (α₀, α₁ or logit); k, n [N, K] int."""
    j = np.arange(n.max() + 1)
    if loss == "b2":
        a0, a1 = out[..., 0], out[..., 1]
        phi, m = a0 + a1, a1 / (a0 + a1)
        d = betabinom(n[..., None], a1[..., None], a0[..., None])
        nll = -betabinom.logpmf(k, n, a1, a0)
        var = m * (1 - m) / n * (1 + (n - 1) / (phi + 1))
    else:
        m = 1 / (1 + np.exp(-out[..., 0]))
        d = binom(n[..., None], m[..., None])
        nll = -binom.logpmf(k, n, m)
        var = m * (1 - m) / n
    F = d.cdf(j)
    claimed, hit = [], []
    for X in LEVELS:
        L, U, c = central_interval(F, X)
        claimed.append(c)
        hit.append(((k >= L) & (k <= U)).astype(float))
    return {"p": m, "y": (k > n / 2), "sq": (k / n - m) ** 2, "nll": nll, "rps": rps_over_n(F, k, n), "var": var,
            "claimed": np.array(claimed), "hit": np.array(hit)}


def _groups(score, order):
    s = score[order]
    return np.flatnonzero(np.r_[True, s[1:] != s[:-1]])


def weighted_auroc(score, y, W):
    """AUROC per row of image weights W [B, N] (ties count one half)."""
    order = np.argsort(score, kind="mergesort")
    st = _groups(score, order)
    Ws, ys = W[:, order], y[order]
    Pg = np.add.reduceat(Ws * ys, st, axis=1)
    Ng = np.add.reduceat(Ws * ~ys, st, axis=1)
    below = np.cumsum(Ng, 1) - Ng
    P, N = Pg.sum(1), Ng.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Pg * (below + 0.5 * Ng)).sum(1) / (P * N)


def weighted_ap(score, y, W):
    """Average precision per row of image weights W [B, N], as sklearn (step-wise, one step per distinct score)."""
    order = np.argsort(-score, kind="mergesort")
    st = _groups(-score, order)
    Ws, ys = W[:, order], y[order]
    TP = np.cumsum(np.add.reduceat(Ws * ys, st, axis=1), 1)
    FP = np.cumsum(np.add.reduceat(Ws * ~ys, st, axis=1), 1)
    P = TP[:, -1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(TP + FP > 0, TP / np.where(TP + FP > 0, TP + FP, 1), 0.0)
        rec = TP / P
    drec = np.diff(np.c_[np.zeros(len(W)), rec], axis=1)
    return (drec * prec).sum(1)


def weighted_metrics(E, W):
    """The 7 metrics for each row of image weights W [B, N]. Macro over labels for AP, AUROC, MSE, NLL, RPS/n;
    pooled over (image, label) for the spread ratio and the coverage gap."""
    tot = W.sum(1, keepdims=True)
    K = E["p"].shape[1]
    out = {"ap": np.mean([weighted_ap(E["p"][:, j], E["y"][:, j], W) for j in range(K)], 0),
           "auroc": np.mean([weighted_auroc(E["p"][:, j], E["y"][:, j], W) for j in range(K)], 0),
           "mse": (W @ E["sq"] / tot).mean(1), "vote_nll": (W @ E["nll"] / tot).mean(1),
           "rps": (W @ E["rps"] / tot).mean(1),
           "spread": (W @ E["sq"].sum(1)) / (W @ E["var"].sum(1))}
    cl = np.stack([W @ c.sum(1) for c in E["claimed"]]) / (tot[:, 0] * K)
    ht = np.stack([W @ h.sum(1) for h in E["hit"]]) / (tot[:, 0] * K)
    out["cov_gap"] = np.abs(ht - cl).mean(0)
    return out


def cluster_weights(cluster, B, rng):
    """[B, N] image weights: draw as many clusters as there are, with replacement; an image's weight = how often its
    cluster was drawn."""
    u, inv = np.unique(cluster, return_inverse=True)
    draws = rng.integers(0, len(u), size=(B, len(u)))
    counts = np.zeros((B, len(u)))
    np.add.at(counts, (np.repeat(np.arange(B), len(u)), draws.ravel()), 1)
    return counts[:, inv]


def improvement(metric, b2, other):
    """Positive = B² better."""
    if metric in HIGHER:
        return b2 - other
    if metric in LOWER:
        return other - b2
    return np.abs(other - 1) - np.abs(b2 - 1)


def welch_one_tailed(metric, b2, other):
    """p-value of a one-tailed Welch t-test over seeds, H1 = B² better. b2, other: [seeds]. NaN if both are constant."""
    if metric == "spread":
        b2, other = np.abs(b2 - 1), np.abs(other - 1)
    if np.std(b2) == 0 and np.std(other) == 0:
        return float("nan")
    alt = "greater" if metric in HIGHER else "less"
    return float(stats.ttest_ind(b2, other, equal_var=False, alternative=alt).pvalue)
