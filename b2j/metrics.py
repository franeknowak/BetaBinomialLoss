"""Module e: per-run metrics on one evaluation set (plan §4.3). Arrays are [N, K] (one column per head).

Targets come from the evaluation panel (k, n): majority y = 1[k > n/2]; the 3-rater split target is
1[0 < k < 3] at n = 3 and its unbiased estimate (losses.split_target) for larger panels. Soft (fractional)
split targets enter AUROC/AP as weighted positives and negatives (weight t and 1 - t).
See journal_plan/specs/e_metrics_stats.md.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from b2j import posthoc


def majority(k, n):
    return np.asarray(k) * 2 > np.asarray(n)


def split_target(k, n):
    """P(a random 3-rater sub-panel is not unanimous); = 1[0 < k < 3] at n = 3. Requires n >= 3."""
    k, n = np.asarray(k, np.float64), np.asarray(n, np.float64)
    if (n < 3).any():
        raise ValueError("split target needs n >= 3")
    m = n - k
    return 1 - (k * (k - 1) * (k - 2) + m * (m - 1) * (m - 2)) / (n * (n - 1) * (n - 2))


def _heads(f, *arrays):
    """Apply a per-head metric f(col_1, col_2, ...) -> float to every head; mean over heads with a defined value."""
    vals = [f(*(a[:, j] for a in arrays)) for j in range(arrays[0].shape[1])]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _weighted(score, target):
    """Soft binary targets -> duplicated rows (positive with weight t, negative with weight 1 - t)."""
    t = np.asarray(target, np.float64)
    return np.concatenate([score, score]), np.concatenate([np.ones_like(t), np.zeros_like(t)]), np.concatenate([t, 1 - t])


def auroc(score, target):
    s, y, w = _weighted(score, target)
    if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
        return None
    return roc_auc_score(y, s, sample_weight=w)


def ap(score, target):
    s, y, w = _weighted(score, target)
    if w[y == 1].sum() == 0:
        return None
    return average_precision_score(y, s, sample_weight=w)


# ---------------------------------------------------------------- discrimination
def discrimination(score, y, threshold):
    """ACC, BACC, F1 at per-head thresholds, plus mAP and mean AUROC. score, y [N, K]; threshold [K]."""
    y = np.asarray(y, bool)
    pred = score >= np.asarray(threshold)[None, :]

    def bacc(p, t):
        if t.all() or (~t).all():
            return None
        return 0.5 * (p[t].mean() + (~p[~t]).mean())

    def f1(p, t):
        tp = (p & t).sum()
        return 2 * tp / (p.sum() + t.sum()) if (p.sum() + t.sum()) else None

    return {"acc": float((pred == y).mean()), "bacc": _heads(bacc, pred, y), "f1": _heads(f1, pred, y),
            "map": _heads(ap, score, y.astype(float)), "auroc": _heads(auroc, score, y.astype(float))}


# ---------------------------------------------------------------- panel prediction
def vote_nll(m, S, k, n):
    """Mean vote NLL per frame-criterion and per vote (NLL / n)."""
    nll = posthoc.vote_nll(m, S, k, n)
    return {"nll": float(nll.mean()), "nll_per_vote": float((nll / np.asarray(n)).mean())}


def ece(p, y, bins=10):
    """Expected calibration error of a probability p for binary y, 10 equal-mass bins per head, mean over heads."""
    def one(pj, yj):
        idx = np.array_split(np.argsort(pj, kind="stable"), bins)
        return sum(len(i) * abs(pj[i].mean() - yj[i].mean()) for i in idx if len(i)) / len(pj)
    return _heads(one, np.asarray(p, np.float64), np.asarray(y, np.float64))


def brier(p, y):
    return float(((np.asarray(p, np.float64) - np.asarray(y, np.float64)) ** 2).mean())


def theta_mse(m, k, n):
    """Mean squared error of m against the full panel's vote fraction (CIFAR-10H, MHIST)."""
    return float(((np.asarray(m) - np.asarray(k) / np.asarray(n)) ** 2).mean())


# ---------------------------------------------------------------- use-case endpoints
def fcr_at_recall(score, y, recall=0.90):
    """False-confirmation rate = 1 - precision at the first operating point (descending score) whose recall
    reaches `recall`, read off the PR curve; tied scores enter together. Mean over heads."""
    def one(s, t):
        if t.sum() == 0:
            return None
        u, inv = np.unique(-s, return_inverse=True)            # distinct thresholds, descending score
        tp = np.cumsum(np.bincount(inv, weights=t, minlength=len(u)))
        npred = np.cumsum(np.bincount(inv, minlength=len(u)))
        i = int(np.argmax(tp / t.sum() >= recall - 1e-12))
        return 1 - tp[i] / npred[i]
    return _heads(one, np.asarray(score, np.float64), np.asarray(y, bool))


def fcr_at_threshold(score, y, threshold):
    """FCR (1 - precision) and the realised recall at fixed per-head thresholds (validation-chosen)."""
    y = np.asarray(y, bool)
    pred = score >= np.asarray(threshold)[None, :]
    fcr = _heads(lambda p, t: 1 - (p & t).sum() / p.sum() if p.sum() else None, pred, y)
    rec = _heads(lambda p, t: (p & t).sum() / t.sum() if t.sum() else None, pred, y)
    return {"fcr": fcr, "recall": rec}


def aurc(p_maj, y):
    """Area under the risk-coverage curve: confidence max(P_maj, 1 - P_maj), risk = decision (P_maj >= 0.5)
    disagrees with the majority label. Mean selective risk over all coverage levels; mean over heads."""
    def one(p, t):
        conf = np.maximum(p, 1 - p)
        err = ((p >= 0.5) != t)[np.argsort(-conf, kind="stable")]
        return float((np.cumsum(err) / np.arange(1, len(err) + 1)).mean())
    return _heads(one, np.asarray(p_maj, np.float64), np.asarray(y, bool))


def split_rate_at_coverage(p_split, target, coverage=0.80):
    """U1-c: mean split target among the `coverage` share of images with the lowest P(split); mean over heads."""
    def one(p, t):
        keep = np.argsort(p, kind="stable")[: int(round(coverage * len(p)))]
        return t[keep].mean()
    return _heads(one, np.asarray(p_split, np.float64), np.asarray(target, np.float64))


def triage(p_split, target, top=0.20):
    """U2: AP and AUROC of P(split) for the split target, and the catch rate (share of the total split mass
    found in the top 20% by P(split)). Means over heads."""
    p, t = np.asarray(p_split, np.float64), np.asarray(target, np.float64)

    def catch(pj, tj):
        if tj.sum() == 0:
            return None
        return tj[np.argsort(-pj, kind="stable")[: int(round(top * len(pj)))]].sum() / tj.sum()
    return {"ap": _heads(ap, p, t), "auroc": _heads(auroc, p, t), "catch": _heads(catch, p, t)}


def split_reliability(p_split, target, bins=10):
    """P(split) reliability (higher-order calibration): per head, 10 equal-mass bins of predicted P(split) with
    mean prediction, mean observed split target and its jackknife standard error (the noise floor).
    Returns [K, bins, 3]."""
    p, t = np.asarray(p_split, np.float64), np.asarray(target, np.float64)
    out = np.empty((p.shape[1], bins, 3))
    for j in range(p.shape[1]):
        for b, i in enumerate(np.array_split(np.argsort(p[:, j], kind="stable"), bins)):
            tb = t[i, j]
            nb = len(tb)
            loo = (tb.sum() - tb) / (nb - 1) if nb > 1 else tb
            se = np.sqrt((nb - 1) / nb * ((loo - loo.mean()) ** 2).sum()) if nb > 1 else np.nan
            out[j, b] = p[i, j].mean(), tb.mean(), se
    return out
