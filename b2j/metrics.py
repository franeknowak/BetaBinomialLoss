"""Phase-1 metrics (PLAN §1.9).

Every loss is read as a predicted distribution over the vote count k = 0…n per image and label: B² the
Beta-Binomial(n; α₁, α₀), hard and soft BCE the Binomial(n, p = sigmoid(logit)).

Per label (macro = plain mean over labels; NaN if any label is NaN, handled at analysis):
  AP, AUROC   ranking against the majority vote 1[k > n/2] (threshold-free); NaN if the reference has one class
  MSE         (predicted fraction − k/n)²; predicted fraction = μ = α₁/(α₀ + α₁) for B², p for the BCE losses
  vote NLL    −log P(observed k of n)
  RPS/n       ranked probability score Σ_{j=0}^{n−1} (P(K ≤ j) − 1[k ≤ j])², divided by n
Pooled over all (image, label) of a split:
  spread ratio   mean observed (k/n − μ)² ÷ mean predicted Var(k/n); B² Var(k/n) = μ(1 − μ)/n · [1 + (n − 1)/(φ + 1)]
                 with φ = α₀ + α₁, BCE p(1 − p)/n. 1 = right size of spread, > 1 too narrow, < 1 too wide
  coverage       per level X = 10%, …, 90%: the central X% interval [L, U] of the predicted distribution, rounded
                 outward (L = smallest j with F(j) > a, U = smallest j with F(j) ≥ 1 − a, a = (1 − X)/2, F(j) = P(K ≤ j));
                 claimed = F(U) − F(L − 1), hit = 1[L ≤ k ≤ U]; mean claimed and hit rate per level;
                 coverage gap = mean over levels of |hit rate − mean claimed| (0 = calibrated)
"""
import numpy as np
import torch
from scipy.stats import betabinom, binom
from sklearn.metrics import average_precision_score, roc_auc_score

from b2j.losses import alphas, predicted_fraction, vote_nll

LEVELS = tuple(round(x / 10, 1) for x in range(1, 10))
PER_LABEL = ("ap", "auroc", "mse", "vote_nll", "rps", "n_ref_pos")
MACRO = ("ap", "auroc", "mse", "vote_nll", "rps")


def predictive(loss_name, out, n):
    """Predicted distribution over k for every (image, label). out: head outputs [N, K, w] (torch); n: [N, K] numpy int.
    Returns mean (μ or p) [N, K], predicted Var(k/n) [N, K] and the cumulative F(j) = P(K ≤ j) for j = 0…max n
    [N, K, max n + 1] (1 beyond an entry's own n), all float64 numpy."""
    j = np.arange(n.max() + 1)
    with torch.no_grad():
        if loss_name == "b2":
            a0, a1 = (a.cpu().numpy() for a in alphas(out.double()))
            phi, m = a0 + a1, a1 / (a0 + a1)
            F = betabinom.cdf(j, n[..., None], a1[..., None], a0[..., None])
            var = m * (1 - m) / n * (1 + (n - 1) / (phi + 1))
        else:
            m = torch.sigmoid(out[..., 0].double()).cpu().numpy()
            F = binom.cdf(j, n[..., None], m[..., None])
            var = m * (1 - m) / n
    return m, var, F


def rps_over_n(F, k, n):
    """RPS/n per entry. F: [..., J]; k, n: [...] int."""
    j = np.arange(F.shape[-1])
    return (((F - (k[..., None] <= j)) ** 2) * (j < n[..., None])).sum(-1) / n


def spread_ratio(m, var, k, n):
    """Pooled over all entries: mean observed (k/n − m)² ÷ mean predicted Var(k/n)."""
    return float(((k / n - m) ** 2).mean() / var.mean())


def central_interval(F, X):
    """Central X% interval of k, rounded outward. F: [..., J]. Returns L, U and the claimed coverage P(L ≤ K ≤ U)."""
    a = (1 - X) / 2
    L = (F > a).argmax(-1)
    U = (F >= 1 - a - 1e-9).argmax(-1)
    FU = np.take_along_axis(F, U[..., None], -1)[..., 0]
    FL = np.where(L > 0, np.take_along_axis(F, np.maximum(L - 1, 0)[..., None], -1)[..., 0], 0.0)
    return L, U, FU - FL


def coverage(F, k, levels=LEVELS):
    """Pooled over all entries: mean claimed coverage and hit rate per level, and the coverage gap."""
    claimed, hit = [], []
    for X in levels:
        L, U, c = central_interval(F, X)
        claimed.append(float(c.mean()))
        hit.append(float(((k >= L) & (k <= U)).mean()))
    gap = float(np.mean(np.abs(np.array(hit) - np.array(claimed))))
    return {"levels": list(levels), "claimed": claimed, "hit": hit, "gap": gap}


def label_metrics(p, y, frac, nll, rps):
    """One label. p: predicted fraction [N]; y: majority-vote reference {0,1} [N]; frac: k/n [N]; nll, rps: [N]."""
    both = 0 < y.sum() < len(y)
    return {"ap": float(average_precision_score(y, p)) if both else float("nan"),
            "auroc": float(roc_auc_score(y, p)) if both else float("nan"),
            "mse": float(np.mean((p - frac) ** 2)), "vote_nll": float(np.mean(nll)), "rps": float(np.mean(rps)),
            "n_ref_pos": int(y.sum())}


def compute(loss_name, out, k, n, labels):
    """Metrics for one split. out: head outputs [N, K, w] (torch); k, n: [N, K] (torch). Returns a plain dict."""
    with torch.no_grad():
        p = predicted_fraction(loss_name, out).cpu().numpy()
        nll = vote_nll(loss_name, out, k, n).cpu().numpy()
    k = np.rint(k.double().cpu().numpy()).astype(int)
    n = np.rint(n.double().cpu().numpy()).astype(int)
    m, var, F = predictive(loss_name, out, n)
    rps = rps_over_n(F, k, n)
    y, frac = (k > n / 2).astype(int), k / n
    per = {lab: label_metrics(p[:, j], y[:, j], frac[:, j], nll[:, j], rps[:, j]) for j, lab in enumerate(labels)}
    return {"per_label": per,
            "macro": {x: float(np.mean([per[lab][x] for lab in labels])) for x in MACRO},
            "pooled": {"spread_ratio": spread_ratio(m, var, k, n), "coverage": coverage(F, k)},
            "n_images": int(len(k))}


def macro_bacc(loss_name, out, k, n):
    """Validation BACC (macro over labels, cut-off 0.5). Training-curve diagnostic only: logged at every evaluation
    and a non-finite value stops the run; it chooses nothing and is not a reported metric."""
    with torch.no_grad():
        p = predicted_fraction(loss_name, out).cpu().numpy()
    k, n = k.double().cpu().numpy(), n.double().cpu().numpy()
    y, pred = k > n / 2, p > 0.5
    sens = (pred & y).sum(0) / y.sum(0)
    spec = (~pred & ~y).sum(0) / (~y).sum(0)
    return float(np.mean((sens + spec) / 2))
