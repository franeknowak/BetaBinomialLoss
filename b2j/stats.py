"""Module e: two-level bootstrap and readings (plan §4.3, §5.1; no equivalence margins, 05_plan amendment 2026-10-16 #1).

Bootstrap (2,000 resamples, 95% percentile CI):
  level 1: test clusters, resampled with replacement and shared by every model in the comparison
           (videos for Endoscapes, images for MHIST, images within class for CIFAR-10H);
  level 2: seeds, resampled independently per model (or jointly within a `paired` group, e.g. two scores
           of the same runs); the model's value is the per-seed metric averaged over the drawn seeds.
Validation-fitted quantities (thresholds, c, Global-S) are inputs and are never refitted.
The metric may return an array (e.g. precision on a recall grid): the CI is then pointwise.
See journal_plan/specs/e_metrics_stats.md.
"""
from dataclasses import dataclass

import numpy as np

B_DEFAULT = 2000


def cluster_indices(clusters, strata=None):
    """Row indices grouped by cluster, and (optionally) clusters grouped by stratum."""
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    rows = [np.flatnonzero(inv == c) for c in range(len(uniq))]
    if strata is None:
        return rows, [np.arange(len(uniq))]
    strata = np.asarray(strata)
    first = np.array([strata[r[0]] for r in rows])
    if any((strata[r] != strata[r[0]]).any() for r in rows):
        raise ValueError("a cluster spans several strata")
    return rows, [np.flatnonzero(first == s) for s in np.unique(first)]


def resample_rows(rng, rows, strata_groups):
    """One level-1 draw: clusters with replacement within each stratum -> concatenated row indices."""
    picked = np.concatenate([rng.choice(g, len(g), replace=True) for g in strata_groups])
    return np.concatenate([rows[c] for c in picked])


@dataclass
class Result:
    estimate: float | np.ndarray
    lo: float | np.ndarray
    hi: float | np.ndarray
    samples: np.ndarray

    @property
    def half_width(self):
        return (self.hi - self.lo) / 2


def two_level_bootstrap(metric, models, combine, clusters, strata=None, paired=(), B=B_DEFAULT, seed=0, alpha=0.05):
    """CI of combine({model: value}) under the two-level bootstrap.

    metric(pred, rows) -> float evaluates one seed's predictions on the given test rows.
    models: {name: [pred_seed_1, ..., pred_seed_S]}.
    combine: {name: value} -> float (e.g. lambda v: v["b2r"] - v["soft_bce"]); values may be arrays of one shape.
    paired: groups of model names that share seed draws (same runs scored two ways); they need equal seed counts.
    The estimate uses all rows and all seeds.
    """
    rng = np.random.default_rng(seed)
    rows, groups = cluster_indices(clusters, strata)
    all_rows = np.concatenate(rows)
    group_of = {m: tuple(g) for g in paired for m in g}
    avg = lambda vs: np.mean(np.asarray(vs, np.float64), 0)
    estimate = combine({m: avg([metric(p, all_rows) for p in ps]) for m, ps in models.items()})
    out = np.empty((B,) + np.shape(estimate))
    for b in range(B):
        r = resample_rows(rng, rows, groups)
        seed_draw = {}
        vals = {}
        for m, ps in models.items():
            key = group_of.get(m, (m,))
            if key not in seed_draw:
                seed_draw[key] = rng.integers(0, len(ps), len(ps))
            vals[m] = avg([metric(ps[s], r) for s in seed_draw[key]])
        out[b] = combine(vals)
    lo, hi = np.nanpercentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    if np.ndim(estimate) == 0:
        return Result(float(estimate), float(lo), float(hi), out)
    return Result(np.asarray(estimate), lo, hi, out)


# ---------------------------------------------------------------- readings
def verdict(res: Result, kind: str, direction: int = -1) -> str:
    """Reading of a difference's CI (plan §5.1). No margins: a CI containing 0 is never read as 'equivalent'.

    kind 'superiority': direction is the predicted sign (-1: < 0, +1: > 0) -> 'success' | 'fail' | 'inconclusive'.
    kind 'difference':  direction is the sign that means the first model is better (+1 for BACC, -1 for NLL or
                        FCR) -> 'better' | 'worse' | 'no evidence of a difference'.
    """
    lo, hi = res.lo, res.hi
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or +1")
    favours = hi < 0 if direction < 0 else lo > 0          # CI excludes 0 on the predicted / better side
    against = lo > 0 if direction < 0 else hi < 0
    if kind == "superiority":
        return "success" if favours else "fail" if against else "inconclusive"
    if kind == "difference":
        return "better" if favours else "worse" if against else "no evidence of a difference"
    raise ValueError(f"unknown verdict kind {kind}")
