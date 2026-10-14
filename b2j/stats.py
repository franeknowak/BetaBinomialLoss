"""Module e: two-level bootstrap, margins and three-way verdicts (plan §4.3, §5.1).

Bootstrap (2,000 resamples, 95% percentile CI):
  level 1: test clusters, resampled with replacement and shared by every model in the comparison
           (videos for Endoscapes, images for MHIST, images within class for CIFAR-10H);
  level 2: seeds, resampled independently per model (or jointly within a `paired` group, e.g. two scores
           of the same runs); the model's value is the per-seed metric averaged over the drawn seeds.
Validation-fitted quantities (thresholds, c, S*, S*(m)) are inputs and are never refitted.
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
    estimate: float
    lo: float
    hi: float
    samples: np.ndarray

    @property
    def half_width(self):
        return (self.hi - self.lo) / 2


def two_level_bootstrap(metric, models, combine, clusters, strata=None, paired=(), B=B_DEFAULT, seed=0, alpha=0.05):
    """CI of combine({model: value}) under the two-level bootstrap.

    metric(pred, rows) -> float evaluates one seed's predictions on the given test rows.
    models: {name: [pred_seed_1, ..., pred_seed_S]}.
    combine: {name: value} -> float (e.g. lambda v: v["b2r"] - v["soft_bce"]).
    paired: groups of model names that share seed draws (same runs scored two ways); they need equal seed counts.
    The estimate uses all rows and all seeds.
    """
    rng = np.random.default_rng(seed)
    rows, groups = cluster_indices(clusters, strata)
    all_rows = np.concatenate(rows)
    group_of = {m: tuple(g) for g in paired for m in g}
    estimate = combine({m: float(np.mean([metric(p, all_rows) for p in ps])) for m, ps in models.items()})
    out = np.empty(B)
    for b in range(B):
        r = resample_rows(rng, rows, groups)
        seed_draw = {}
        vals = {}
        for m, ps in models.items():
            key = group_of.get(m, (m,))
            if key not in seed_draw:
                seed_draw[key] = rng.integers(0, len(ps), len(ps))
            vals[m] = float(np.mean([metric(ps[s], r) for s in seed_draw[key]]))
        out[b] = combine(vals)
    lo, hi = np.nanpercentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Result(estimate, float(lo), float(hi), out)


# ---------------------------------------------------------------- margins and verdicts
CAPS = {"bacc": 0.03, "map": 0.03, "fcr": 0.05, "ap": 0.05}     # vote NLL: 1% of the soft-BCE plug-in validation NLL


def margin(pilot: Result, cap: float):
    """δ = 2h from the pilot CI half-width h (plan §4.3). Above the cap -> None ("CI reported, no claim")."""
    d = 2 * pilot.half_width
    return d if d <= cap else None


def verdict(res: Result, kind: str, delta: float | None = None, direction: int = -1) -> str:
    """Three-way verdict (plan §5.1): 'success' | 'fail' | 'inconclusive' | 'no claim'.

    kind: 'superiority' (direction -1: predicted < 0; +1: predicted > 0), 'equivalence' (needs delta),
    'noninferiority' (success if hi < delta; fail if the CI excludes 0 against the prediction, i.e. lo > 0).
    A margin of None (above its cap) gives 'no claim'.
    """
    lo, hi = res.lo, res.hi
    if kind == "superiority":
        if direction < 0:
            return "success" if hi < 0 else "fail" if lo > 0 else "inconclusive"
        return "success" if lo > 0 else "fail" if hi < 0 else "inconclusive"
    if delta is None:
        return "no claim"
    if kind == "equivalence":
        if -delta <= lo and hi <= delta:
            return "success"
        return "fail" if (lo > delta or hi < -delta) else "inconclusive"
    if kind == "noninferiority":
        return "success" if hi < delta else "fail" if lo > 0 else "inconclusive"
    raise ValueError(f"unknown verdict kind {kind}")
