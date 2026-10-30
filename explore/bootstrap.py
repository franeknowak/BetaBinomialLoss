"""Two-level bootstrap of the key B² comparisons: resample test clusters (Endoscapes videos, AMi-Br slides, otherwise
images) with replacement, and the 5 seeds with replacement; 2,000 resamples, 95% percentile intervals.

  python -m explore.bootstrap <dataset> [--m 3]     main final runs, or the runs trained with m raters per image
                                                    -> results/explore/bootstrap/<dataset>[_m<m>].json

Per resample, each metric is computed per seed on the resampled test set (images weighted by how often their cluster
was drawn), averaged over the drawn seeds, then over labels (labels where it is undefined are skipped). Differences
use the same clusters and seed draws for both methods. Comparisons (checkpoint pairs "own" and "bacc", as E1a/E1b):
  disagree_auroc   disputed (0 < k < n) vs unanimous; score = P(disagree): B² Beta-Binomial, B² mean only
                   (Binomial on the mean), soft / hard BCE Binomial on p (as E2, no cut-off)
  majority_auroc   majority vote; score = B²-B1, soft p, hard p (as E1)
  majority_bacc    the same with each seed's E1 cut-off fitted on validation (fixed during the bootstrap)
  vote_nll         -log P(k | n) per image: B² Beta-Binomial vs soft BCE Binomial
"""
import argparse
import json

import numpy as np
from scipy import special, stats

from b2j.data import load_table
from b2j.e1 import RESULTS, SEEDS, fit_cutoff, load_outputs, score
from b2j.e2 import p_disagree
from explore import EXPLORE, sub_root

B = 2000
EPS = 1e-9
CKPT = {"own": {"b2": "nll", "soft_bce": "softbce", "hard_bce": "hardbce"},
        "bacc": {"b2": "bacc", "soft_bce": "bacc", "hard_bce": "bacc"}}
COMPARE = {"disagree_auroc": [("b2", "soft"), ("b2", "hard"), ("b2", "b2_mean"), ("b2_mean", "soft")],
           "majority_auroc": [("b2", "soft"), ("b2", "hard")],
           "majority_bacc": [("b2", "soft"), ("b2", "hard")],
           "vote_nll": [("soft", "b2")]}                                    # positive = B² lower (better)


def weighted_auroc(score_, y, W):
    """AUROC for every row of weights W [B, N] (ties count half)."""
    order = np.argsort(score_, kind="stable")
    s, yy, W = score_[order], y[order].astype(bool), W[:, order]
    first = np.searchsorted(s, s, side="left")
    last = np.searchsorted(s, s, side="right") - 1
    neg = np.cumsum(W * ~yy, axis=1)
    below = np.where(first > 0, neg[:, np.maximum(first - 1, 0)], 0.0)
    tie = neg[:, last] - below
    pos_w = W * yy
    num = (pos_w * (below + 0.5 * tie)).sum(1)
    den = pos_w.sum(1) * (W * ~yy).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def weighted_bacc(pred, y, W):
    yb = y.astype(bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        sens = (W * (pred & yb)).sum(1) / (W * yb).sum(1)
        spec = (W * (~pred & ~yb)).sum(1) / (W * ~yb).sum(1)
    return (sens + spec) / 2


def per_seed_inputs(dataset, root, seed, ck):
    """Per label: dict of test arrays (scores, truth, log-likelihoods, E1 calls)."""
    out, labels, ids = {}, None, None
    for loss in ("b2", "soft_bce", "hard_bce"):
        data, labels = load_outputs(dataset, loss, seed, ck[loss], root)
        out[loss] = data
    for loss in ("b2", "soft_bce", "hard_bce"):                            # same frames, same order
        if not np.array_equal(out[loss]["test"][1], out["b2"]["test"][1]):
            raise ValueError(f"{loss}: test votes differ from b2")
    res = []
    for j in range(len(labels)):
        (bv, kv, nv), (bt, kt, nt) = ((o[:, j], k[:, j], n[:, j]) for o, k, n in (out["b2"]["val"], out["b2"]["test"]))
        sv, st = out["soft_bce"]["val"][0][:, j], out["soft_bce"]["test"][0][:, j]
        hv, ht = out["hard_bce"]["val"][0][:, j], out["hard_bce"]["test"][0][:, j]
        m = bt[:, 1] / bt.sum(1)
        lm = (np.log(np.clip(m, EPS, 1)) - np.log(np.clip(1 - m, EPS, 1)))[:, None]
        yv, y = (kv > nv / 2).astype(int), (kt > nt / 2).astype(int)
        calls = {}
        for name, method, ov, ot in (("b2", "b2_b1", bv, bt), ("soft", "soft_bce", sv, st), ("hard", "hard_bce", hv, ht)):
            c, _, _ = fit_cutoff(score(method, ov, nv), yv)
            calls[name] = (score(method, ot, nt), score(method, ot, nt) >= c)
        res.append({"disputed": ((kt > 0) & (kt < nt)).astype(int), "y": y,
                    "disagree": {"b2": p_disagree("b2", bt, nt), "b2_mean": p_disagree("soft_bce", lm, nt),
                                 "soft": p_disagree("soft_bce", st, nt), "hard": p_disagree("soft_bce", ht, nt)},
                    "majority": {k: v[0] for k, v in calls.items()}, "call": {k: v[1] for k, v in calls.items()},
                    "nll": {"b2": -stats.betabinom.logpmf(kt, nt, bt[:, 1], bt[:, 0]),
                            "soft": -stats.binom.logpmf(kt, nt, special.expit(st[:, 0]))}})
    return res, labels


def metric_rows(inp, W):
    """{metric: {method: [B] macro value}} for one seed under weights W."""
    acc = {"disagree_auroc": {}, "majority_auroc": {}, "majority_bacc": {}, "vote_nll": {}}
    for lab in inp:
        for meth, sc in lab["disagree"].items():
            acc["disagree_auroc"].setdefault(meth, []).append(weighted_auroc(sc, lab["disputed"], W))
        for meth, sc in lab["majority"].items():
            acc["majority_auroc"].setdefault(meth, []).append(weighted_auroc(sc, lab["y"], W))
            acc["majority_bacc"].setdefault(meth, []).append(weighted_bacc(lab["call"][meth], lab["y"], W))
        for meth, ll in lab["nll"].items():
            acc["vote_nll"].setdefault(meth, []).append((W * ll).sum(1) / W.sum(1))
    with np.errstate(invalid="ignore"):
        return {k: {m: np.nanmean(np.stack(v), 0) for m, v in d.items()} for k, d in acc.items()}


def run(dataset, root, b=B, rng_seed=0):
    table = load_table(dataset)
    cluster_of = dict(zip(table.ids, table.cluster))
    res = {}
    for pair, ck in CKPT.items():
        inputs, ids = [], None
        for seed in SEEDS:
            inp, labels = per_seed_inputs(dataset, root, seed, ck)
            inputs.append(inp)
        test_ids = [i["id"] for i in json.loads((root / "final" / dataset / "b2" / f"seed{SEEDS[0]}" /
                                                 f"outputs_{ck['b2']}.json").read_text())["images"] if i["split"] == "test"]
        cl = np.array([cluster_of[i] for i in test_ids])
        uniq, inv = np.unique(cl, return_inverse=True)
        rng = np.random.default_rng(rng_seed)
        counts = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(b)])
        W = np.vstack([np.ones(len(cl)), counts[:, inv]]).astype(np.float64)   # row 0 = the observed test set
        seed_draw = np.vstack([np.arange(len(SEEDS)), rng.integers(0, len(SEEDS), (b, len(SEEDS)))])
        per = [metric_rows(inp, W) for inp in inputs]                     # per seed: {metric: {method: [b+1]}}
        out = {}
        for metric, pairs in COMPARE.items():
            vals = {meth: np.stack([p[metric][meth] for p in per]) for meth in per[0][metric]}   # [seeds, b+1]
            avg = {meth: np.take_along_axis(v, seed_draw.T, 0).mean(0) for meth, v in vals.items()}  # [b+1]
            for a, c in pairs:
                d = avg[a] - avg[c]
                boot = d[1:][~np.isnan(d[1:])]
                out[f"{metric}|{a}-{c}"] = {"estimate": float(d[0]), "ci95": [float(np.percentile(boot, 2.5)),
                                                                                float(np.percentile(boot, 97.5))],
                                            "share_le_0": float((boot <= 0).mean()), "n_boot": int(len(boot)),
                                            "values": {a: float(avg[a][0]), c: float(avg[c][0])}}
        res[pair] = {"n_clusters": int(len(uniq)), "n_test": int(len(cl)), "comparisons": out}
    return res


def main(dataset, m=None, b=B):
    root = RESULTS if m is None else sub_root(m)
    res = run(dataset, root, b)
    d = EXPLORE / "bootstrap"
    d.mkdir(parents=True, exist_ok=True)
    name = dataset if m is None else f"{dataset}_m{m}"
    (d / f"{name}.json").write_text(json.dumps({"dataset": dataset, "m": m, "B": b, "results": res}, indent=1))
    for pair, r in res.items():
        print(f"\n{name} · {pair} checkpoints · {r['n_clusters']} clusters, {r['n_test']} test images")
        for k, v in r["comparisons"].items():
            star = "*" if v["ci95"][0] > 0 or v["ci95"][1] < 0 else " "
            print(f"  {k:28} {v['estimate']:+.4f}  [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]{star}  "
                  + "  ".join(f"{a} {x:.4f}" for a, x in v["values"].items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--b", type=int, default=B)
    a = ap.parse_args()
    main(a.dataset, a.m, a.b)
