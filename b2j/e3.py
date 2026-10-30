"""Phase 2, E3: setting aside the frames each model is least sure about (risk–coverage) (PLAN §1.12.1).

  python -m b2j.e3 <dataset>        reads results/final*/<dataset>/..., writes results/phase2/<dataset>/e3.json

Decisions are those of E1: each method's E1 score against its E1 cut-off c (fitted on all validation frames), so a
kept frame gets the same call at every coverage. Only the order in which frames are set aside comes from E3.
Each method has q = its probability that the majority of n raters says yes, and c_q = the cut-off on that scale:
  hard_bce             q = p = sigmoid(logit)                                  c_q = c
  soft_bce             q = P(k > n/2) under Binomial(n, p)                     c_q = the same map applied to c
  b2_b0                q = P(k > n/2) under Binomial(n, m), m = a1 / (a0 + a1) c_q = the same map applied to c
  b2_b1, b2_b2         q = the E1 score                                        c_q = c
Risk = the distance from the cut-off divided by the room on that side, taken from 1:
  called yes (q >= c_q): (1 - q) / (1 - c_q)        called no: q / c_q        (1 at the cut-off, 0 at q = 1 or 0)
b2_b3: decision as b2_b0 (mean against b2_b0's E1 cut-off c); risk = the mass of Beta(a1, a0) on the other side of c,
  P(theta < c) if called yes, P(theta > c) if called no.
q and 1 - q are both computed directly (no 1 - q subtraction), so confident frames keep distinct risks.
At each coverage, the round(coverage * N) test frames with the lowest risk are kept (ties at the boundary broken by
frame order; their number is recorded). Reported on the kept frames: BACC, sensitivity, specificity, kept positives
(majority yes); on the set-aside frames: the share that is disputed (0 < k < n). Plus the mean BACC over coverage
50–100% in 1% steps (trapezoid rule; the area under the curve divided by its width).
Checkpoint sets: E3a = best val BACC, E3b = own loss (same mapping as E1a / E1b).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import special, stats

from b2j.e1 import CKPTS as E1_CKPTS, RESULTS, SEEDS, _div, fit_cutoff, load_outputs, score

METHODS = {"hard_bce": "hard_bce", "soft_bce": "soft_bce", "b2_b0": "b2", "b2_b1": "b2", "b2_b2": "b2",
           "b2_b3": "b2"}                                                     # -> loss
E1_SCORE = {"b2_b3": "b2_b0"}                                                 # B3 decides as B0
CKPTS = {"E3a": E1_CKPTS["E1a"], "E3b": E1_CKPTS["E1b"]}
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
AUC_GRID = np.round(np.arange(0.50, 1.0001, 0.01), 2)
METRICS = ("bacc", "sensitivity", "specificity", "n_pos_kept", "disputed_share_aside")


def _binom_major(p, pc, n):
    """P(k > n/2) and P(k <= n/2) under Binomial(n, p), given p and pc = 1 - p computed separately."""
    yes = np.zeros(len(n))
    no = np.zeros(len(n))
    for j in range(int(n.max()) + 1):
        m = j <= n
        logc = special.gammaln(n[m] + 1) - special.gammaln(j + 1) - special.gammaln(n[m] - j + 1)
        pj = np.exp(logc + special.xlogy(j, p[m]) + special.xlogy(n[m] - j, pc[m]))
        big = j > n[m] / 2
        yes[np.flatnonzero(m)[big]] += pj[big]
        no[np.flatnonzero(m)[~big]] += pj[~big]
    return yes, no


def q_and_cut(method, out, n, c):
    """q, 1 - q and the cut-off c_q on the q scale (per frame) for the methods ranked by scaled distance."""
    out, n = np.asarray(out, dtype=np.float64), np.asarray(n, dtype=np.float64)
    c = min(max(c, 0.0), 1.0)                                                 # E1's outer candidates lie outside [0, 1]
    if method in ("hard_bce", "soft_bce"):
        z = out[:, 0]
        p, pc = special.expit(z), special.expit(-z)
    elif method == "b2_b0":
        a0, a1 = out[:, 0], out[:, 1]
        p, pc = a1 / (a0 + a1), a0 / (a0 + a1)
    elif method == "b2_b1":
        a0, a1 = out[:, 0], out[:, 1]
        h = np.floor(n / 2)
        return stats.betabinom.sf(h, n, a1, a0), stats.betabinom.cdf(h, n, a1, a0), np.full(len(n), c)
    elif method == "b2_b2":
        a0, a1 = out[:, 0], out[:, 1]
        return stats.beta.sf(0.5, a1, a0), stats.beta.cdf(0.5, a1, a0), np.full(len(n), c)
    else:
        raise ValueError(method)
    if method == "hard_bce":
        return p, pc, np.full(len(n), c)
    q, qc = _binom_major(p, pc, n)
    cq, _ = _binom_major(np.full(len(n), c), np.full(len(n), 1 - c), n)
    return q, qc, cq


def risk(method, out, n, c):
    """(risk [N], called yes [N]); c = the method's E1 cut-off (for b2_b3 the one of b2_b0)."""
    out, n = np.asarray(out, dtype=np.float64), np.asarray(n, dtype=np.float64)
    yes = score(E1_SCORE.get(method, method), out, n) >= c                    # E1 decision
    if method == "b2_b3":
        a0, a1 = out[:, 0], out[:, 1]
        return np.where(yes, stats.beta.cdf(c, a1, a0), stats.beta.sf(c, a1, a0)), yes
    q, qc, cq = q_and_cut(method, out, n, c)
    with np.errstate(divide="ignore", invalid="ignore"):                     # a side with no room has no frames
        return np.where(yes, qc / (1 - cq), q / cq), yes


def keep_order(r):
    """Frame indices from lowest to highest risk; equal risks keep frame order."""
    return np.argsort(r, kind="stable")


def at_coverage(r, yes, y, disputed, cov):
    N = len(r)
    n_keep = int(round(cov * N))
    order = keep_order(r)
    kept, aside = order[:n_keep], order[n_keep:]
    pk, yk = yes[kept], y[kept]
    sens = _div((pk & (yk == 1)).sum(), (yk == 1).sum())
    spec = _div((~pk & (yk == 0)).sum(), (yk == 0).sum())
    n_tied = 0
    if 0 < n_keep < N:
        b = r[order[n_keep - 1]]
        if r[order[n_keep]] == b:
            n_tied = int((r == b).sum())
    return {"bacc": (sens + spec) / 2, "sensitivity": sens, "specificity": spec, "n_kept": n_keep,
            "n_pos_kept": int(yk.sum()), "disputed_share_aside": _div(disputed[aside].sum(), len(aside)),
            "n_tied_at_boundary": n_tied}


def mean_bacc(r, yes, y, disputed):
    b = np.array([at_coverage(r, yes, y, disputed, c)["bacc"] for c in AUC_GRID])
    return float(np.sum((b[1:] + b[:-1]) / 2 * np.diff(AUC_GRID)) / (AUC_GRID[-1] - AUC_GRID[0]))


def run(dataset, root=RESULTS, seeds=SEEDS):
    rows = []
    for exp, ck in CKPTS.items():
        for method, loss in METHODS.items():
            for seed in seeds:
                data, labels = load_outputs(dataset, loss, seed, ck[loss], root)
                for j, lab in enumerate(labels):
                    (ov, kv, nv), (ot, kt, nt) = ((o[:, j], k[:, j], n[:, j]) for o, k, n in
                                                  (data["val"], data["test"]))
                    e1m = E1_SCORE.get(method, method)
                    c, _, _ = fit_cutoff(score(e1m, ov, nv), (kv > nv / 2).astype(int))
                    r, yes = risk(method, ot, nt, c)
                    y = (kt > nt / 2).astype(int)
                    disputed = (kt > 0) & (kt < nt)
                    rows.append({"exp": exp, "method": method, "checkpoint": ck[loss], "seed": seed, "label": lab,
                                 "cutoff": c, "n_pos": int(y.sum()),
                                 "coverage": {f"{cov:.2f}": at_coverage(r, yes, y, disputed, cov)
                                              for cov in COVERAGES},
                                 "mean_bacc_50_100": mean_bacc(r, yes, y, disputed)})
    return rows, labels


def _ms(v):
    v = np.asarray(v, dtype=float)
    return [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")]


def summary(rows, labels):
    """mean ± sd over seeds, per exp / method / label and macro (plain mean over labels per seed)."""
    out = {}
    for exp in CKPTS:
        for method in METHODS:
            r = [x for x in rows if x["exp"] == exp and x["method"] == method]
            seeds = sorted({x["seed"] for x in r})
            for lab in list(labels) + ["macro"]:
                sel = lambda s: [x for x in r if x["seed"] == s and (lab == "macro" or x["label"] == lab)]
                f = np.mean if lab == "macro" else (lambda v: v[0])
                vals = {f"{cov:.2f}": {m: _ms([f([x["coverage"][f"{cov:.2f}"][m] for x in sel(s)]) for s in seeds])
                                       for m in METRICS} for cov in COVERAGES}
                vals["mean_bacc_50_100"] = _ms([f([x["mean_bacc_50_100"] for x in sel(s)]) for s in seeds])
                out[f"{exp}|{method}|{lab}"] = vals
    return out


def print_table(summ, labels):
    covs = [f"{c:.2f}" for c in COVERAGES]
    for exp in CKPTS:
        for m in METRICS:
            print(f"\n== {exp}: {m} on test by coverage (mean ± sd over seeds)")
            print(f"{'label':6} {'method':9} " + " ".join(f"{c:>13}" for c in covs) +
                  ("   mean BACC 50-100%" if m == "bacc" else ""))
            for lab in list(labels) + ["macro"]:
                for method in METHODS:
                    v = summ[f"{exp}|{method}|{lab}"]
                    line = " ".join(f"{v[c][m][0]:7.3f}±{v[c][m][1]:5.3f}" for c in covs)
                    if m == "bacc":
                        line += f"   {v['mean_bacc_50_100'][0]:.3f}±{v['mean_bacc_50_100'][1]:.3f}"
                    print(f"{lab:6} {method:9} {line}")


def main(dataset, root=RESULTS):
    rows, labels = run(dataset, root)
    summ = summary(rows, labels)
    d = Path(root) / "phase2" / dataset
    d.mkdir(parents=True, exist_ok=True)
    (d / "e3.json").write_text(json.dumps({"dataset": dataset, "labels": labels, "coverages": list(COVERAGES),
                                           "rows": rows, "summary": summ}, indent=1))
    print_table(summ, labels)
    ties = [(x["exp"], x["method"], x["seed"], x["label"], cov, v["n_tied_at_boundary"])
            for x in rows for cov, v in x["coverage"].items() if v["n_tied_at_boundary"] > 0]
    print(f"\n{len(rows)} rankings; {len(ties)} coverage boundaries inside a tie" +
          (f" (largest tie {max(t[-1] for t in ties)}): {ties[:20]}" if ties else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    main(ap.parse_args().dataset)
