"""Round D, D8 (journal_plan/iterations.md): B²-rba against B²-r in the 2 × 2 {rater bias no / yes} × {per-image
spread no / yes} (soft BCE, B²-r, soft BCE-ba, B²-rba). Validation only.

  python -m b2j.d8 report [block]      -> results/journal/<block>/tables_val/d8_report.md, d8_long.csv

Per run: checkpoint A for discrimination (m against the panel majority, threshold max-BACC on validation), checkpoint
B for the panel endpoints after the spread step. Panel endpoints for every variant on one footing:
  - ordered-vote NLL per vote (P1; interchangeable models: count NLL + log C(n, k));
  - count NLL per vote, P_maj ECE / Brier, split AUROC / AP / catch-20 from the count pmf of the known panel
    (rater-aware: summed over the 2^R vote patterns);
  - spread ablation (P2): per-image S against one S per label refitted on validation with m (and β) fixed.
Differences carry the two-level bootstrap 95% CI (videos / slides, seeds).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from b2j import metrics as M
from b2j import posthoc as P
from b2j import stats as St
from b2j.analysis import LABEL, guard, main_runs
from b2j.data import load_table
from b2j.runs import RESULTS
from b2j.variants import VARIANTS

LABEL = {**LABEL, "soft_bce_ba": "soft BCE-ba (Global-S)", "b2rba": "B²-rba"}
D8 = ("soft_bce", "b2r", "soft_bce_ba", "b2rba")
SPREAD = ("b2r", "b2rba")                  # the variants with a per-image spread (P2 ablation)


def _rater(v):
    return VARIANTS[v].loss.rater


def run_arrays(variant, d, val):
    """Per-row arrays [N, K] of one run on validation, plus its learned β [K, R] (None if raters are interchangeable)."""
    za, zb = np.load(d / "val_A.npz"), np.load(d / "val_B.npz")
    assert (za["ids"] == val.ids).all() and (zb["ids"] == val.ids).all()
    y = M.majority(val.k, val.n)
    mA = P.mean_spread(variant, za["out"])[0]
    m, S = P.panel(variant, zb["out"], zb["scale"])
    beta = P.rater_offsets(variant, zb["out"]) if _rater(variant) else None
    pmf = P.rater_count_pmf(m, S, beta, val.votes) if _rater(variant) else P.vote_pmf(m, S, val.n)
    r = {"mA": mA, "thr": P.threshold_max_bacc(mA, y), "m": m, "pmf": pmf,
         "onll": P.ordered_nll(m, S, beta, val.votes) / val.n,
         "cnll": -np.log(np.clip(np.take_along_axis(pmf, val.k.astype(int)[..., None], -1)[..., 0], 1e-300, None)) / val.n,
         "psplit": 1 - P.p_unanimous(pmf, val.n), "beta": beta}
    if variant in SPREAD:                  # P2: one S per label instead of the per-image S (same m, same β)
        S0 = np.ones_like(m)
        sc = P.fit_scale_rater(m, S0, beta, val.votes) if beta is not None else P.fit_scale(m, S0, val.k, val.n)
        Sc = np.broadcast_to(sc, m.shape)
        pc = P.rater_count_pmf(m, Sc, beta, val.votes) if beta is not None else P.vote_pmf(m, Sc, val.n)
        r.update({"onll_const": P.ordered_nll(m, Sc, beta, val.votes) / val.n,
                  "psplit_const": 1 - P.p_unanimous(pc, val.n)})
    return r


def _split_target(val):
    return ((val.k > 0) & (val.k < val.n)).astype(float)


def scores(r, val, rows=None):
    """Scalar endpoints of one run on the given rows (all rows if None)."""
    rows = slice(None) if rows is None else rows
    y, t = M.majority(val.k, val.n)[rows], _split_target(val)[rows]
    pmaj = P.p_majority(r["pmf"][rows], val.n[rows])
    tr = M.triage(r["psplit"][rows], t)
    out = {**M.discrimination(r["mA"][rows], y, r["thr"]), "onll": float(r["onll"][rows].mean()),
           "cnll": float(r["cnll"][rows].mean()), "ece_pmaj": M.ece(pmaj, y), "brier_pmaj": M.brier(pmaj, y),
           "split_auroc": tr["auroc"], "split_ap": tr["ap"], "split_catch20": tr["catch"]}
    if "onll_const" in r:
        out["onll_const"] = float(r["onll_const"][rows].mean())
        out["d_onll_spread"] = out["onll"] - out["onll_const"]       # < 0: the per-image S is informative
        out["split_auroc_const"] = M.triage(r["psplit_const"][rows], t)["auroc"]
    return out


def _boot(models, key, combine, clusters, paired=(), B=St.B_DEFAULT, val=None):
    y, t = M.majority(val.k, val.n).astype(float), _split_target(val)
    metric = {"map": lambda r, rows: M._heads(M.ap, r["mA"][rows], y[rows]),
              "split_ap": lambda r, rows: M._heads(M.ap, r["psplit"][rows], t[rows])}.get(
        key, lambda r, rows: float(r[key][rows].mean()))
    return St.two_level_bootstrap(metric, models, combine, clusters, paired=paired, B=B)


def contrasts(runs, val, B=St.B_DEFAULT):
    """The pre-specified differences (iterations.md D8) with their bootstrap CIs."""
    cl = val.cluster
    out = []
    def add(name, key, models, combine, direction, paired=()):
        res = _boot(models, key, combine, cl, paired=paired, B=B, val=val)
        out.append({"contrast": name, "estimate": res.estimate, "lo": res.lo, "hi": res.hi,
                    "reading": St.verdict(res, "difference", direction)})
    g = lambda *vs: {v: runs[v] for v in vs if v in runs}
    if {"b2r", "b2rba"} <= set(runs):
        add("P1 ordered NLL/vote: B²-rba − B²-r", "onll", g("b2r", "b2rba"), lambda x: x["b2rba"] - x["b2r"], -1)
        add("count NLL/vote: B²-rba − B²-r", "cnll", g("b2r", "b2rba"), lambda x: x["b2rba"] - x["b2r"], -1)
        add("mAP: B²-rba − B²-r", "map", g("b2r", "b2rba"), lambda x: x["b2rba"] - x["b2r"], +1)
        add("split AP: B²-rba − B²-r", "split_ap", g("b2r", "b2rba"), lambda x: x["b2rba"] - x["b2r"], +1)
        # P2: the spread ablation within each model; same runs scored two ways -> paired
        both = {f"{v}:{t}": [dict(r, onll=r["onll" if t == "own" else "onll_const"]) for r in runs[v]]
                for v in ("b2r", "b2rba") for t in ("own", "const")}
        add("P2 per-image S − one S per label, ordered NLL/vote, within B²-rba", "onll", both,
            lambda x: x["b2rba:own"] - x["b2rba:const"], -1, paired=(("b2rba:own", "b2rba:const"), ("b2r:own", "b2r:const")))
        add("P2 same, within B²-r", "onll", both, lambda x: x["b2r:own"] - x["b2r:const"], -1,
            paired=(("b2rba:own", "b2rba:const"), ("b2r:own", "b2r:const")))
        add("P2 difference of the two (B²-rba's gain from S − B²-r's)", "onll", both,
            lambda x: (x["b2rba:own"] - x["b2rba:const"]) - (x["b2r:own"] - x["b2r:const"]), -1,
            paired=(("b2rba:own", "b2rba:const"), ("b2r:own", "b2r:const")))
    if {"soft_bce", "soft_bce_ba"} <= set(runs):
        add("ordered NLL/vote: soft BCE-ba − soft BCE", "onll", g("soft_bce", "soft_bce_ba"),
            lambda x: x["soft_bce_ba"] - x["soft_bce"], -1)
    if set(D8) <= set(runs):
        add("P3 interaction: (soft BCE-ba − B²-rba) − (soft BCE − B²-r), ordered NLL/vote", "onll", g(*D8),
            lambda x: (x["soft_bce_ba"] - x["b2rba"]) - (x["soft_bce"] - x["b2r"]), +1)
    return out


def report(block="d8", root=RESULTS, datasets=("endoscapes", "amibr"), B=St.B_DEFAULT):
    guard("val")
    outdir = Path(root) / block / "tables_val"
    outdir.mkdir(parents=True, exist_ok=True)
    md = [f"# D8 — B²-rba (validation; block {block})", "",
          "Checkpoint A: discrimination of m against the panel majority. Checkpoint B: panel endpoints after the spread "
          "step. onll = NLL per vote of the ORDERED votes (who voted what); cnll = NLL per vote of the count k. "
          "d_onll_spread = per-image S − one S per label (same m, β), < 0 means the per-image spread is informative. "
          "Mean ± sd over seeds.", ""]
    long = []
    for ds in datasets:
        dirs = main_runs(block, ds, root)
        if not dirs:
            continue
        val = load_table(ds).take("val")
        runs = {v: [run_arrays(v, d, val) for d in dirs[v]] for v in D8 if v in dirs}
        for v, rs in runs.items():
            for s, r in enumerate(rs):
                long.append({"dataset": ds, "variant": v, "seed": s, **scores(r, val)})
        df = pd.DataFrame([x for x in long if x["dataset"] == ds])
        cols = ["map", "auroc", "bacc", "f1", "acc", "onll", "cnll", "ece_pmaj", "brier_pmaj", "split_auroc", "split_ap",
                "split_catch20", "d_onll_spread", "split_auroc_const"]
        md += [f"## {ds}", "", "| | seeds | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
        for v in D8:
            t = df[df.variant == v]
            if t.empty:
                continue
            cell = lambda c: "–" if c not in t or t[c].isna().all() else f"{t[c].mean():.4f} ± {t[c].std():.4f}"
            md.append(f"| {LABEL[v]} | {len(t)} | " + " | ".join(cell(c) for c in cols) + " |")
        md += ["", "Pre-specified differences (two-level bootstrap 95% CI):", "",
               "| contrast | estimate | 95% CI | reading |", "|---|---|---|---|"]
        for c in contrasts(runs, val, B):
            md.append(f"| {c['contrast']} | {c['estimate']:+.5f} | [{c['lo']:+.5f}, {c['hi']:+.5f}] | {c['reading']} |")
        rates = np.nanmean(load_table(ds).take("train").votes, 0)            # [R, K] observed yes-rate per rater
        md += ["", "Rater biases: observed training yes-rate per rater, and the learned β (logit scale, sum 0; mean ± sd "
               "over seeds):", "", "| label | rater | yes-rate | " + " | ".join(LABEL[v] + " β" for v in D8 if v in runs
                                                                            and _rater(v)) + " |",
               "|---|---|---|" + "---|" * sum(_rater(v) for v in runs)]
        for j in range(rates.shape[1]):
            for r_ in range(rates.shape[0]):
                bs = [np.array([x["beta"][j, r_] for x in runs[v]]) for v in D8 if v in runs and _rater(v)]
                md.append(f"| {j} | {r_ + 1} | {rates[r_, j]:.3f} | " + " | ".join(f"{b.mean():+.3f} ± {b.std():.3f}"
                                                                             for b in bs) + " |")
        md.append("")
    pd.DataFrame(long).to_csv(outdir / "d8_long.csv", index=False)
    (outdir / "d8_report.md").write_text("\n".join(md))
    return outdir / "d8_report.md"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("report",))
    ap.add_argument("block", nargs="?", default="d8")
    a = ap.parse_args()
    print(report(a.block).read_text())
