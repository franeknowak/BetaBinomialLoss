"""Round D, D8 (journal_plan/iterations.md): B²-rba against B²-r in the 2 × 2 {rater bias no / yes} × {per-image
concentration no / yes} (soft BCE, B²-r, soft BCE-ba, B²-rba); since the R3 lock also the B²-final endpoints E0–E5 of 05_plan
§5 (renamed from F0–F5, amendment 2026-10-23 #6): E0–E4 from the frozen block d8, E5 = E1, E2 and the count-NLL
advantage from the fine-tuned block r2.

  python -m b2j.d8 report [block] [--split val]   -> results/journal/<block>/tables_<split>/d8_report.md, d8_long.csv

split="test" needs --open-test (R3 only). The threshold and P2's single S are fitted on validation and applied
unchanged; c and β are the run's own (fitted on validation by the trainer).

Per run: checkpoint A for discrimination (m against the panel majority, threshold max-BACC on validation), checkpoint
B for the panel endpoints after the concentration step. Panel endpoints for every variant on one footing:
  - ordered-vote NLL per vote (P1; interchangeable models: count NLL + log C(n, k));
  - count NLL per vote, P_maj ECE / Brier, split AUROC / AP / catch-20 from the count pmf of the known panel
    (rater-aware: summed over the 2^R vote patterns);
  - concentration ablation (P2): per-image S against one S per label refitted on validation with m (and β) fixed.
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
CONCENTRATION = ("b2r", "b2rba")                  # the variants with a per-image concentration (P2 ablation)


def _rater(v):
    return VARIANTS[v].loss.rater


def run_arrays(variant, d, val, split="val", ev=None):
    """Per-row arrays [N, K] of one run on `split` (label table `ev`, default `val`), plus its learned β [K, R] (None
    if raters are interchangeable). The max-BACC threshold and P2's single S are fitted on validation (`val`)."""
    ev = val if ev is None else ev
    za, zb = np.load(d / f"{split}_A.npz"), np.load(d / f"{split}_B.npz")
    va, vb = np.load(d / "val_A.npz"), np.load(d / "val_B.npz")
    assert (za["ids"] == ev.ids).all() and (zb["ids"] == ev.ids).all() and (va["ids"] == val.ids).all()
    mA = P.mean_concentration(variant, za["out"])[0]
    thr = P.threshold_max_bacc(P.mean_concentration(variant, va["out"])[0], M.majority(val.k, val.n))
    m, S = P.panel(variant, zb["out"], zb["scale"])
    beta = P.rater_offsets(variant, zb["out"]) if _rater(variant) else None
    pmf = P.rater_count_pmf(m, S, beta, ev.votes) if _rater(variant) else P.vote_pmf(m, S, ev.n)
    r = {"mA": mA, "thr": thr, "m": m, "pmf": pmf,
         "onll": P.ordered_nll(m, S, beta, ev.votes) / ev.n,
         "cnll": -np.log(np.clip(np.take_along_axis(pmf, ev.k.astype(int)[..., None], -1)[..., 0], 1e-300, None)) / ev.n,
         "psplit": 1 - P.p_unanimous(pmf, ev.n), "beta": beta}
    if variant in CONCENTRATION:                  # P2 / E2: one S per label instead of the per-image S (same m, same β)
        vm = P.panel(variant, vb["out"], vb["scale"])[0]
        vbeta = P.rater_offsets(variant, vb["out"]) if beta is not None else None
        S0 = np.ones_like(vm)
        sc = P.fit_scale_rater(vm, S0, vbeta, val.votes) if beta is not None else P.fit_scale(vm, S0, val.k, val.n)
        Sc = np.broadcast_to(sc, m.shape)
        pc = P.rater_count_pmf(m, Sc, beta, ev.votes) if beta is not None else P.vote_pmf(m, Sc, ev.n)
        r.update({"onll_const": P.ordered_nll(m, Sc, beta, ev.votes) / ev.n,
                  "psplit_const": 1 - P.p_unanimous(pc, ev.n)})
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
        out["d_onll_concentration"] = out["onll"] - out["onll_const"]       # < 0: the per-image S is informative
        out["split_auroc_const"] = M.triage(r["psplit_const"][rows], t)["auroc"]
    return out


def _boot(models, key, combine, clusters, paired=(), B=St.B_DEFAULT, val=None):
    y, t = M.majority(val.k, val.n).astype(float), _split_target(val)
    metric = {"map": lambda r, rows: M._heads(M.ap, r["mA"][rows], y[rows]),
              "bacc": lambda r, rows: M.discrimination(r["mA"][rows], y[rows], r["thr"])["bacc"],
              "split_ap": lambda r, rows: M._heads(M.ap, r["psplit"][rows], t[rows])}.get(
        key, lambda r, rows: float(r[key][rows].mean()))
    return St.two_level_bootstrap(metric, models, combine, clusters, paired=paired, B=B)


def contrasts(runs, val, B=St.B_DEFAULT):
    """The pre-specified differences (iterations.md D8; 05_plan §5 E0–E5) with their bootstrap CIs on the rows of
    `val` (the evaluation table: validation or, in R3, test)."""
    cl = val.cluster
    out = []
    def add(name, key, models, combine, direction, paired=(), kind="difference"):
        res = _boot(models, key, combine, cl, paired=paired, B=B, val=val)
        out.append({"contrast": name, "estimate": res.estimate, "lo": res.lo, "hi": res.hi,
                    "reading": St.verdict(res, kind, direction)})
    g = lambda *vs: {v: runs[v] for v in vs if v in runs}
    if {"soft_bce", "b2rba"} <= set(runs):              # E0: S0 with B²-final (checkpoint A)
        for key in ("bacc", "map"):
            add(f"E0 Δ {key.upper()}: B²-rba − soft BCE", key, g("soft_bce", "b2rba"),
                lambda x: x["b2rba"] - x["soft_bce"], +1)
    if {"b2r", "b2rba"} <= set(runs):
        add("E1 (P1) ordered NLL/vote: B²-rba − B²-r", "onll", g("b2r", "b2rba"), lambda x: x["b2rba"] - x["b2r"], -1,
            kind="superiority")
        add("E4 count NLL/vote: B²-rba − B²-r", "cnll", g("b2r", "b2rba"), lambda x: x["b2rba"] - x["b2r"], -1)
        add("E4 mAP: B²-rba − B²-r", "map", g("b2r", "b2rba"), lambda x: x["b2rba"] - x["b2r"], +1)
        add("E4 split AP: B²-rba − B²-r", "split_ap", g("b2r", "b2rba"), lambda x: x["b2rba"] - x["b2r"], +1)
        # P2: the concentration ablation within each model; same runs scored two ways -> paired
        both = {f"{v}:{t}": [dict(r, onll=r["onll" if t == "own" else "onll_const"]) for r in runs[v]]
                for v in ("b2r", "b2rba") for t in ("own", "const")}
        add("E2 (P2) per-image S − one S per label, ordered NLL/vote, within B²-rba", "onll", both,
            lambda x: x["b2rba:own"] - x["b2rba:const"], -1, paired=(("b2rba:own", "b2rba:const"), ("b2r:own", "b2r:const")))
        add("E2 secondary: same, within B²-r", "onll", both, lambda x: x["b2r:own"] - x["b2r:const"], -1,
            paired=(("b2rba:own", "b2rba:const"), ("b2r:own", "b2r:const")))
        add("E2 secondary: difference of the two (B²-rba's gain from S − B²-r's)", "onll", both,
            lambda x: (x["b2rba:own"] - x["b2rba:const"]) - (x["b2r:own"] - x["b2r:const"]), -1,
            paired=(("b2rba:own", "b2rba:const"), ("b2r:own", "b2r:const")))
    if {"soft_bce", "soft_bce_ba"} <= set(runs):
        add("E1c ordered NLL/vote: soft BCE-ba − soft BCE", "onll", g("soft_bce", "soft_bce_ba"),
            lambda x: x["soft_bce_ba"] - x["soft_bce"], -1)
    if {"soft_bce", "b2rba"} <= set(runs):              # E5 (fine-tuned, block r2), also shown frozen
        add("E5 count NLL/vote advantage: soft BCE (Global-S) − B²-rba", "cnll", g("soft_bce", "b2rba"),
            lambda x: x["soft_bce"] - x["b2rba"], +1)
    if set(D8) <= set(runs):
        add("E3 (P3) interaction: (soft BCE-ba − B²-rba) − (soft BCE − B²-r), ordered NLL/vote", "onll", g(*D8),
            lambda x: (x["soft_bce_ba"] - x["b2rba"]) - (x["soft_bce"] - x["b2r"]), +1)
    return out


def report(block="d8", split="val", root=RESULTS, open_test=False, datasets=("endoscapes", "amibr"), B=St.B_DEFAULT):
    guard(split, open_test)
    outdir = Path(root) / block / f"tables_{split}"
    outdir.mkdir(parents=True, exist_ok=True)
    md = [f"# D8 / E0–E5 — B²-rba ({split}; block {block})", "",
          "Checkpoint A: discrimination of m against the panel majority. Checkpoint B: panel endpoints after the concentration "
          "step. onll = NLL per vote of the ORDERED votes (who voted what); cnll = NLL per vote of the count k. "
          "d_onll_concentration = per-image S − one S per label (same m, β), < 0 means the per-image concentration is informative. "
          "Mean ± sd over seeds.", ""]
    long = []
    for ds in datasets:
        dirs = main_runs(block, ds, root)
        if not dirs:
            continue
        full = load_table(ds)
        val, ev = full.take("val"), full.take(split)
        runs = {v: [run_arrays(v, d, val, split, ev) for d in dirs[v]] for v in D8 if v in dirs}
        for v, rs in runs.items():
            for s, r in enumerate(rs):
                long.append({"dataset": ds, "variant": v, "seed": s, **scores(r, ev)})
        df = pd.DataFrame([x for x in long if x["dataset"] == ds])
        cols = ["map", "auroc", "bacc", "f1", "acc", "onll", "cnll", "ece_pmaj", "brier_pmaj", "split_auroc", "split_ap",
                "split_catch20", "d_onll_concentration", "split_auroc_const"]
        md += [f"## {ds}", "", "| | seeds | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
        for v in D8:
            t = df[df.variant == v]
            if t.empty:
                continue
            cell = lambda c: "–" if c not in t or t[c].isna().all() else f"{t[c].mean():.4f} ± {t[c].std():.4f}"
            md.append(f"| {LABEL[v]} | {len(t)} | " + " | ".join(cell(c) for c in cols) + " |")
        md += ["", "Pre-specified differences (two-level bootstrap 95% CI):", "",
               "| contrast | estimate | 95% CI | reading |", "|---|---|---|---|"]
        for c in contrasts(runs, ev, B):
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
    ap.add_argument("--split", default="val")
    ap.add_argument("--open-test", action="store_true", help="R3 only")
    a = ap.parse_args()
    print(report(a.block, a.split, open_test=a.open_test).read_text())
