"""Module f: tables T1-T3 and figures F1-F5 (plan §4.4 R3), written and checked against VALIDATION outputs.

  python -m b2j.analysis tables  <block> [--split val]
  python -m b2j.analysis figures <block> [--split val]

split="test" raises unless --open-test is given: that flag belongs to R3 only (CLAUDE.md, HANDOFF rule 3).
Every quantity fitted post hoc (thresholds, c, Global-S) comes from validation and is applied unchanged.
Rows per dataset: every trained variant (checkpoint A for discrimination, B for panel and use-case endpoints),
plus post-hoc rows: <variant>:plugin (S = inf; step 1 of the spread ladder plug-in -> Global-S -> B²-r) and the
soft-BCE ensemble (mean m; mixture of the members' Binomial vote distributions). Seeds: mean ± sd.
Comparisons carry a two-level bootstrap 95% CI; there are no margins (05_plan amendment 2026-10-16 #1).
See journal_plan/specs/f_launcher.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from b2j import metrics as M
from b2j import posthoc as P
from b2j import stats as St
from b2j.data import DATA_ROOT, load_table
from b2j.runs import RESULTS

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # fixed order
LABEL = {"hard_bce": "hard BCE", "hard_bce_w": "hard BCE-w", "soft_bce": "soft BCE (Global-S)", "b2r": "B²-r",
         "b2r_w": "B²-r-w", "b2r_kl": "B²-r + KL", "b2m": "B²-M", "b2r_linear": "B²-r linear", "dup": "DUP",
         "b2ms": "B²-ms", "b2ms_shrink0.03": "B²-ms λ 0.03", "b2ms_shrink0.3": "B²-ms λ 0.3", "b2ms_shrink3": "B²-ms λ 3",
         "b2ms_var0.03": "B²-ms var λ 0.03", "b2ms_var0.3": "B²-ms var λ 0.3", "b2ms_var3": "B²-ms var λ 3",
         "soft_bce:plugin": "soft BCE plug-in", "hard_bce:plugin": "hard BCE plug-in", "ensemble": "soft-BCE ensemble",
         "ensemble_gs": "soft-BCE ensemble + Global-S"}
ENSEMBLE_B2 = ("b2r", "b2ms", "b2ms_var0.03", "b2ms_var0.3", "b2ms_var3")   # round D, D3: seed ensembles of B² variants
LABEL.update({f"ensemble_{v}": f"{LABEL[v]} ensemble" for v in ENSEMBLE_B2})
PLUGIN_ROWS = ("soft_bce", "hard_bce")
RECALL_GRID = np.round(np.linspace(0.5, 1.0, 51), 4)     # F5


def guard(split, open_test=False):
    if split == "test" and not open_test:
        raise PermissionError("test outputs are opened once, in R3 (pass --open-test only then)")
    if split not in ("val", "test"):
        raise ValueError(split)


def boot_strata(dataset, tab):
    """Bootstrap strata (plan §4.3): CIFAR-10H images are resampled within their CIFAR class; else none."""
    if dataset == "cifar10h":
        return np.load(DATA_ROOT / "cifar10h" / "test_labels.npy")[tab.cluster]
    return None


def main_runs(block, dataset, root=RESULTS, regime="frozen"):
    """{variant: [run dirs over seeds]} for the main configuration (full panel, all training rows)."""
    out = {}
    for f in sorted((Path(root) / block / dataset).glob("*/spec.json")):
        s = json.loads(f.read_text())
        if (s["regime"] == regime and s["n_sub"] is None and s.get("part", "all") == "all"
                and s.get("train_frac", 1.0) == 1.0 and not s["val_full_panel"]
                and (f.parent / "summary.json").exists()):
            out.setdefault(s["variant"], []).append(f.parent)
    return out


# ---------------------------------------------------------------- predictions per row
def _pred(m_disc, thr, m, pmf, psplit):
    return {"disc": m_disc, "thr": thr, "m": m, "pmf": pmf, "psplit": psplit}


def run_rows(variant, d, split, ev, val):
    """Rows {name: pred} for one run on the evaluation table `ev` (validation-fitted quantities from `val`)."""
    za, zb = np.load(d / f"{split}_A.npz"), np.load(d / f"{split}_B.npz")
    va = np.load(d / "val_A.npz")
    assert (za["ids"] == ev.ids).all() and (va["ids"] == val.ids).all()
    yv = M.majority(val.k, val.n)
    if variant == "dup":
        return {"dup": _pred(None, None, None, None, expit(zb["out"][..., 0].astype(np.float64)))}
    thr = P.threshold_max_bacc(P.mean_spread(variant, va["out"])[0], yv)
    mA = P.mean_spread(variant, za["out"])[0]
    mB, SB = P.panel(variant, zb["out"], zb["scale"])
    rows = {variant: _pred(mA, thr, mB, P.vote_pmf(mB, SB, ev.n), P.p_split3(mB, SB))}
    if variant in PLUGIN_ROWS:
        inf = np.full_like(mB, np.inf)
        rows[f"{variant}:plugin"] = _pred(mA, thr, mB, P.vote_pmf(mB, inf, ev.n), P.p_split3(mB, inf))
    return rows


def ensemble_row(dirs, split, ev, val, variant="soft_bce", spread=False):
    """Seed ensemble: mean m (threshold from the validation mean m); mixture of the members' vote pmfs.

    spread=False (the pre-registered soft-BCE ensemble): Binomial members (S = inf). spread=True (round D, D3): each
    member keeps its own post-hoc spread (Global-S for soft BCE, c-scaled S for B²), i.e. a mixture of Beta-Binomials.
    """
    yv = M.majority(val.k, val.n)
    mA = np.mean([P.mean_spread(variant, np.load(d / f"{split}_A.npz")["out"])[0] for d in dirs], 0)
    vmA = np.mean([P.mean_spread(variant, np.load(d / "val_A.npz")["out"])[0] for d in dirs], 0)
    if spread:
        members = [P.panel(variant, z["out"], z["scale"]) for z in (np.load(d / f"{split}_B.npz") for d in dirs)]
    else:
        members = [(P.mean_spread(variant, np.load(d / f"{split}_B.npz")["out"])[0], None) for d in dirs]
        members = [(m, np.full_like(m, np.inf)) for m, _ in members]
    return _pred(mA, P.threshold_max_bacc(vmA, yv), np.mean([m for m, _ in members], 0),
                 P.mixture_pmf(members, ev.n), P.mixture_p_split3(members))


# ---------------------------------------------------------------- metrics per row
def nll_rows(p, ev, per_vote):
    """Vote NLL of each test row [N, K] from the row's pmf (divided by n when per vote, as on CIFAR-10H)."""
    lik = np.take_along_axis(p["pmf"], ev.k.astype(int)[..., None], -1)[..., 0]
    nll = -np.log(np.clip(lik, 1e-300, None))
    return nll / ev.n if per_vote else nll


def row_metrics(p, ev, per_vote):
    y = M.majority(ev.k, ev.n)
    out = {}
    if p["disc"] is not None:
        out.update(M.discrimination(p["disc"], y, p["thr"]))
    if p["pmf"] is not None:
        out["vote_nll"] = float(nll_rows(p, ev, per_vote).mean())
        pmaj = P.p_majority(p["pmf"], ev.n)
        out.update({"ece_pmaj": M.ece(pmaj, y), "brier_pmaj": M.brier(pmaj, y), "fcr90_m": M.fcr_at_recall(p["m"], y),
                    "fcr90_pmaj": M.fcr_at_recall(pmaj, y), "aurc": M.aurc(pmaj, y)})
        if (ev.n > 3).any():
            out["theta_mse"] = M.theta_mse(p["m"], ev.k, ev.n)
    if p["psplit"] is not None:
        t = M.split_target(ev.k, ev.n)
        tr = M.triage(p["psplit"], t)
        out.update({"psplit_auroc": tr["auroc"], "brier_psplit": M.brier(p["psplit"], t), "u1c_split_cov80":
                    M.split_rate_at_coverage(p["psplit"], t), "u2_ap": tr["ap"], "u2_catch20": tr["catch"]})
    return out


def collect(block, dataset, split="val", root=RESULTS, open_test=False):
    """Long table: one row per (row name, seed) with every metric."""
    guard(split, open_test)
    full = load_table(dataset)
    ev, val = full.take(split), full.take("val")
    per_vote = dataset == "cifar10h"
    recs = []
    runs = main_runs(block, dataset, root)
    for variant, dirs in runs.items():
        for seed, d in enumerate(dirs):
            for name, p in run_rows(variant, d, split, ev, val).items():
                recs.append({"row": name, "seed": seed, **row_metrics(p, ev, per_vote)})
    if len(runs.get("soft_bce", [])) > 1:
        recs.append({"row": "ensemble", "seed": 0, **row_metrics(ensemble_row(runs["soft_bce"], split, ev, val), ev, per_vote)})
        recs.append({"row": "ensemble_gs", "seed": 0,
                     **row_metrics(ensemble_row(runs["soft_bce"], split, ev, val, spread=True), ev, per_vote)})
    for v in ENSEMBLE_B2:
        if len(runs.get(v, [])) > 1:
            recs.append({"row": f"ensemble_{v}", "seed": 0,
                         **row_metrics(ensemble_row(runs[v], split, ev, val, v, spread=True), ev, per_vote)})
    return pd.DataFrame(recs)


TABLES = {
    "T1_discrimination": ["acc", "bacc", "f1", "map", "auroc"],
    "T2_panel": ["vote_nll", "ece_pmaj", "brier_pmaj", "psplit_auroc", "brier_psplit", "theta_mse"],
    "T3_use_cases": ["fcr90_m", "fcr90_pmaj", "aurc", "u1c_split_cov80", "u2_ap", "u2_catch20"],
}
ORDER = list(LABEL)                       # fixed row order: trained variants, then post-hoc rows, then the ensemble


def summarise(df, cols, posthoc_rows=True):
    """Mean ± sd over seeds per row. Post-hoc rows share their model's m, so T1 leaves them out."""
    if not posthoc_rows:
        df = df[~df["row"].str.contains(":")]
    cols = [c for c in cols if c in df]
    g = df.groupby("row")[cols]
    mean, sd, n = g.mean(), g.std(), g.size()
    out = mean.map(lambda v: f"{v:.4f}").astype(str)
    for c in cols:
        out[c] = [f"{mu:.4f} ± {s:.4f}" if k > 1 and np.isfinite(mu) else (f"{mu:.4f}" if np.isfinite(mu) else "–")
                  for mu, s, k in zip(mean[c], sd[c].fillna(0), n)]
    out.insert(0, "seeds", n)
    out = out.loc[sorted(out.index, key=lambda r: ORDER.index(r) if r in ORDER else len(ORDER))]
    out.index = [LABEL.get(i, i) for i in out.index]
    return out.dropna(how="all")


def _md_table(t):
    """DataFrame → GitHub markdown table (no `tabulate` dependency; it is not in the cluster env)."""
    rows = [["", *map(str, t.columns)], *[[str(i), *map(str, r)] for i, r in zip(t.index, t.to_numpy())]]
    return "\n".join(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
                     + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def tables(block, split="val", root=RESULTS, open_test=False, datasets=("endoscapes", "cifar10h", "mhist", "amibr", "ferplus")):
    outdir = Path(root) / block / f"tables_{split}"
    outdir.mkdir(parents=True, exist_ok=True)
    md = [f"# Tables ({block}, {split})", ""]
    for ds in datasets:
        if not (Path(root) / block / ds).exists():
            continue
        df = collect(block, ds, split, root, open_test)
        df.to_csv(outdir / f"{ds}_long.csv", index=False)
        for name, cols in TABLES.items():
            t = summarise(df, cols, posthoc_rows=name != "T1_discrimination")
            t = t[[c for c in t.columns if c == "seeds" or t[c].ne("–").any()]]
            t = t[t.drop(columns="seeds").ne("–").any(axis=1)]
            md += [f"## {name} — {ds}", "", _md_table(t), ""]
    (outdir / "tables.md").write_text("\n".join(md))
    return outdir / "tables.md"


# ---------------------------------------------------------------- figures
def _plain_log(ax, axis="y"):
    """Log axis with plain-number tick labels (no 2.2×10⁰)."""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


def _style(ax):
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8980")
    ax.tick_params(colors="#5c5b55", labelsize=8)


def fig1_theory(outdir):
    """F1: (a) decision boundary in m against the P_maj threshold t, n = 3, per S; (b) P_maj at m = 0.3 against S."""
    import matplotlib.pyplot as plt
    from scipy.optimize import brentq
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    ts = np.linspace(0.1, 0.5, 41)
    for c, S in zip(COLORS, (0.5, 1, 2, 5, np.inf)):
        b = [brentq(lambda m: P.p_majority(P.vote_pmf(np.array([[m]]), np.array([[S]]), np.array([[3]])), 3)[0, 0] - t,
                    1e-6, 0.5 + 1e-9) for t in ts]
        ax[0].plot(ts, b, color=c, lw=1.5, label=f"S = {S:g}" if np.isfinite(S) else "S = ∞ (plug-in)")
    ax[0].set(xlabel="threshold t on P_maj", ylabel="boundary in m (n = 3)")
    ax[0].legend(frameon=False, fontsize=7)
    Ss = np.logspace(-0.5, 3, 80)
    for c, n in zip(COLORS, (3, 11, 51)):
        pm = P.p_majority(P.vote_pmf(np.full((80, 1), 0.3), Ss[:, None], np.full((80, 1), n)), n)[:, 0]
        ax[1].plot(Ss, pm, color=c, lw=1.5, label=f"n = {n}")
    ax[1].set(xscale="log", xlabel="spread S", ylabel="P_maj at m = 0.3")
    ax[1].legend(frameon=False, fontsize=7)
    for a in ax:
        _style(a)
    fig.tight_layout()
    return _save(fig, outdir, "F1_theory")


def fig3_spread_vs_m(block, dataset, split, outdir, root=RESULTS):
    """F3: S against m per B² variant (seed 0, all heads pooled), log S."""
    import matplotlib.pyplot as plt
    ev = load_table(dataset).take(split)
    runs = main_runs(block, dataset, root)
    vs = [v for v in ("b2r", "b2r_w", "b2r_kl", "b2m") if v in runs]
    fig, ax = plt.subplots(1, len(vs), figsize=(2.6 * len(vs), 2.6), sharey=True, squeeze=False)
    for a, v in zip(ax[0], vs):
        z = np.load(runs[v][0] / f"{split}_B.npz")
        m, S = P.panel(v, z["out"], z["scale"])
        a.scatter(m.ravel(), S.ravel(), s=3, alpha=0.3, color=COLORS[1], linewidths=0)
        a.set(yscale="log", xlabel="m", title=LABEL[v])
        a.title.set_fontsize(9)
        _style(a)
        _plain_log(a)
    ax[0][0].set_ylabel(f"S ({dataset}, {split})")
    fig.tight_layout()
    return _save(fig, outdir, f"F3_spread_vs_m_{dataset}")


def fig4_spread_over_training(block, dataset, outdir, root=RESULTS):
    """F4: validation median S (mean over heads) per evaluation, B²-r frozen vs fine-tuned; fitted c per evaluation."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for c, regime in zip(COLORS, ("frozen", "finetune")):
        for i, d in enumerate(main_runs(block, dataset, root, regime).get("b2r", [])):
            log = [json.loads(l) for l in (d / "log.jsonl").read_text().splitlines()]
            ep = [r["epoch"] for r in log]
            ax[0].plot(ep, [np.mean(r["val_median_S"]) for r in log], color=c, lw=1.2, alpha=0.8,
                       label=regime if i == 0 else None)
            ax[1].plot(ep, [np.mean(r["scale"]) for r in log], color=c, lw=1.2, alpha=0.8, label=regime if i == 0 else None)
    ax[0].set(xlabel="epoch", ylabel="validation median S (after c)")
    ax[1].set(xlabel="epoch", ylabel="fitted c")
    for a in ax:
        a.legend(frameon=False, fontsize=7)
        _style(a)
    fig.suptitle(f"B²-r, {dataset} (validation)", fontsize=9)
    fig.tight_layout()
    return _save(fig, outdir, f"F4_spread_over_training_{dataset}")


def fig5_pr_and_risk(block, dataset, split, outdir, root=RESULTS, open_test=False, B=St.B_DEFAULT):
    """F5 (05_plan amendment 2026-10-16 #2), descriptive; the only claim is at recall 0.90 (S6a/S6b).

    (a) precision against recall for B²-r at checkpoint B ranked by P_maj and by m, soft BCE's m as a faint third
        curve: precision on RECALL_GRID (mean over heads, then over seeds);
    (b) the precision difference P_maj − m within B²-r with its two-level bootstrap 95% band (the same runs scored
        two ways: seeds paired);
    (c) risk–coverage of each score (first head, first seed).
    The numbers go to F5_pr_<dataset>.json beside the figure.
    """
    import matplotlib.pyplot as plt
    guard(split, open_test)
    full = load_table(dataset)
    ev, val = full.take(split), full.take("val")
    y = M.majority(ev.k, ev.n)
    runs = main_runs(block, dataset, root)
    b2 = [run_rows("b2r", d, split, ev, val)["b2r"] for d in runs["b2r"]]
    scores = {"B²-r P_maj": [P.p_majority(p["pmf"], ev.n) for p in b2], "B²-r m": [p["m"] for p in b2]}
    if "soft_bce" in runs:
        scores["soft BCE m"] = [run_rows("soft_bce", d, split, ev, val)["soft_bce"]["m"] for d in runs["soft_bce"]]
    curves = {k: np.mean([M.precision_at_recalls(s, y, RECALL_GRID) for s in v], 0) for k, v in scores.items()}
    diff = St.two_level_bootstrap(lambda s, r: M.precision_at_recalls(s[r], y[r], RECALL_GRID),
                                  {"pmaj": scores["B²-r P_maj"], "m": scores["B²-r m"]},
                                  lambda v: v["pmaj"] - v["m"], ev.cluster, boot_strata(dataset, ev),
                                  paired=[("pmaj", "m")], B=B)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    style = {"B²-r P_maj": (COLORS[1], 1.6, 1.0), "B²-r m": (COLORS[0], 1.6, 1.0), "soft BCE m": ("#8a8980", 1.0, 0.6)}
    for name, prec in curves.items():
        c, lw, al = style[name]
        ax[0].plot(RECALL_GRID, prec, color=c, lw=lw, alpha=al, label=name)
        s = scores[name][0][:, 0]
        conf = np.maximum(s, 1 - s)
        err = ((s >= 0.5) != y[:, 0])[np.argsort(-conf, kind="stable")]
        ax[2].plot(np.arange(1, len(err) + 1) / len(err), np.cumsum(err) / np.arange(1, len(err) + 1),
                   color=c, lw=lw, alpha=al, label=name)
    ax[1].fill_between(RECALL_GRID, diff.lo, diff.hi, color=COLORS[1], alpha=0.2, lw=0)
    ax[1].plot(RECALL_GRID, diff.estimate, color=COLORS[1], lw=1.5)
    ax[1].axhline(0, color="#8a8980", lw=0.8)
    for a in ax[:2]:
        a.axvline(0.9, color="#8a8980", lw=0.8, ls="--")
    ax[0].set(xlabel="recall", ylabel="precision")
    ax[1].set(xlabel="recall", ylabel="precision, P_maj − m (B²-r)")
    ax[2].set(xlabel="coverage", ylabel="selective risk (first head)")
    for a in (ax[0], ax[2]):
        a.legend(frameon=False, fontsize=7)
    for a in ax:
        _style(a)
    fig.suptitle(f"{dataset} ({split}; {len(b2)} B²-r seeds, mean over heads; 95% bootstrap band)", fontsize=9)
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"F5_pr_{dataset}.json").write_text(json.dumps({
        "recall": RECALL_GRID.tolist(), "precision": {k: v.tolist() for k, v in curves.items()},
        "diff_pmaj_minus_m": {"estimate": diff.estimate.tolist(), "lo": diff.lo.tolist(), "hi": diff.hi.tolist()}},
        indent=1))
    return _save(fig, outdir, f"F5_pr_risk_{dataset}")


def fig2_advantage_vs_n(block, dataset, split, outdir, root=RESULTS, open_test=False, B=St.B_DEFAULT):
    """F2: H4 advantage = vote NLL of soft BCE + Global-S minus that of B²-r (positive = B²-r better; per vote on
    CIFAR-10H), checkpoint B, against the training raters n. Different runs: the bootstrap shares the test clusters
    and resamples seeds per model. Global-S sits on soft BCE's m, so the advantage mixes mean quality and spread.
    The numbers go to F2_advantage_<dataset>.json beside the figure."""
    import matplotlib.pyplot as plt
    from b2j.runs import RunSpec, tables as run_tables
    guard(split, open_test)
    full = load_table(dataset)
    ev = full.take(split)
    per_vote = dataset == "cifar10h"
    nll = {}                                            # (variant, n) -> per-seed [N, K] NLL
    for f in sorted((Path(root) / block / dataset).glob("*/spec.json")):
        s = json.loads(f.read_text())
        spec = RunSpec.from_json(s)
        if (spec is None or spec.variant not in ("soft_bce", "b2r") or spec.regime != "frozen" or spec.train_frac != 1.0
                or spec.val_full_panel or not (f.parent / "summary.json").exists()):
            continue
        _, val, _ = run_tables(spec, full)
        p = run_rows(s["variant"], f.parent, split, ev, val)[s["variant"]]
        n = s["n_sub"] or int(np.median(full.n))
        nll.setdefault((s["variant"], n), []).append(nll_rows(p, ev, per_vote))
    ns = sorted(n for v, n in nll if v == "b2r" and ("soft_bce", n) in nll)
    if not ns:
        raise ValueError(f"no soft BCE / B²-r pair at any n for {dataset}")
    res = {n: St.two_level_bootstrap(lambda a, r: a[r].mean(), {"gs": nll[("soft_bce", n)], "b2r": nll[("b2r", n)]},
                                     lambda v: v["gs"] - v["b2r"], ev.cluster, boot_strata(dataset, ev), B=B)
           for n in ns}
    fig, ax = plt.subplots(figsize=(4, 3))
    est = np.array([res[n].estimate for n in ns])
    err = np.array([[res[n].estimate - res[n].lo for n in ns], [res[n].hi - res[n].estimate for n in ns]])
    ax.errorbar(ns, est, yerr=err, color=COLORS[1], lw=1.5, marker="o", ms=4, capsize=3)
    ax.axhline(0, color="#8a8980", lw=0.8)
    unit = "NLL/vote" if per_vote else "NLL"
    ax.set(xscale="log", xlabel="training raters per image n", ylabel=f"advantage ({unit}), Global-S − B²-r", xticks=ns)
    _style(ax)
    _plain_log(ax, "x")
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"F2_advantage_{dataset}.json").write_text(json.dumps(
        {str(n): {"estimate": r.estimate, "lo": r.lo, "hi": r.hi, "seeds_gs": len(nll[("soft_bce", n)]),
                  "seeds_b2r": len(nll[("b2r", n)])} for n, r in res.items()}, indent=1))
    return _save(fig, outdir, f"F2_advantage_vs_n_{dataset}")


def figS_train_fraction(block, split, outdir, root=RESULTS, open_test=False, B=St.B_DEFAULT,
                        datasets=("endoscapes", "cifar10h")):
    """Supplement (05_plan amendment 2026-10-16 #6): Δ = B²-r − soft BCE on mAP and BACC (m at checkpoint A,
    validation threshold) against the share of training clusters kept, with two-level bootstrap 95% CIs (different
    runs: shared test clusters, seeds resampled per model). The 100% point is the main configuration.
    The numbers go to S_train_fraction.json beside the figure."""
    import matplotlib.pyplot as plt
    from b2j.runs import RunSpec
    guard(split, open_test)
    res = {}
    for ds in datasets:
        full = load_table(ds)
        ev, val = full.take(split), full.take("val")
        y = M.majority(ev.k, ev.n)
        preds = {}                                      # (variant, frac) -> per-seed predictions
        for f in sorted((Path(root) / block / ds).glob("*/spec.json")):
            spec = RunSpec.from_json(json.loads(f.read_text()))
            if (spec is None or spec.variant not in ("soft_bce", "b2r") or spec.regime != "frozen"
                    or spec.n_sub is not None or spec.val_full_panel or not (f.parent / "summary.json").exists()):
                continue
            preds.setdefault((spec.variant, spec.train_frac), []).append(
                run_rows(spec.variant, f.parent, split, ev, val)[spec.variant])
        fracs = sorted(fr for v, fr in preds if v == "b2r" and ("soft_bce", fr) in preds)
        res[ds] = {}
        for fr in fracs:
            res[ds][fr] = {}
            for key in ("map", "bacc"):
                r = St.two_level_bootstrap(lambda p, rows: M.discrimination(p["disc"][rows], y[rows], p["thr"])[key],
                                           {"b2r": preds[("b2r", fr)], "soft": preds[("soft_bce", fr)]},
                                           lambda v: v["b2r"] - v["soft"], ev.cluster, boot_strata(ds, ev), B=B)
                res[ds][fr][key] = {"estimate": r.estimate, "lo": r.lo, "hi": r.hi}
    if not any(res.values()):
        raise ValueError("no training-fraction runs")
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
    for c, (ds, by) in zip(COLORS, res.items()):
        fr = sorted(by)
        for a, key in zip(ax, ("map", "bacc")):
            est = np.array([by[f][key]["estimate"] for f in fr])
            err = np.array([[by[f][key]["estimate"] - by[f][key]["lo"] for f in fr],
                            [by[f][key]["hi"] - by[f][key]["estimate"] for f in fr]])
            a.errorbar(fr, est, yerr=err, color=c, lw=1.5, marker="o", ms=4, capsize=3, label=ds)
    for a, key in zip(ax, ("mAP", "BACC")):
        a.axhline(0, color="#8a8980", lw=0.8)
        a.set(xscale="log", xlabel="share of training clusters", ylabel=f"Δ {key}, B²-r − soft BCE")
        a.set_xticks([0.1, 0.3, 1.0])
        _style(a)
        _plain_log(a, "x")
    ax[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Training-set size ({split}; 95% bootstrap CI)", fontsize=9)
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "S_train_fraction.json").write_text(json.dumps(
        {ds: {str(f): v for f, v in by.items()} for ds, by in res.items()}, indent=1))
    return _save(fig, outdir, "S_train_fraction")


def _save(fig, outdir, name):
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outdir / f"{name}.png"


def figures(block, split="val", root=RESULTS, open_test=False):
    import matplotlib
    matplotlib.use("Agg")
    guard(split, open_test)
    out = Path(root) / block / f"figures_{split}"
    made = [fig1_theory(out)]
    try:
        made.append(figS_train_fraction(block, split, out, root, open_test))
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"skipped the training-fraction figure: {e!r}")
    for ds in ("endoscapes", "cifar10h", "mhist", "amibr", "ferplus"):
        if not (Path(root) / block / ds).exists():
            continue
        for f in (lambda: fig3_spread_vs_m(block, ds, split, out, root),
                  lambda: fig5_pr_and_risk(block, ds, split, out, root, open_test),
                  lambda: fig4_spread_over_training(block, ds, out, root),
                  lambda: fig2_advantage_vs_n(block, ds, split, out, root, open_test)):
            try:
                made.append(f())
            except (FileNotFoundError, KeyError, IndexError, ValueError) as e:
                print(f"skipped a figure for {ds}: {e!r}")
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("tables", "figures"))
    ap.add_argument("block")
    ap.add_argument("--split", default="val")
    ap.add_argument("--open-test", action="store_true", help="R3 only")
    a = ap.parse_args()
    if a.what == "tables":
        print(tables(a.block, a.split, open_test=a.open_test).read_text())
    else:
        print("\n".join(map(str, figures(a.block, a.split, open_test=a.open_test))))
