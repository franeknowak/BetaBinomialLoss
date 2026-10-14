"""Module f: tables T1-T3 and figures F1-F5 (plan §4.4 R3), written and checked against VALIDATION outputs.

  python -m b2j.analysis tables  <block> [--split val]
  python -m b2j.analysis figures <block> [--split val]

split="test" raises unless --open-test is given: that flag belongs to R3 only (CLAUDE.md, HANDOFF rule 3).
Every quantity fitted post hoc (thresholds, c, Global-S, S*(m)) comes from validation and is applied unchanged.
Rows per dataset: every trained variant (checkpoint A for discrimination, B for panel and use-case endpoints),
plus post-hoc rows: <variant>:plugin (S = inf), <variant>:S*(m) (fitted on that variant's own validation m), and the
soft-BCE ensemble (mean m; mixture of the members' Binomial vote distributions). Seeds: mean ± sd.
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
from b2j.data import load_table
from b2j.runs import RESULTS

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # fixed order
LABEL = {"hard_bce": "hard BCE", "hard_bce_w": "hard BCE-w", "soft_bce": "soft BCE (Global-S)", "b2r": "B²-r",
         "b2r_w": "B²-r-w", "b2r_kl": "B²-r + KL", "b2m": "B²-M", "b2r_linear": "B²-r linear", "dup": "DUP",
         "soft_bce:plugin": "soft BCE plug-in", "soft_bce:S*(m)": "S*(m)", "b2r:S*(m)": "S*(m_B²)",
         "hard_bce:plugin": "hard BCE plug-in", "ensemble": "soft-BCE ensemble"}
PLUGIN_ROWS = ("soft_bce", "hard_bce")
SOFM_ROWS = ("soft_bce", "b2r")


def guard(split, open_test=False):
    if split == "test" and not open_test:
        raise PermissionError("test outputs are opened once, in R3 (pass --open-test only then)")
    if split not in ("val", "test"):
        raise ValueError(split)


def main_runs(block, dataset, root=RESULTS, regime="frozen"):
    """{variant: [run dirs over seeds]} for the main configuration (full panel, all training rows)."""
    out = {}
    for f in sorted((Path(root) / block / dataset).glob("*/spec.json")):
        s = json.loads(f.read_text())
        if (s["regime"] == regime and s["n_sub"] is None and s["part"] == "all" and not s["val_full_panel"]
                and (f.parent / "summary.json").exists()):
            out.setdefault(s["variant"], []).append(f.parent)
    return out


# ---------------------------------------------------------------- predictions per row
def _pred(m_disc, thr, m, pmf, psplit):
    return {"disc": m_disc, "thr": thr, "m": m, "pmf": pmf, "psplit": psplit}


def run_rows(variant, d, split, ev, val):
    """Rows {name: pred} for one run on the evaluation table `ev` (validation-fitted quantities from `val`)."""
    za, zb = np.load(d / f"{split}_A.npz"), np.load(d / f"{split}_B.npz")
    va, vb = np.load(d / "val_A.npz"), np.load(d / "val_B.npz")
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
    if variant in SOFM_ROWS:
        vmB = P.panel(variant, vb["out"], vb["scale"])[0]
        S = P.s_of_m(P.fit_s_of_m(vmB, val.k, val.n), mB)
        rows[f"{variant}:S*(m)"] = _pred(mA, thr, mB, P.vote_pmf(mB, S, ev.n), P.p_split3(mB, S))
    return rows


def ensemble_row(dirs, split, ev, val):
    """Soft-BCE ensemble over seeds: mean m (threshold from the validation mean m); mixture of Binomial pmfs."""
    yv = M.majority(val.k, val.n)
    mA = np.mean([P.mean_spread("soft_bce", np.load(d / f"{split}_A.npz")["out"])[0] for d in dirs], 0)
    vmA = np.mean([P.mean_spread("soft_bce", np.load(d / "val_A.npz")["out"])[0] for d in dirs], 0)
    members = [(P.mean_spread("soft_bce", np.load(d / f"{split}_B.npz")["out"])[0], None) for d in dirs]
    members = [(m, np.full_like(m, np.inf)) for m, _ in members]
    return _pred(mA, P.threshold_max_bacc(vmA, yv), np.mean([m for m, _ in members], 0),
                 P.mixture_pmf(members, ev.n), P.mixture_p_split3(members))


# ---------------------------------------------------------------- metrics per row
def row_metrics(p, ev, per_vote):
    y = M.majority(ev.k, ev.n)
    out = {}
    if p["disc"] is not None:
        out.update(M.discrimination(p["disc"], y, p["thr"]))
    if p["pmf"] is not None:
        lik = np.take_along_axis(p["pmf"], ev.k.astype(int)[..., None], -1)[..., 0]
        nll = -np.log(np.clip(lik, 1e-300, None))
        out["vote_nll"] = float((nll / ev.n).mean() if per_vote else nll.mean())
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


def tables(block, split="val", root=RESULTS, open_test=False, datasets=("endoscapes", "cifar10h", "mhist")):
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
            md += [f"## {name} — {ds}", "", t.to_markdown(), ""]
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


def fig5_fcr_and_risk(block, dataset, split, outdir, root=RESULTS, open_test=False):
    """F5: (a) FCR (1 - precision) against recall for B²-r P_maj, B²-r m and soft-BCE m; (b) risk-coverage of P_maj."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    guard(split, open_test)
    full = load_table(dataset)
    ev, val = full.take(split), full.take("val")
    y = M.majority(ev.k, ev.n)
    runs = main_runs(block, dataset, root)
    curves = {}
    if "b2r" in runs:
        p = run_rows("b2r", runs["b2r"][0], split, ev, val)["b2r"]
        curves["B²-r P_maj"] = P.p_majority(p["pmf"], ev.n)
        curves["B²-r m"] = p["m"]
    if "soft_bce" in runs:
        curves["soft BCE m"] = run_rows("soft_bce", runs["soft_bce"][0], split, ev, val)["soft_bce"]["m"]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    for c, (name, s) in zip(COLORS, curves.items()):
        prec, rec, _ = precision_recall_curve(y[:, 0], s[:, 0])
        ax[0].plot(rec, 1 - prec, color=c, lw=1.5, label=name)
        conf = np.maximum(s[:, 0], 1 - s[:, 0])
        err = ((s[:, 0] >= 0.5) != y[:, 0])[np.argsort(-conf, kind="stable")]
        cov = np.arange(1, len(err) + 1) / len(err)
        ax[1].plot(cov, np.cumsum(err) / np.arange(1, len(err) + 1), color=c, lw=1.5, label=name)
    ax[0].axvline(0.9, color="#8a8980", lw=0.8, ls="--")
    ax[0].set(xlabel="recall", ylabel="false-confirmation rate", xlim=(0.5, 1))
    ax[1].set(xlabel="coverage", ylabel="selective risk")
    for a in ax:
        a.legend(frameon=False, fontsize=7)
        _style(a)
    fig.suptitle(f"{dataset}, first head ({split})", fontsize=9)
    fig.tight_layout()
    return _save(fig, outdir, f"F5_fcr_risk_{dataset}")


def fig2_margin_vs_n(block, dataset, split, outdir, root=RESULTS, open_test=False):
    """F2: H4 margin = vote NLL per vote of B²-r minus S*(m_B²) (same run), against training raters n."""
    import matplotlib.pyplot as plt
    guard(split, open_test)
    full = load_table(dataset)
    ev = full.take(split)
    pts = {}
    for f in sorted((Path(root) / block / dataset).glob("b2r__frozen*/spec.json")):
        s = json.loads(f.read_text())
        if s["part"] != "all" or s["val_full_panel"] or not (f.parent / "summary.json").exists():
            continue
        from b2j.runs import RunSpec, tables as run_tables
        _, val, _ = run_tables(RunSpec(**s), full)
        r = run_rows("b2r", f.parent, split, ev, val)
        nll = lambda p: float((-np.log(np.clip(np.take_along_axis(p["pmf"], ev.k.astype(int)[..., None], -1)[..., 0],
                                                1e-300, None)) / ev.n).mean())
        n = s["n_sub"] or int(np.median(full.n))
        pts.setdefault(n, []).append(nll(r["b2r"]) - nll(r["b2r:S*(m)"]))
    fig, ax = plt.subplots(figsize=(4, 3))
    ns = sorted(pts)
    ax.plot(ns, [np.mean(pts[n]) for n in ns], color=COLORS[1], lw=1.5, marker="o", ms=4)
    ax.axhline(0, color="#8a8980", lw=0.8)
    ax.set(xscale="log", xlabel="training raters per image n", ylabel="Δ NLL/vote, B²-r − S*(m_B²)", xticks=ns)
    _style(ax)
    _plain_log(ax, "x")
    fig.tight_layout()
    return _save(fig, outdir, f"F2_margin_vs_n_{dataset}")


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
    for ds in ("endoscapes", "cifar10h", "mhist"):
        if not (Path(root) / block / ds).exists():
            continue
        for f in (lambda: fig3_spread_vs_m(block, ds, split, out, root),
                  lambda: fig5_fcr_and_risk(block, ds, split, out, root, open_test),
                  lambda: fig4_spread_over_training(block, ds, out, root),
                  lambda: fig2_margin_vs_n(block, ds, split, out, root, open_test)):
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
