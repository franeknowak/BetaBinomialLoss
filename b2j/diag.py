"""Round D, D0: diagnostics of where B²-r's panel advantage comes from (validation only; journal_plan/iterations.md).

  python -m b2j.diag r1        -> results/journal/r1/diag_d0_val.md

1. Gain by ambiguity: per (image, head), NLL of soft BCE + Global-S minus NLL of B²-r (positive = B²-r better),
   seed s against seed s, binned by the observed minority share min(k, n - k) / n of the full panel.
2. Spread beyond the mean: within deciles of m, the Spearman correlation between B²-r's intra-image correlation
   rho = 1 / (S + 1) and the observed split indicator (any disagreement); averaged over deciles and seeds. Zero
   means S only re-encodes m.
3. Ceiling (CIFAR-10H only, ≈50 raters per image): a fresh panel of 3 validation votes per image, drawn from the
   full panel; the reference is the empirical-Bayes posterior predictive from the image's other ≈47 votes (per-class
   Beta prior fitted to the training panels; until 2026-10-19 a Jeffreys-smoothed Binomial, which every model beat).
   A strong reference, not a strict bound: a model could still beat it on some images. Reported: NLL per vote of
   each model on
   the 3-vote panel and the share of the Global-S → reference gap that B²-r closes.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import betabinom, spearmanr

from b2j import posthoc as P
from b2j.analysis import guard, main_runs
from b2j.data import load_table
from b2j.runs import RESULTS, RunSpec

BINS = [(0.0, 0.0), (1e-9, 0.15), (0.15, 0.30), (0.30, 0.50)]
BIN_NAMES = ["unanimous", "minority ≤ 15%", "15–30%", "30–50%"]


def _panel(variant, d, split="val"):
    z = np.load(d / f"{split}_B.npz")
    return P.panel(variant, z["out"], z["scale"])


def gain_by_ambiguity(runs, ev):
    minority = np.minimum(ev.k, ev.n - ev.k) / ev.n
    rows = []
    for s, (dg, db) in enumerate(zip(runs["soft_bce"], runs["b2r"])):
        mg, Sg = _panel("soft_bce", dg)
        mb, Sb = _panel("b2r", db)
        g = P.vote_nll(mg, Sg, ev.k, ev.n) - P.vote_nll(mb, Sb, ev.k, ev.n)
        for name, (lo, hi) in zip(BIN_NAMES, BINS):
            sel = (minority >= lo) & (minority <= hi) if hi == 0 else (minority > lo) & (minority <= hi)
            rows.append({"seed": s, "bin": name, "share": sel.mean(), "gain_mean": g[sel].mean() if sel.any() else np.nan,
                         "gain_sum": g[sel].sum(), "total": g.sum()})
    return rows


def spread_beyond_mean(runs, ev):
    split = ((ev.k > 0) & (ev.k < ev.n)).astype(float)
    out = []
    for d in runs["b2r"]:
        m, S = _panel("b2r", d)
        rho = 1.0 / (S + 1.0)
        for j in range(m.shape[1]):
            edges = np.quantile(m[:, j], np.linspace(0, 1, 11))
            b = np.clip(np.searchsorted(edges, m[:, j], side="right") - 1, 0, 9)
            cs = [spearmanr(rho[b == i, j], split[b == i, j])[0] for i in range(10)
                  if (b == i).sum() > 10 and split[b == i, j].std() > 0 and rho[b == i, j].std() > 0]
            out.append({"head": j, "spearman_within_m_deciles": float(np.mean(cs)) if cs else np.nan,
                        "log10S_p10": float(np.log10(np.quantile(S[:, j], 0.1))),
                        "log10S_p90": float(np.log10(np.quantile(S[:, j], 0.9)))})
    return out


def beta_prior(counts):
    """Per class, the Beta(a, b) prior on the one-vs-rest vote share that maximises the Beta-Binomial likelihood of
    the full panels (empirical Bayes). Jeffreys smoothing (0.5 per class) is far too heavy for CIFAR-10H, where most
    images are unanimous: every model beat that reference (2026-10-19)."""
    n = counts.sum(1)
    ab = []
    for j in range(counts.shape[1]):
        nll = lambda x: -betabinom.logpmf(counts[:, j], n, *np.exp(x)).sum()
        ab.append(np.exp(minimize(nll, np.log([0.1, 1.0]), method="Nelder-Mead").x))
    a, b = np.array(ab).T
    return a, b


def cifar_ceiling(block, root, full, n_panel=3, seed=999):
    """NLL per vote on a fresh 3-vote validation panel, for each (variant, training n) in the block."""
    val = full.take("val")
    rng = np.random.default_rng(seed)
    drawn = np.stack([rng.multivariate_hypergeometric(c.astype(np.int64), n_panel) for c in val.counts]).astype(float)
    rest = val.counts - drawn
    k3, n3 = drawn, np.full_like(drawn, float(n_panel))
    a, b = beta_prior(full.take("train").counts)
    rk, rn = rest, rest.sum(1, keepdims=True)
    ref = (P.vote_nll((a + rk) / (a + b + rn), a + b + rn, k3, n3) / n3).mean()
    res = {}
    for f in sorted((Path(root) / block / "cifar10h").glob("*/spec.json")):
        s = json.loads(f.read_text())
        spec = RunSpec.from_json(s)
        if (spec is None or spec.variant not in ("soft_bce", "b2r") or spec.regime != "frozen" or spec.train_frac != 1.0
                or spec.val_full_panel or not (f.parent / "summary.json").exists()):
            continue
        m, S = _panel(spec.variant, f.parent)
        n_tr = spec.n_sub or "all"
        res.setdefault((spec.variant, n_tr), []).append((P.vote_nll(m, S, k3, n3) / n3).mean())
        if spec.variant == "soft_bce":
            res.setdefault(("soft_bce plug-in", n_tr), []).append(
                (P.vote_nll(m, np.full_like(m, np.inf), k3, n3) / n3).mean())
    return ref, {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in res.items()}


def report(block, root=RESULTS, datasets=("cifar10h", "mhist", "endoscapes", "amibr")):
    guard("val")
    md = [f"# D0 diagnostics ({block}, validation)", ""]
    for ds in datasets:
        if not (Path(root) / block / ds).exists():
            continue
        full = load_table(ds)
        ev = full.take("val")
        runs = main_runs(block, ds, root)
        if "b2r" not in runs or "soft_bce" not in runs:
            continue
        md += [f"## {ds}", "", "Gain = NLL(soft BCE + Global-S) − NLL(B²-r) per (image, head); positive = B²-r better. "
               "Mean over seeds.", "", "| observed minority share | share of (image, head) | mean gain | share of total gain |",
               "|---|---|---|---|"]
        rows = gain_by_ambiguity(runs, ev)
        for name in BIN_NAMES:
            r = [x for x in rows if x["bin"] == name]
            md.append(f"| {name} | {np.mean([x['share'] for x in r]):.3f} | {np.nanmean([x['gain_mean'] for x in r]):+.4f} | "
                      f"{np.mean([x['gain_sum'] / x['total'] for x in r]):+.2f} |")
        sb = spread_beyond_mean(runs, ev)
        md += ["", "Spread beyond the mean (B²-r): Spearman(ρ = 1/(S+1), observed split) within deciles of m, "
               "and the spread of S across images (10th–90th percentile of log10 S).", "",
               "| head | Spearman within m deciles | log10 S p10 | log10 S p90 |", "|---|---|---|---|"]
        for j in sorted({x["head"] for x in sb}):
            r = [x for x in sb if x["head"] == j]
            md.append(f"| {j} | {np.nanmean([x['spearman_within_m_deciles'] for x in r]):+.3f} | "
                      f"{np.mean([x['log10S_p10'] for x in r]):.2f} | {np.mean([x['log10S_p90'] for x in r]):.2f} |")
        if ds == "cifar10h":
            ref, res = cifar_ceiling(block, root, full)
            md += ["", f"Ceiling on a fresh 3-vote validation panel (NLL per vote). Reference (empirical-Bayes Beta prior "
                   f"updated with the other ≈47 votes): **{ref:.4f}**.", "", "| model | trained on n | NLL/vote | sd | seeds |", "|---|---|---|---|---|"]
            for (v, n), (mu, sd, c) in sorted(res.items(), key=lambda kv: (str(kv[0][1]), kv[0][0])):
                md.append(f"| {v} | {n} | {mu:.4f} | {sd:.4f} | {c} |")
            for n in sorted({n for _, n in res}, key=str):
                if ("soft_bce", n) in res and ("b2r", n) in res:
                    gs, b2 = res[("soft_bce", n)][0], res[("b2r", n)][0]
                    md.append(f"\nn = {n}: B²-r closes {(gs - b2) / (gs - ref):.0%} of the Global-S → reference gap "
                              f"({gs - ref:.4f} per vote).")
        md.append("")
    out = Path(root) / block / "diag_d0_val.md"
    out.write_text("\n".join(md))
    return out


if __name__ == "__main__":
    print(report(sys.argv[1] if len(sys.argv) > 1 else "r1").read_text())
