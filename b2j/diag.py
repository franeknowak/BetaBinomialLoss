"""Round D, D0: diagnostics of where B²-r's panel advantage comes from (validation only; journal_plan/iterations.md).

  python -m b2j.diag r1        -> results/journal/r1/diag_d0_val.md
  python -m b2j.diag d12       -> results/journal/d12/diag_d0_val.md (sections 1-2 for every B² variant in the block)
  python -m b2j.diag recovery r1 -> results/journal/r1/mean_recovery_val.md (D5: mean recovery from few votes)

1. Gain by ambiguity: per (image, head), NLL of soft BCE + Global-S minus NLL of B²-r (positive = B²-r better),
   seed s against seed s, binned by the observed minority share min(k, n - k) / n of the full panel.
2. Concentration beyond the mean: within deciles of m, the Spearman correlation between B²-r's intra-image correlation
   rho = 1 / (S + 1) and the observed split indicator (any disagreement); averaged over deciles and seeds.
   Sign: at fixed m a smaller S (larger rho) puts more mass on unanimous panels, so an informative S gives a
   NEGATIVE correlation (until 2026-10-20 read the other way round; iterations.md D0b). Confounded by m within
   a decile, hence also:
2b. Concentration ablation: each B² model's own m with its per-image S replaced by one S per head (fitted on the same
   validation votes, as the models' own concentration step is). Vote NLL and split AUROC (score = 1 − P(unanimous))
   with and without the per-image S; a better score with it = S carries information beyond m.
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
from b2j import stats as St
from b2j.analysis import boot_strata, guard, main_runs
from b2j.data import load_table
from b2j.metrics import auroc
from b2j.runs import RESULTS, RunSpec
from b2j.variants import VARIANTS

BINS = [(0.0, 0.0), (1e-9, 0.15), (0.15, 0.30), (0.30, 0.50)]
BIN_NAMES = ["unanimous", "minority ≤ 15%", "15–30%", "30–50%"]


def _panel(variant, d, split="val"):
    z = np.load(d / f"{split}_B.npz")
    return P.panel(variant, z["out"], z["scale"])


def gain_by_ambiguity(runs, ev, variant="b2r"):
    minority = np.minimum(ev.k, ev.n - ev.k) / ev.n
    rows = []
    for s, (dg, db) in enumerate(zip(runs["soft_bce"], runs[variant])):
        mg, Sg = _panel("soft_bce", dg)
        mb, Sb = _panel(variant, db)
        g = P.vote_nll(mg, Sg, ev.k, ev.n) - P.vote_nll(mb, Sb, ev.k, ev.n)
        for name, (lo, hi) in zip(BIN_NAMES, BINS):
            sel = (minority >= lo) & (minority <= hi) if hi == 0 else (minority > lo) & (minority <= hi)
            rows.append({"seed": s, "bin": name, "share": sel.mean(), "gain_mean": g[sel].mean() if sel.any() else np.nan,
                         "gain_sum": g[sel].sum(), "total": g.sum()})
    return rows


def concentration_beyond_mean(runs, ev, variant="b2r"):
    split = ((ev.k > 0) & (ev.k < ev.n)).astype(float)
    out = []
    for d in runs[variant]:
        m, S = _panel(variant, d)
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


def concentration_ablation(runs, ev, variant="b2r"):
    """Per seed: vote NLL per (image, head) and mean split AUROC over heads, per-image S vs one S per head."""
    split = ((ev.k > 0) & (ev.k < ev.n)).astype(float)
    out = []
    for s, d in enumerate(runs[variant]):
        m, S = _panel(variant, d)
        Sc = np.broadcast_to(P.fit_scale(m, np.ones_like(m), ev.k, ev.n), m.shape)
        r = {"seed": s}
        for tag, SS in (("own", S), ("const", Sc)):
            r[f"nll_{tag}"] = float(P.vote_nll(m, SS, ev.k, ev.n).mean())
            pu = P.p_unanimous(P.vote_pmf(m, SS, ev.n), ev.n)
            a = [auroc(1 - pu[ok, j], split[ok, j]) for j in range(m.shape[1]) for ok in [ev.n[:, j] >= 2]]
            r[f"auroc_{tag}"] = float(np.mean([x for x in a if x is not None]))
        out.append(r)
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
        if (spec is None or not (spec.variant == "soft_bce" or (VARIANTS[spec.variant].loss.kind == "b2"
                                                               and not VARIANTS[spec.variant].loss.plus_one)) or spec.regime != "frozen" or spec.train_frac != 1.0
                or spec.val_full_panel or not (f.parent / "summary.json").exists()):
            continue
        m, S = _panel(spec.variant, f.parent)
        n_tr = spec.n_sub or "all"
        res.setdefault((spec.variant, n_tr), []).append((P.vote_nll(m, S, k3, n3) / n3).mean())
        if spec.variant == "soft_bce":
            res.setdefault(("soft_bce plug-in", n_tr), []).append(
                (P.vote_nll(m, np.full_like(m, np.inf), k3, n3) / n3).mean())
    return ref, {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in res.items()}


def report(block, root=RESULTS, datasets=("cifar10h", "mhist", "endoscapes", "amibr", "ferplus")):
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
        md.append(f"## {ds}")
        for v in [v for v in runs if VARIANTS[v].loss.kind == "b2" and not VARIANTS[v].loss.plus_one]:
            md += ["", f"### {v}", "", f"Gain = NLL(soft BCE + Global-S) − NLL({v}) per (image, head); positive = {v} "
                   "better. Mean over seeds.", "",
                   "| observed minority share | share of (image, head) | mean gain | share of total gain |", "|---|---|---|---|"]
            rows = gain_by_ambiguity(runs, ev, v)
            for name in BIN_NAMES:
                r = [x for x in rows if x["bin"] == name]
                md.append(f"| {name} | {np.mean([x['share'] for x in r]):.3f} | "
                          f"{np.nanmean([x['gain_mean'] for x in r]):+.4f} | {np.mean([x['gain_sum'] / x['total'] for x in r]):+.2f} |")
            sb = concentration_beyond_mean(runs, ev, v)
            md += ["", f"Concentration beyond the mean ({v}): Spearman(ρ = 1/(S+1), observed split) within deciles of m, "
                   "and the spread of S across images (10th–90th percentile of log10 S).", "",
                   "| head | Spearman within m deciles | log10 S p10 | log10 S p90 |", "|---|---|---|---|"]
            for j in sorted({x["head"] for x in sb}):
                r = [x for x in sb if x["head"] == j]
                md.append(f"| {j} | {np.nanmean([x['spearman_within_m_deciles'] for x in r]):+.3f} | "
                          f"{np.mean([x['log10S_p10'] for x in r]):.2f} | {np.mean([x['log10S_p90'] for x in r]):.2f} |")
        md += ["", "Concentration ablation: own m, per-image S vs one S per head (validation; mean ± sd over seeds). "
               "ΔNLL = NLL(one S) − NLL(per-image S) per (image, head), ΔAUROC = AUROC(per-image) − AUROC(one S); "
               "positive = the per-image S carries information beyond m.", "",
               "| variant | NLL per-image S | ΔNLL | split AUROC per-image S | ΔAUROC |", "|---|---|---|---|---|"]
        for v in [v for v in runs if VARIANTS[v].loss.kind == "b2" and not VARIANTS[v].loss.plus_one]:
            r = concentration_ablation(runs, ev, v)
            dn = [x["nll_const"] - x["nll_own"] for x in r]
            da = [x["auroc_own"] - x["auroc_const"] for x in r]
            md.append(f"| {v} | {np.mean([x['nll_own'] for x in r]):.4f} | {np.mean(dn):+.4f} ± {np.std(dn):.4f} | "
                      f"{np.mean([x['auroc_own'] for x in r]):.4f} | {np.mean(da):+.4f} ± {np.std(da):.4f} |")
        if ds == "cifar10h":
            ref, res = cifar_ceiling(block, root, full)
            md += ["", f"Ceiling on a fresh 3-vote validation panel (NLL per vote). Reference (empirical-Bayes Beta prior "
                   f"updated with the other ≈47 votes): **{ref:.4f}**.", "", "| model | trained on n | NLL/vote | sd | seeds |", "|---|---|---|---|---|"]
            for (v, n), (mu, sd, c) in sorted(res.items(), key=lambda kv: (str(kv[0][1]), kv[0][0])):
                md.append(f"| {v} | {n} | {mu:.4f} | {sd:.4f} | {c} |")
            for n in sorted({n for _, n in res}, key=str):
                for v in sorted({v for v, m in res if m == n and v not in ("soft_bce", "soft_bce plug-in")}):
                    if ("soft_bce", n) in res:
                        gs, b2 = res[("soft_bce", n)][0], res[(v, n)][0]
                        md.append(f"\nn = {n}: {v} closes {(gs - b2) / (gs - ref):.0%} of the Global-S → reference gap "
                                  f"({gs - ref:.4f} per vote).")
        md.append("")
    out = Path(root) / block / "diag_d0_val.md"
    out.write_text("\n".join(md))
    return out


# ---------------------------------------------------------------- D5: recovery of the hidden mean from few votes
def mean_recovery(block, dataset, root=RESULTS, B=2000):
    """User hypothesis (2026-10-20): trained on few votes per image, B² recovers the full-panel vote share better than
    soft BCE. Runs of `block` with n_sub votes per training (and validation) image; their m on the validation images
    is scored against the FULL validation panel: squared error (m − k/n)², per (image, head).

    The full-panel share is itself noisy; its expected share of the error, mean p(1 − p)/(n − 1), is the noise floor
    (the same for every model), so excess = MSE − floor estimates E(m − θ)². Returns {(checkpoint, n): row}."""
    guard("val")
    full = load_table(dataset)
    ev = full.take("val")
    share = ev.k / ev.n
    floor = float((share * (1 - share) / np.maximum(ev.n - 1, 1)).mean())
    se = {}                                     # (checkpoint, variant, n) -> per-seed [N, K] squared error
    for f in sorted((Path(root) / block / dataset).glob("*/spec.json")):
        s = json.loads(f.read_text())
        if (s["variant"] not in ("soft_bce", "b2r") or s["regime"] != "frozen" or s.get("train_frac", 1.0) != 1.0
                or s["val_full_panel"] or not (f.parent / "summary.json").exists()):
            continue
        n = s["n_sub"] or "all"
        for ck in ("A", "B"):
            z = np.load(f.parent / f"val_{ck}.npz")
            assert (z["ids"] == ev.ids).all()
            m = P.mean_concentration(s["variant"], z["out"])[0]
            se.setdefault((ck, s["variant"], n), []).append((m - share) ** 2)
    out = {}
    for ck, v, n in sorted(se, key=lambda t: (t[0], str(t[2]))):
        if v != "b2r" or (ck, "soft_bce", n) not in se:
            continue
        g, b = se[(ck, "soft_bce", n)], se[(ck, "b2r", n)]
        r = St.two_level_bootstrap(lambda a, rows: a[rows].mean(), {"gs": g, "b2r": b}, lambda x: x["gs"] - x["b2r"],
                                   ev.cluster, boot_strata(dataset, ev), B=B)
        mg, mb = float(np.mean([x.mean() for x in g])), float(np.mean([x.mean() for x in b]))
        out[(ck, n)] = {"mse_soft_bce": mg, "mse_b2r": mb, "diff": r.estimate, "lo": r.lo, "hi": r.hi, "floor": floor,
                        "seeds": (len(g), len(b))}
    return out


def recovery_report(block="r1", root=RESULTS, datasets=("cifar10h", "mhist")):
    md = [f"# D5 — recovery of the full-panel vote share from few votes ({block}, validation)", "",
          "m of runs trained (and checkpointed) on n votes per image, scored against the full validation panel. "
          "MSE = mean (m − k/n)² per (image, head); floor = the full panel's own sampling noise (same for every model); "
          "excess = MSE − floor ≈ E(m − θ)². Δ = MSE(soft BCE) − MSE(B²-r), positive = B²-r closer; 95% CI from the "
          "two-level bootstrap (images, seeds). Checkpoint A = best validation discrimination, B = best validation "
          "vote NLL.", ""]
    for ds in datasets:
        if not (Path(root) / block / ds).exists():
            continue
        res = mean_recovery(block, ds, root)
        md += [f"## {ds}", "", "| checkpoint | n | MSE soft BCE | MSE B²-r | Δ [95% CI] | Δ / excess soft BCE | seeds |",
               "|---|---|---|---|---|---|---|"]
        for (ck, n), r in res.items():
            exc = r["mse_soft_bce"] - r["floor"]
            md.append(f"| {ck} | {n} | {r['mse_soft_bce']:.5f} | {r['mse_b2r']:.5f} | {r['diff']:+.5f} "
                      f"[{r['lo']:+.5f}, {r['hi']:+.5f}] | {r['diff'] / exc:+.1%} | {r['seeds'][0]}/{r['seeds'][1]} |")
        md += ["", f"Noise floor of the full panel: {next(iter(res.values()))['floor']:.5f}.", ""]
    out = Path(root) / block / "mean_recovery_val.md"
    out.write_text("\n".join(md))
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "recovery":
        print(recovery_report(sys.argv[2] if len(sys.argv) > 2 else "r1").read_text())
    else:
        print(report(sys.argv[1] if len(sys.argv) > 1 else "r1").read_text())
