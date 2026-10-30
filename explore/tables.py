"""Markdown tables of the exploration (X1-X5) from results/explore/ and results/phase2/endoscapes/:
  python -m explore.tables > tables.md
"""
import json
from pathlib import Path
import numpy as np
from b2j.e1 import RESULTS
from explore import EXPLORE as E
P2 = Path(RESULTS) / "phase2" / "endoscapes"
SEEDS = [1, 2, 3, 4, 5]
DS = [("endoscapes", "Endoscapes", 3), ("amibr", "AMi-Br", 3), ("mhist", "MHIST", 7), ("ferplus", "FER+", "~10"), ("cifar10h", "CIFAR-10H", "~51")]
def ps(rows, filt, get):
    return np.array([np.nanmean([get(x) for x in rows if filt(x) and x["seed"] == s]) for s in SEEDS])
def f(a): return f"{np.nanmean(a):.3f}±{np.nanstd(a, ddof=1):.3f}"
def path(ds, name, sub=None):
    if sub: return E / "sub" / f"n{sub}" / "phase2" / ds / name
    return (P2 if ds == "endoscapes" and name != "spread.json" else E / "root" / "phase2" / ds) / name
out = []
w = out.append
w("**X1 · E1–E4 headline numbers per dataset** (macro over labels, test, mean ± sd over seeds; E1 = BACC at the E1 cut-off; E2 = disagreement AUROC; E3 = mean BACC over 50–100% coverage; E4 = share of frames where soft BCE and B²-B1 make the same call)\n")
w("| Dataset | Ckpt | E1 hard | E1 soft | E1 B²-B1 | E1 B²-B2 | E2 hard | E2 soft | E2 B² | E3 soft | E3 B²-B1 | E4 soft=B² |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|")
for ds, name, _ in DS:
    r1 = json.load(open(path(ds, "e1.json")))["rows"]; r2 = json.load(open(path(ds, "e2.json")))["rows"]
    r3 = json.load(open(path(ds, "e3.json")))["rows"]; r4 = json.load(open(path(ds, "e4.json")))["results"]
    for a, (x1, x2, x3, x4) in {"best-BACC": ("E1a", "E2a", "E3a", "E4a"), "own-loss": ("E1b", "E2b", "E3b", "E4b")}.items():
        e1 = {m: ps(r1, lambda x, m=m: x["exp"] == x1 and x["method"] == m, lambda x: x["test"]["all"]["bacc"]) for m in ("hard_bce", "soft_bce", "b2_b1", "b2_b2")}
        e2 = {m: ps(r2, lambda x, m=m: x["exp"] == x2 and x["method"] == m, lambda x: x["test"]["auroc"]) for m in ("hard_bce", "soft_bce", "b2")}
        e3 = {m: ps(r3, lambda x, m=m: x["exp"] == x3 and x["method"] == m, lambda x: x["mean_bacc_50_100"]) for m in ("soft_bce", "b2_b1")}
        same = np.mean([v["same_call"]["soft_bce|b2_b1"] for k, v in r4.items() if k.startswith(x4)])
        e3s = "n/a (rare labels)" if ds == "ferplus" else "saturated" if ds == "cifar10h" else None
        w(f"| {name} | {a} | {f(e1['hard_bce'])} | {f(e1['soft_bce'])} | {f(e1['b2_b1'])} | {f(e1['b2_b2'])} | {f(e2['hard_bce'])} | {f(e2['soft_bce'])} | {f(e2['b2'])} | "
          + (f"{e3s} | {e3s}" if e3s else f"{f(e3['soft_bce'])} | {f(e3['b2_b1'])}") + f" | {same:.3f} |")
w("\n**X2 · Spread check** (own-loss checkpoints; S = α₀ + α₁; vote NLL on test, lower is better; disagreement AUROC with and without B²'s spread)\n")
w("| Dataset | Raters | S median (5–95%), first label | Vote NLL soft BCE | Vote NLL B² const. S | Vote NLL B² | (side check: soft + 1 fitted S) | Disagree AUROC soft | B² mean only | B² |")
w("|---|---|---|---|---|---|---|---|---|---|")
def spread_row(ds, name, n, sub=None):
    r = [x for x in json.load(open(path(ds, "spread.json", sub)))["rows"] if x["pair"] == "own"]
    lab = r[0]["label"]; rr = [x for x in r if x["label"] == lab]
    med, p5, p95 = (np.mean([x["S_pct"][i] for x in rr]) for i in (2, 0, 4))
    nll = {k: np.mean(ps(r, lambda x: True, lambda x, k=k: x["nll"][k])) for k in r[0]["nll"]}
    auc = {k: ps(r, lambda x: True, lambda x, k=k: x["auroc_disagree"][k]) for k in r[0]["auroc_disagree"]}
    return (f"| {name} | {n} | {lab}: {med:.1f} ({p5:.1f}–{p95:.1f}) | {nll['soft_binom']:.3f} | {nll['b2_constS']:.3f} | {nll['b2']:.3f} | {nll['soft_fitS']:.3f} | "
            f"{f(auc['soft'])} | {f(auc['b2_mean'])} | {f(auc['b2'])} |")
for ds, name, n in DS:
    w(spread_row(ds, name, n))
w("\n**X3 · Trained on m raters per image, tested on the full panel** (own-loss checkpoints; disagreement AUROC and vote NLL against the full test panel; majority BACC at the E1 cut-off)\n")
w("| Dataset | m (train) | BACC soft | BACC B²-B1 | Disagree AUROC hard | soft | B² mean only | B² | Vote NLL soft | B² |")
w("|---|---|---|---|---|---|---|---|---|---|")
for ds, name, levels in (("mhist", "MHIST", [3, 5, None]), ("ferplus", "FER+", [3, 5, None]), ("cifar10h", "CIFAR-10H", [3, 10, None])):
    for m in levels:
        r1 = json.load(open(path(ds, "e1.json", m)))["rows"]; r2 = json.load(open(path(ds, "e2.json", m)))["rows"]
        sp = [x for x in json.load(open(path(ds, "spread.json", m)))["rows"] if x["pair"] == "own"]
        b = {k: ps(r1, lambda x, k=k: x["exp"] == "E1b" and x["method"] == k, lambda x: x["test"]["all"]["bacc"]) for k in ("soft_bce", "b2_b1")}
        h = ps(r2, lambda x: x["exp"] == "E2b" and x["method"] == "hard_bce", lambda x: x["test"]["auroc"])
        auc = {k: ps(sp, lambda x: True, lambda x, k=k: x["auroc_disagree"][k]) for k in ("soft", "b2_mean", "b2")}
        nll = {k: np.mean(ps(sp, lambda x: True, lambda x, k=k: x["nll"][k])) for k in ("soft_binom", "b2")}
        w(f"| {name} | {m or 'full'} | {f(b['soft_bce'])} | {f(b['b2_b1'])} | {f(h)} | {f(auc['soft'])} | {f(auc['b2_mean'])} | {f(auc['b2'])} | {nll['soft_binom']:.3f} | {nll['b2']:.3f} |")
w("\n**X4 · Same m = 3 models, scored against the full test panel vs against 3-rater test panels** (20 draws; own-loss checkpoints; mean over seeds)\n")
w("| Dataset | Test truth | Disagree AUROC B² − soft | …from the spread (B² − B² mean only) | Vote NLL gain from per-image S (B² const. S − B²) |")
w("|---|---|---|---|---|")
for ds, name, _ in (("mhist", "MHIST", 0), ("ferplus", "FER+", 0), ("cifar10h", "CIFAR-10H", 0)):
    r = json.load(open(path(ds, "evalsub.json", 3)))
    for key, lab in (("full", "full panel"), ("three", "3 raters")):
        g = lambda k: ps(r, lambda x: x[key] is not None, lambda x: x[key][k])
        w(f"| {name} | {lab} | {np.mean(g('auc_b2') - g('auc_soft')):+.4f} | {np.mean(g('auc_b2') - g('auc_b2_mean')):+.4f} | {np.mean(g('nll_b2_constS') - g('nll_b2')):.4f} |")
w("| Endoscapes (all 3 raters; for comparison) | 3 raters | +0.005 (X2) | ≈ 0 | 0.0018 |")
w("\n**X5 · Two-level cluster bootstrap** (difference, 95% interval; * = interval excludes 0; clusters = videos / slides / images; 2,000 resamples)\n")
NAMES = [("endoscapes", "Endoscapes"), ("amibr", "AMi-Br"), ("mhist", "MHIST"), ("ferplus", "FER+"), ("cifar10h", "CIFAR-10H"),
         ("mhist_m3", "MHIST, trained on 3"), ("ferplus_m3", "FER+, trained on 3"), ("cifar10h_m3", "CIFAR-10H, trained on 3")]
for pair, lab in (("own", "own-loss checkpoints"), ("bacc", "best-BACC checkpoints")):
    w(f"\n*{lab}*\n")
    w("| Dataset | Clusters | Disagree AUROC B² − soft | B² − hard | spread part (B² − B² mean) | Majority BACC B² − soft | B² − hard | Vote NLL soft − B² |")
    w("|---|---|---|---|---|---|---|---|")
    for fn, name in NAMES:
        r = json.load(open(E / "bootstrap" / f"{fn}.json"))["results"][pair]; c = r["comparisons"]
        def fm(k):
            v = c[k]; s = "*" if v["ci95"][0] > 0 or v["ci95"][1] < 0 else ""
            return f"{v['estimate']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]{s}"
        w(f"| {name} | {r['n_clusters']} | {fm('disagree_auroc|b2-soft')} | {fm('disagree_auroc|b2-hard')} | {fm('disagree_auroc|b2-b2_mean')} | {fm('majority_bacc|b2-soft')} | {fm('majority_bacc|b2-hard')} | {fm('vote_nll|soft-b2')} |")
print("\n".join(out))
