"""Headline numbers per dataset (E1-E4, spread check) with Welch tests over seeds, printed:
  python -m explore.summary <dir with <dataset>/e*.json> <dataset> ...
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

D = Path(sys.argv[1]); DS = sys.argv[2:]
SEEDS = [1, 2, 3, 4, 5]
def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b): return 1.0
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)
def ms(a): a = np.asarray(a, float); return f"{np.nanmean(a):.3f}±{np.nanstd(a, ddof=1):.3f}"
def pstar(p): return f"p={p:.3f}" + ("*" if p < 0.05 else "")

def per_seed(rows, filt, get):
    return [np.nanmean([get(x) for x in rows if filt(x) and x["seed"] == s]) for s in SEEDS]

for ds in DS:
    print(f"\n==================== {ds}")
    p = D / ds
    if (p / "e1.json").exists():
        r = json.load(open(p / "e1.json"))["rows"]
        for exp in ("E1a", "E1b"):
            for sub in ("all", "unanimous", "disputed"):
                for met in ("bacc", "auroc", "ap"):
                    v = {m: per_seed(r, lambda x, m=m: x["exp"] == exp and x["method"] == m, lambda x: x["test"][sub][met])
                         for m in ("hard_bce", "soft_bce", "b2_b0", "b2_b1", "b2_b2")}
                    best = max(v, key=lambda m: np.nanmean(v[m]))
                    print(f"E1 {exp} {sub:9} {met:5} " + "  ".join(f"{m} {ms(v[m])}" for m in v) +
                          f" | b1-soft {pstar(welch(v['b2_b1'], v['soft_bce']))} b2-soft {pstar(welch(v['b2_b2'], v['soft_bce']))}"
                          f" b0-soft {pstar(welch(v['b2_b0'], v['soft_bce']))} b1-hard {pstar(welch(v['b2_b1'], v['hard_bce']))}")
    if (p / "e2.json").exists():
        r = json.load(open(p / "e2.json"))["rows"]
        for exp in ("E2a", "E2b"):
            for met in ("auroc", "ap", "bacc"):
                v = {m: per_seed(r, lambda x, m=m: x["exp"] == exp and x["method"] == m, lambda x: x["test"][met])
                     for m in ("hard_bce", "soft_bce", "b2")}
                print(f"E2 {exp} {met:5} " + "  ".join(f"{m} {ms(v[m])}" for m in v) +
                      f" | b2-soft {pstar(welch(v['b2'], v['soft_bce']))} b2-hard {pstar(welch(v['b2'], v['hard_bce']))}")
    if (p / "e3.json").exists():
        r = json.load(open(p / "e3.json"))["rows"]
        for exp in ("E3a", "E3b"):
            v = {m: per_seed(r, lambda x, m=m: x["exp"] == exp and x["method"] == m, lambda x: x["mean_bacc_50_100"])
                 for m in ("hard_bce", "soft_bce", "b2_b0", "b2_b1", "b2_b2")}
            peak = {m: max(np.nanmean(per_seed(r, lambda x, m=m: x["exp"] == exp and x["method"] == m,
                                                lambda x, c=c: x["coverage"][c]["bacc"])) for c in ("1.00", "0.90", "0.80", "0.70", "0.60", "0.50")) for m in v}
            full = {m: np.nanmean(per_seed(r, lambda x, m=m: x["exp"] == exp and x["method"] == m, lambda x: x["coverage"]["1.00"]["bacc"])) for m in v}
            print(f"E3 {exp} meanBACC50-100 " + "  ".join(f"{m} {ms(v[m])}" for m in v) +
                  f" | b1-soft {pstar(welch(v['b2_b1'], v['soft_bce']))} b2-soft {pstar(welch(v['b2_b2'], v['soft_bce']))}"
                  f" b1-b0 {pstar(welch(v['b2_b1'], v['b2_b0']))} b0-soft {pstar(welch(v['b2_b0'], v['soft_bce']))}")
            print(f"   at 100%: " + " ".join(f"{m} {full[m]:.3f}" for m in v) + "  peak: " + " ".join(f"{m} {peak[m]:.3f}" for m in v))
    if (p / "e4.json").exists():
        R = json.load(open(p / "e4.json"))["results"]
        for exp in ("E4a", "E4b"):
            keys = [k for k in R if k.startswith(exp)]
            same = {pr: np.mean([R[k]["same_call"][pr] for k in keys]) for pr in R[keys[0]]["same_call"]}
            macro = {m: np.nanmean([R[k]["models"][m]["test"]["all"]["bacc"] for k in keys]) for m in R[keys[0]]["models"]}
            grp = {g: {} for g in ("none", "minority", "majority", "all")}
            for k in keys:
                for pat, v in R[k]["combos"].items():
                    if pat == "all": continue
                    a, b = map(int, pat.split("/"))
                    g = "none" if a == 0 else "all" if a == b else "minority" if a <= b / 2 else "majority"
                    for c, cnt in v.items():
                        grp[g][c] = grp[g].get(c, 0) + cnt
            print(f"E4 {exp} macro BACC " + " ".join(f"{m} {v:.3f}" for m, v in macro.items()) +
                  " | same call " + " ".join(f"{a} {v:.3f}" for a, v in same.items()))
            for g, v in grp.items():
                if v.get("n", 0):
                    print(f"   {g:8} n={v['n']:6}  all right {v['all right']/v['n']:.3f}  all wrong {v['all wrong']/v['n']:.3f}  "
                          f"only hard right {v['only hard_bce right']} only soft right {v['only soft_bce right']} only B2 right {v['only b2_b1 right']} | "
                          f"only hard wrong {v['only hard_bce wrong']} only soft wrong {v['only soft_bce wrong']} only B2 wrong {v['only b2_b1 wrong']}")
    if (p / "spread.json").exists():
        r = json.load(open(p / "spread.json"))["rows"]
        labs = sorted({x["label"] for x in r}, key=[x["label"] for x in r].index)
        for pair in ("own", "bacc"):
            rr = [x for x in r if x["pair"] == pair]
            med = {l: np.mean([x["S_pct"][2] for x in rr if x["label"] == l]) for l in labs}
            p5 = {l: np.mean([x["S_pct"][0] for x in rr if x["label"] == l]) for l in labs}
            p95 = {l: np.mean([x["S_pct"][4] for x in rr if x["label"] == l]) for l in labs}
            print(f"SPREAD {pair}: n median {np.median([x['n_median'] for x in rr]):.0f}; S median per label: " +
                  " ".join(f"{l}:{med[l]:.1f}[{p5[l]:.1f}-{p95[l]:.1f}]" for l in labs))
            print(f"   soft fitted S per label: " + " ".join(f"{l}:{np.mean([x['soft_S_fit'] for x in rr if x['label']==l]):.1f}" for l in labs) +
                  f"   corr(S, m) mean {np.nanmean([x['corr_S_m'] for x in rr]):.2f}")
            nll = {k: per_seed(rr, lambda x: True, lambda x, k=k: x["nll"][k]) for k in rr[0]["nll"]}
            print("   vote NLL " + "  ".join(f"{k} {np.mean(v):.4f}" for k, v in nll.items()) +
                  f" | b2-soft_fitS {pstar(welch(nll['b2'], nll['soft_fitS']))} b2-constS {pstar(welch(nll['b2'], nll['b2_constS']))}"
                  f" constS-soft_fitS {pstar(welch(nll['b2_constS'], nll['soft_fitS']))}")
            gain = np.mean(nll["soft_binom"]) - np.mean(nll["b2"])
            if gain > 0:
                print(f"   share of B2's NLL gain over soft-Binomial recovered by soft+1 fitted S: {(np.mean(nll['soft_binom']) - np.mean(nll['soft_fitS'])) / gain:.2f}")
            g = {k: np.mean([x["groups"][k] for x in rr], 0) for k in rr[0]["groups"]}
            print("   vote groups (none/minority/majority/all): " + "  ".join(f"{k} {np.round(v, 3).tolist()}" for k, v in g.items()))
            a = {k: per_seed(rr, lambda x: True, lambda x, k=k: x["auroc_disagree"][k]) for k in rr[0]["auroc_disagree"]}
            print("   disagree AUROC " + "  ".join(f"{k} {ms(v)}" for k, v in a.items()) +
                  f" | b2-b2_mean {pstar(welch(a['b2'], a['b2_mean']))} b2_mean-soft {pstar(welch(a['b2_mean'], a['soft']))}")
