"""Phase 2, E2: spotting disagreement — disputed (0 < k < n) vs unanimous (k = 0 or n) frames (PLAN §1.12.1).

  python -m b2j.e2 <dataset>        reads results/final*/<dataset>/..., writes results/phase2/<dataset>/e2.json

Score = the model's probability that n new raters (n = the frame's number of raters) do not all agree,
P(disagree) = sum over 0 < j < n of P(j yes votes):
  hard_bce, soft_bce   Binomial(n, p), p = sigmoid(logit); for n = 3 this is 3p(1 - p)
  b2                   Beta-Binomial(n, a1, a0); for n = 3 this is 3m(1 - m) S / (S + 1), m = a1 / S, S = a0 + a1
Summed in log space from the logits / alphas, so confident frames keep distinct scores.
Per label, method, seed and checkpoint set: one cut-off fitted on all validation frames as in E1 (highest BACC for
disputed vs unanimous, same exact search and tie rule); test metrics as in E1 (BACC, sensitivity, specificity, F1,
precision, AUROC, AP, cut-off) plus, per vote pattern k/n, the share of test frames flagged as disputed.
Checkpoint sets: E2a = best val BACC, E2b = own loss (same mapping as E1a / E1b).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import special, stats

from b2j.e1 import CKPTS as E1_CKPTS, RESULTS, SEEDS, fit_cutoff, load_outputs, metrics_at

METHODS = ("hard_bce", "soft_bce", "b2")
CKPTS = {"E2a": E1_CKPTS["E1a"], "E2b": E1_CKPTS["E1b"]}
METRICS = ("bacc", "sensitivity", "specificity", "f1", "precision", "auroc", "ap")


def p_disagree(method, out, n):
    """out: [N, 1] logits (BCE) or [N, 2] = (alpha0, alpha1) (B²); n: [N]. Returns P(0 < k < n) [N], float64."""
    out, n = np.asarray(out, dtype=np.float64), np.asarray(n, dtype=np.float64)
    res = np.zeros(len(n))
    for j in range(1, int(n.max())):
        m = j < n
        if method == "b2":
            logp = stats.betabinom.logpmf(j, n[m], out[m, 1], out[m, 0])
        else:
            z = out[m, 0]
            log_comb = special.gammaln(n[m] + 1) - special.gammaln(j + 1) - special.gammaln(n[m] - j + 1)
            logp = log_comb + j * -np.logaddexp(0, -z) + (n[m] - j) * -np.logaddexp(0, z)   # log sigmoid(+-z)
        res[m] += np.exp(logp)
    return res


def run(dataset, root=RESULTS, seeds=SEEDS):
    rows = []
    for exp, ck in CKPTS.items():
        for method in METHODS:
            for seed in seeds:
                data, labels = load_outputs(dataset, method, seed, ck[method], root)
                for j, lab in enumerate(labels):
                    (ov, kv, nv), (ot, kt, nt) = ((o[:, j], k[:, j], n[:, j]) for o, k, n in
                                                  (data["val"], data["test"]))
                    sv, st = p_disagree(method, ov, nv), p_disagree(method, ot, nt)
                    t, val_bacc, n_tied = fit_cutoff(sv, ((kv > 0) & (kv < nv)).astype(int))
                    yt = ((kt > 0) & (kt < nt)).astype(int)
                    pattern = np.array([f"{int(a)}/{int(b)}" for a, b in zip(kt, nt)])
                    flagged = {p: {"n": int((pattern == p).sum()), "share_flagged": float((st[pattern == p] >= t).mean())}
                               for p in sorted(set(pattern), key=lambda s: (int(s.split("/")[1]), int(s.split("/")[0])))}
                    rows.append({"exp": exp, "method": method, "checkpoint": ck[method], "seed": seed, "label": lab,
                                 "cutoff": t, "val_bacc": val_bacc, "n_tied": n_tied,
                                 "test": metrics_at(st, yt, t), "by_pattern": flagged})
    return rows, labels


def _ms(v):
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
                vals = {m: _ms([np.mean([x["test"][m] for x in sel(s)]) for s in seeds]) for m in METRICS}
                vals["cutoff"] = _ms([np.mean([x["cutoff"] for x in sel(s)]) for s in seeds])
                if lab != "macro":
                    pats = sel(seeds[0])[0]["by_pattern"].keys()                     # per label: n may differ
                    vals["by_pattern"] = {p: _ms([sel(s)[0]["by_pattern"][p]["share_flagged"] for s in seeds])
                                          for p in pats}
                out[f"{exp}|{method}|{lab}"] = vals
    return out


def print_table(summ, labels):
    for exp in CKPTS:
        print(f"\n== {exp}: disputed vs unanimous (test, mean ± sd over seeds)")
        print(f"{'label':6} {'method':9} " + " ".join(f"{m:>12}" for m in METRICS + ("cutoff",)))
        for lab in list(labels) + ["macro"]:
            for method in METHODS:
                v = summ[f"{exp}|{method}|{lab}"]
                print(f"{lab:6} {method:9} " + " ".join(f"{v[m][0]:6.3f}±{v[m][1]:5.3f}" for m in METRICS + ("cutoff",)))
        print(f"\n-- {exp}: share of test frames flagged as disputed, by vote pattern")
        for lab in labels:
            for method in METHODS:
                bp = summ[f"{exp}|{method}|{lab}"]["by_pattern"]
                print(f"{lab:6} {method:9} " + " ".join(f"{p}: {v[0]:.3f}±{v[1]:.3f}" for p, v in bp.items()))


def main(dataset, root=RESULTS):
    rows, labels = run(dataset, root)
    summ = summary(rows, labels)
    d = Path(root) / "phase2" / dataset
    d.mkdir(parents=True, exist_ok=True)
    (d / "e2.json").write_text(json.dumps({"dataset": dataset, "labels": labels, "rows": rows, "summary": summ},
                                          indent=1))
    print_table(summ, labels)
    print(f"\n{len(rows)} fits; {sum(x['n_tied'] > 1 for x in rows)} with tied best cut-offs "
          f"(max {max(x['n_tied'] for x in rows)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    main(ap.parse_args().dataset)
