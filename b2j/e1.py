"""Phase 2, E1: a cut-off fitted on validation, reported on test on all / unanimous / disputed frames (PLAN §1.12.1).

  python -m b2j.e1 <dataset>        reads results/final*/<dataset>/..., writes results/phase2/<dataset>/e1.json

Methods (one score per frame and label, from the saved val/test outputs of each final run):
  hard_bce, soft_bce   p = sigmoid(logit)
  b2_b0                mean of the predicted Beta, a1 / (a0 + a1)
  b2_b1                P(more than half of n new raters say yes) under the predicted Beta-Binomial (n = the frame's n)
  b2_b2                P(theta > 0.5) under the predicted Beta(a1, a0)
Per label, method, seed and checkpoint set, one cut-off is fitted on all validation frames: the candidate (midpoint of
each gap between neighbouring distinct validation scores, plus one below and one above all) with the highest BACC
against the majority vote 1[k > n/2]; frames with score >= cut-off are positive. Ties: the middle of the tied
candidates sorted by cut-off (the lower of the two middle ones for an even number). Checkpoint sets: E1a = best val
BACC for every loss; E1b = own loss (B² "nll", soft BCE "softbce", hard BCE "hardbce").
Only results are saved; the fits are recomputed from the outputs in seconds.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
SEEDS = (1, 2, 3, 4, 5)
METHODS = {"hard_bce": "hard_bce", "soft_bce": "soft_bce", "b2_b0": "b2", "b2_b1": "b2", "b2_b2": "b2"}  # -> loss
CKPTS = {"E1a": {"hard_bce": "bacc", "soft_bce": "bacc", "b2": "bacc"},
         "E1b": {"hard_bce": "hardbce", "soft_bce": "softbce", "b2": "nll"}}
RUN_DIR = {"hard_bce": "final_hardbce", "soft_bce": "final", "b2": "final"}   # hard BCE re-run with its own-loss ckpt
SUBSETS = ("all", "unanimous", "disputed")
METRICS = ("bacc", "sensitivity", "specificity", "f1", "precision", "auroc", "ap")


def score(method, out, n):
    """out: [N, 1] logits (BCE) or [N, 2] = (alpha0, alpha1) (B²); n: [N] raters. Returns the score [N], float64."""
    out = np.asarray(out, dtype=np.float64)
    if method in ("hard_bce", "soft_bce"):
        return 1.0 / (1.0 + np.exp(-out[:, 0]))
    a0, a1 = out[:, 0], out[:, 1]
    if method == "b2_b0":
        return a1 / (a0 + a1)
    if method == "b2_b1":
        return stats.betabinom.sf(np.floor(n / 2), n, a1, a0)           # P(k > n/2)
    if method == "b2_b2":
        return stats.beta.sf(0.5, a1, a0)
    raise ValueError(method)


def _div(a, b):
    return a / b if b > 0 else float("nan")


def bacc_at(s, y, t):
    pred = s >= t
    return (_div((pred & (y == 1)).sum(), (y == 1).sum()) + _div((~pred & (y == 0)).sum(), (y == 0).sum())) / 2


def fit_cutoff(s, y):
    """Cut-off with the highest BACC on (s, y); exact search over the gaps between distinct scores."""
    u = np.unique(s)
    cand = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2, [u[-1] + 1.0]])
    # BACC for every candidate at once: positives / negatives with score >= candidate
    order = np.sort(s[y == 1]), np.sort(s[y == 0])
    tp = len(order[0]) - np.searchsorted(order[0], cand, side="left")
    fp = len(order[1]) - np.searchsorted(order[1], cand, side="left")
    bacc = (tp / max(len(order[0]), 1) + (len(order[1]) - fp) / max(len(order[1]), 1)) / 2
    best = np.flatnonzero(bacc == bacc.max())
    return float(cand[best[(len(best) - 1) // 2]]), float(bacc.max()), int(len(best))


def metrics_at(s, y, t):
    pred = s >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    sens, spec, prec = _div(tp, tp + fn), _div(tn, tn + fp), _div(tp, tp + fp)
    f1 = float("nan") if np.isnan(prec) or np.isnan(sens) else _div(2 * tp, 2 * tp + fp + fn)
    both = 0 < y.sum() < len(y)
    return {"bacc": (sens + spec) / 2, "sensitivity": sens, "specificity": spec, "f1": f1, "precision": prec,
            "auroc": float(roc_auc_score(y, s)) if both else float("nan"),
            "ap": float(average_precision_score(y, s)) if both else float("nan"),
            "n": int(len(y)), "n_pos": int(y.sum())}


def load_outputs(dataset, loss, seed, ckpt, root=RESULTS):
    """{split: (out [N, K, w], k [N, K], n [N, K])} and the label names, from outputs_<ckpt>.json."""
    f = Path(root) / RUN_DIR[loss] / dataset / loss / f"seed{seed}" / f"outputs_{ckpt}.json"
    d = json.loads(f.read_text())
    res = {}
    for split in ("val", "test"):
        im = [i for i in d["images"] if i["split"] == split]
        res[split] = tuple(np.array([i[key] for i in im], dtype=np.float64) for key in ("out", "k", "n"))
    return res, d["labels"]


def run(dataset, root=RESULTS, seeds=SEEDS):
    rows = []
    for exp, ck in CKPTS.items():
        for method, loss in METHODS.items():
            for seed in seeds:
                data, labels = load_outputs(dataset, loss, seed, ck[loss], root)
                for j, lab in enumerate(labels):
                    (ov, kv, nv), (ot, kt, nt) = ((o[:, j], k[:, j], n[:, j]) for o, k, n in
                                                  (data["val"], data["test"]))
                    sv, st = score(method, ov, nv), score(method, ot, nt)
                    t, val_bacc, n_tied = fit_cutoff(sv, (kv > nv / 2).astype(int))
                    yt = (kt > nt / 2).astype(int)
                    unanimous = (kt == 0) | (kt == nt)
                    masks = {"all": np.ones_like(yt, dtype=bool), "unanimous": unanimous, "disputed": ~unanimous}
                    rows.append({"exp": exp, "method": method, "checkpoint": ck[loss], "seed": seed, "label": lab,
                                 "cutoff": t, "val_bacc": val_bacc, "n_tied": n_tied,
                                 "test": {sub: metrics_at(st[m], yt[m], t) for sub, m in masks.items()}})
    return rows, labels


def summary(rows, labels):
    """mean ± sd over seeds, per exp / method / subset / label and macro (plain mean over labels per seed)."""
    out = {}
    for exp in CKPTS:
        for method in METHODS:
            r = [x for x in rows if x["exp"] == exp and x["method"] == method]
            for sub in SUBSETS:
                for lab in list(labels) + ["macro"]:
                    vals = {}
                    for m in METRICS + ("cutoff",):
                        per_seed = []
                        for seed in sorted({x["seed"] for x in r}):
                            xs = [x for x in r if x["seed"] == seed and (lab == "macro" or x["label"] == lab)]
                            v = [x["cutoff"] if m == "cutoff" else x["test"][sub][m] for x in xs]
                            per_seed.append(float(np.mean(v)))
                        vals[m] = [float(np.mean(per_seed)), float(np.std(per_seed, ddof=1)) if len(per_seed) > 1
                                   else float("nan")]
                    out[f"{exp}|{method}|{sub}|{lab}"] = vals
    return out


def print_table(summ, labels):
    for exp in CKPTS:
        for sub in SUBSETS:
            print(f"\n== {exp} · {sub} frames (test, mean ± sd over seeds)")
            print(f"{'label':6} {'method':9} " + " ".join(f"{m:>12}" for m in METRICS + ("cutoff",)))
            for lab in list(labels) + ["macro"]:
                for method in METHODS:
                    v = summ[f"{exp}|{method}|{sub}|{lab}"]
                    print(f"{lab:6} {method:9} " + " ".join(f"{v[m][0]:6.3f}±{v[m][1]:5.3f}" for m in METRICS + ("cutoff",)))


def main(dataset, root=RESULTS):
    rows, labels = run(dataset, root)
    summ = summary(rows, labels)
    d = Path(root) / "phase2" / dataset
    d.mkdir(parents=True, exist_ok=True)
    (d / "e1.json").write_text(json.dumps({"dataset": dataset, "labels": labels, "rows": rows, "summary": summ},
                                          indent=1))
    print_table(summ, labels)
    tied = [x for x in rows if x["n_tied"] > 1]
    print(f"\n{len(rows)} fits; {len(tied)} with tied best cut-offs (max {max([x['n_tied'] for x in rows])})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    main(ap.parse_args().dataset)
