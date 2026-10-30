"""Phase 2, E4: agreement between the three losses, per frame and vote pattern (PLAN §1.12.1).

  python -m b2j.e4 <dataset>        reads results/final*/<dataset>/..., writes results/phase2/<dataset>/e4.json and
                                    e4_frames.csv (one row per test frame and label, for inspecting images)

One model per loss = the mean of its 5 seeds' E1 scores (hard_bce, soft_bce: p; B²: b2_b1, P(k > n/2) under the
predicted Beta-Binomial), on validation and test. Per label and loss, one cut-off is fitted on the averaged
validation scores exactly as in E1 (highest BACC against the majority vote, same search and tie rule).
Reported on test: the E1 metrics on all / unanimous / disputed frames; per vote pattern k/n, the number of frames in
each of the 8 right/wrong combinations of the three losses (all right, only one right, only one wrong, all wrong);
and the share of frames on which each pair of losses makes the same call.
Checkpoint sets: E4a = best val BACC, E4b = own loss (same mapping as E1a / E1b).
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from b2j.e1 import CKPTS as E1_CKPTS, RESULTS, RUN_DIR, SEEDS, fit_cutoff, metrics_at, score

MODELS = {"hard_bce": "hard_bce", "soft_bce": "soft_bce", "b2_b1": "b2"}       # E1 score -> loss
CKPTS = {"E4a": E1_CKPTS["E1a"], "E4b": E1_CKPTS["E1b"]}
SUBSETS = ("all", "unanimous", "disputed")


def load_ids(dataset, loss, seed, ckpt, root=RESULTS):
    """Per split: frame ids, outputs [N, K, w], k [N, K], n [N, K] from outputs_<ckpt>.json."""
    f = Path(root) / RUN_DIR[loss] / dataset / loss / f"seed{seed}" / f"outputs_{ckpt}.json"
    d = json.loads(f.read_text())
    res = {}
    for split in ("val", "test"):
        im = [i for i in d["images"] if i["split"] == split]
        res[split] = ([i["id"] for i in im],) + tuple(np.array([i[key] for i in im], dtype=np.float64)
                                                       for key in ("out", "k", "n"))
    return res, d["labels"]


def averaged_scores(dataset, method, ckpt, root=RESULTS, seeds=SEEDS):
    """{split: (ids, mean score over seeds [N, K], k, n)}; checks every seed has the same frames in the same order."""
    loss = MODELS[method]
    acc, ref = {}, None
    for seed in seeds:
        data, labels = load_ids(dataset, loss, seed, ckpt, root)
        for split, (ids, out, k, n) in data.items():
            s = np.stack([score(method, out[:, j], n[:, j]) for j in range(len(labels))], 1)
            if split not in acc:
                acc[split] = [ids, s / len(seeds), k, n]
            else:
                if ids != acc[split][0] or not (np.array_equal(k, acc[split][2]) and np.array_equal(n, acc[split][3])):
                    raise ValueError(f"seed {seed}: frames differ from seed {seeds[0]} ({split})")
                acc[split][1] = acc[split][1] + s / len(seeds)
        ref = labels if ref is None else ref
    return {sp: tuple(v) for sp, v in acc.items()}, ref


def combo_name(right):
    """right: dict method -> bool. 'all right', 'all wrong', 'only <m> right', 'only <m> wrong'."""
    r = [m for m in MODELS if right[m]]
    if len(r) == len(MODELS):
        return "all right"
    if not r:
        return "all wrong"
    if len(r) == 1:
        return f"only {r[0]} right"
    return f"only {next(m for m in MODELS if not right[m])} wrong"


COMBOS = ["all right"] + [f"only {m} wrong" for m in MODELS] + [f"only {m} right" for m in MODELS] + ["all wrong"]


def run(dataset, root=RESULTS, seeds=SEEDS):
    res, frames = {}, []
    for exp, ck in CKPTS.items():
        avg = {}
        for method, loss in MODELS.items():
            avg[method], labels = averaged_scores(dataset, method, ck[loss], root, seeds)
        ids = avg["hard_bce"]["test"][0]
        if any(avg[m]["test"][0] != ids for m in MODELS):
            raise ValueError("losses differ in their test frames")
        for j, lab in enumerate(labels):
            kt, nt = avg["hard_bce"]["test"][2][:, j], avg["hard_bce"]["test"][3][:, j]
            yt = (kt > nt / 2).astype(int)
            unanimous = (kt == 0) | (kt == nt)
            masks = {"all": np.ones_like(yt, dtype=bool), "unanimous": unanimous, "disputed": ~unanimous}
            calls, entry = {}, {"models": {}}
            for method in MODELS:
                _, sv, kv, nv = avg[method]["val"]
                st = avg[method]["test"][1][:, j]
                t, val_bacc, n_tied = fit_cutoff(sv[:, j], (kv[:, j] > nv[:, j] / 2).astype(int))
                calls[method] = (st >= t, st, t)
                entry["models"][method] = {"checkpoint": ck[MODELS[method]], "cutoff": t, "val_bacc": val_bacc,
                                           "n_tied": n_tied,
                                           "test": {sub: metrics_at(st[m], yt[m], t) for sub, m in masks.items()}}
            right = {m: calls[m][0] == (yt == 1) for m in MODELS}
            pattern = np.array([f"{int(a)}/{int(b)}" for a, b in zip(kt, nt)])
            combo = np.array([combo_name({m: right[m][i] for m in MODELS}) for i in range(len(yt))])
            pats = sorted(set(pattern), key=lambda s: (int(s.split("/")[1]), int(s.split("/")[0])))
            entry["combos"] = {}
            for p in pats + ["all"]:
                sel = np.ones(len(yt), dtype=bool) if p == "all" else pattern == p
                entry["combos"][p] = {"n": int(sel.sum()), **{c: int((sel & (combo == c)).sum()) for c in COMBOS}}
            entry["same_call"] = {f"{a}|{b}": float((calls[a][0] == calls[b][0]).mean())
                                  for a, b in itertools.combinations(MODELS, 2)}
            res[f"{exp}|{lab}"] = entry
            for i, fid in enumerate(ids):
                row = {"exp": exp, "label": lab, "id": fid, "k": int(kt[i]), "n": int(nt[i]), "pattern": pattern[i],
                       "majority": int(yt[i]), "combo": combo[i]}
                for m in MODELS:
                    row[f"{m}_score"], row[f"{m}_cutoff"] = float(calls[m][1][i]), float(calls[m][2])
                    row[f"{m}_call"], row[f"{m}_right"] = int(calls[m][0][i]), int(right[m][i])
                frames.append(row)
    return res, labels, pd.DataFrame(frames)


def print_table(res, labels):
    for exp in CKPTS:
        print(f"\n== {exp}: test metrics of the seed-averaged models (all frames)")
        for lab in labels:
            for m, v in res[f"{exp}|{lab}"]["models"].items():
                t = v["test"]["all"]
                print(f"{lab:4} {m:9} cutoff {v['cutoff']:.3f}  " +
                      "  ".join(f"{k} {t[k]:.3f}" for k in ("bacc", "sensitivity", "specificity", "f1", "precision",
                                                           "auroc", "ap")))
        print(f"\n== {exp}: right/wrong combinations by vote pattern (test frames)")
        print(f"{'label':5} {'pattern':7} {'n':>5} " + " ".join(f"{c[:22]:>22}" for c in COMBOS))
        for lab in labels:
            for p, v in res[f"{exp}|{lab}"]["combos"].items():
                print(f"{lab:5} {p:7} {v['n']:5} " + " ".join(f"{v[c]:22}" for c in COMBOS))


def main(dataset, root=RESULTS):
    res, labels, frames = run(dataset, root)
    d = Path(root) / "phase2" / dataset
    d.mkdir(parents=True, exist_ok=True)
    (d / "e4.json").write_text(json.dumps({"dataset": dataset, "labels": labels, "combos": COMBOS, "results": res},
                                          indent=1))
    frames.to_csv(d / "e4_frames.csv", index=False)
    print_table(res, labels)
    n_tied = [v["n_tied"] for r in res.values() for v in r["models"].values()]
    print(f"\n{len(n_tied)} fits; {sum(t > 1 for t in n_tied)} with tied best cut-offs (max {max(n_tied)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    main(ap.parse_args().dataset)
