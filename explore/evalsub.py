"""Models trained on m=3 raters, scored against the full test panel vs against 3-rater test panels
(20 random draws). Question: does B²'s per-image spread advantage remain visible when the test truth has 3 raters?
  python -m explore.evalsub <dataset>    -> results/explore/sub/n3/phase2/<ds>/evalsub.json
"""
import json
import sys

import numpy as np
from scipy import special, stats
from sklearn.metrics import roc_auc_score
from b2j.e1 import SEEDS, load_outputs
from b2j.e2 import p_disagree
from explore import sub_root
from explore.subsample import subsample

ROOT = sub_root(3)
EPS = 1e-9
DRAWS = 20


def metrics(bt, st, kt, nt, Sc):
    a0, a1 = bt[:, 0], bt[:, 1]
    m = a1 / (a0 + a1)
    dis = ((kt > 0) & (kt < nt)).astype(int)
    if not 0 < dis.sum() < len(dis):
        return None
    lm = np.log(np.clip(m, EPS, 1)) - np.log(np.clip(1 - m, EPS, 1))
    mc = np.clip(m, EPS, 1 - EPS)
    return {"auc_b2": roc_auc_score(dis, p_disagree("b2", bt, nt)),
            "auc_b2_mean": roc_auc_score(dis, p_disagree("soft_bce", lm[:, None], nt)),
            "auc_soft": roc_auc_score(dis, p_disagree("soft_bce", st[:, None], nt)),
            "nll_b2": -stats.betabinom.logpmf(kt, nt, a1, a0).mean(),
            "nll_b2_constS": -stats.betabinom.logpmf(kt, nt, mc * Sc, (1 - mc) * Sc).mean(),
            "nll_soft": -stats.binom.logpmf(kt, nt, special.expit(st)).mean()}


def main(ds):
    rows = []
    for seed in SEEDS:
        b, labels = load_outputs(ds, "b2", seed, "nll", ROOT)
        s, _ = load_outputs(ds, "soft_bce", seed, "softbce", ROOT)
        bt, kt, nt = b["test"]
        Sv = b["val"][0].sum(-1)
        st = s["test"][0][:, :, 0]
        draws = [subsample(kt, nt, 3, np.random.default_rng(seed * 100 + d)) for d in range(DRAWS)]
        for j, lab in enumerate(labels):
            Sc = float(np.median(Sv[:, j]))
            full = metrics(bt[:, j], st[:, j], kt[:, j], nt[:, j], Sc)
            sub = [metrics(bt[:, j], st[:, j], k3[:, j].astype(float), n3[:, j].astype(float), Sc) for k3, n3 in draws]
            sub = [x for x in sub if x is not None]
            rows.append({"seed": seed, "label": lab, "full": full,
                         "three": {k: float(np.mean([x[k] for x in sub])) for k in sub[0]} if sub else None})
    d = ROOT / "phase2" / ds
    d.mkdir(parents=True, exist_ok=True)
    (d / "evalsub.json").write_text(json.dumps(rows, indent=1, default=float))
    for key in ("full", "three"):
        r = [x[key] for x in rows if x[key] is not None]
        print(key, {k: round(float(np.mean([x[k] for x in r])), 4) for k in r[0]})


if __name__ == "__main__":
    main(sys.argv[1])
