"""Does B²'s spread (S = a0 + a1) carry information? Per dataset, label, seed, checkpoint pair, on test.
- S percentiles (5/25/50/75/95) per label
- test vote NLL: B² as trained; B² with one constant S (median val S); soft BCE as Binomial; soft BCE + one
  concentration per label fitted on val (Beta-Binomial with mean p)
- predicted vs observed share of images per vote group: none (k=0), minority (0<k<=n/2), majority (n/2<k<n), all (k=n)
- disagreement AUROC (disputed 0<k<n vs unanimous): B² full P(disagree); B² mean only (Binomial(n, m)); soft BCE
"""
import json
from pathlib import Path
import numpy as np
from scipy import optimize, special, stats
from sklearn.metrics import roc_auc_score
from b2j.e1 import SEEDS, load_outputs
from b2j.e2 import p_disagree

PAIRS = {"own": {"b2": "nll", "soft_bce": "softbce"}, "bacc": {"b2": "bacc", "soft_bce": "bacc"}}
EPS = 1e-9


def bb_nll(k, n, m, S):
    m = np.clip(m, EPS, 1 - EPS)
    return float(-stats.betabinom.logpmf(k, n, m * S, (1 - m) * S).mean())


def fit_S(k, n, m):
    r = optimize.minimize_scalar(lambda ls: bb_nll(k, n, m, np.exp(ls)), bounds=(-6, 10), method="bounded")
    return float(np.exp(r.x))


def groups(pmf_fn, n):
    """mean predicted probability of none / minority / majority / all, given pmf_fn(j) -> [N]."""
    g = np.zeros((len(n), 4))
    for j in range(int(n.max()) + 1):
        pj = pmf_fn(j) * (j <= n)
        g[:, 0] += pj * (j == 0)
        g[:, 1] += pj * ((j > 0) & (j <= n / 2))
        g[:, 2] += pj * ((j > n / 2) & (j < n))
        g[:, 3] += pj * ((j == n) & (j > 0))
    return g.mean(0).tolist()


def obs_groups(k, n):
    return [float((k == 0).mean()), float(((k > 0) & (k <= n / 2)).mean()), float(((k > n / 2) & (k < n)).mean()),
            float(((k == n) & (k > 0)).mean())]


def run(dataset, root):
    rows = []
    for pair, ck in PAIRS.items():
        for seed in SEEDS:
            b, labels = load_outputs(dataset, "b2", seed, ck["b2"], root)
            s, _ = load_outputs(dataset, "soft_bce", seed, ck["soft_bce"], root)
            for j, lab in enumerate(labels):
                (bv, kv, nv), (bt, kt, nt) = ((o[:, j], k[:, j], n[:, j]) for o, k, n in (b["val"], b["test"]))
                sv, st = s["val"][0][:, j, 0], s["test"][0][:, j, 0]
                a0, a1 = bt[:, 0], bt[:, 1]
                St, mt = a0 + a1, a1 / (a0 + a1)
                Sv = bv[:, 0] + bv[:, 1]
                pt, pv = special.expit(st), special.expit(sv)
                Sg = fit_S(kv, nv, pv)
                Sc = float(np.median(Sv))
                dis = ((kt > 0) & (kt < nt)).astype(int)
                both = 0 < dis.sum() < len(dis)
                auc = lambda sc: float(roc_auc_score(dis, sc)) if both else float("nan")
                logit_m = np.log(np.clip(mt, EPS, 1)) - np.log(np.clip(1 - mt, EPS, 1))
                rows.append({
                    "pair": pair, "seed": seed, "label": lab, "n_median": float(np.median(nt)),
                    "S_pct": np.percentile(St, [5, 25, 50, 75, 95]).tolist(), "S_const": Sc, "soft_S_fit": Sg,
                    "nll": {"b2": float(-stats.betabinom.logpmf(kt, nt, a1, a0).mean()),
                            "b2_constS": bb_nll(kt, nt, mt, Sc),
                            "soft_binom": float(-stats.binom.logpmf(kt, nt, pt).mean()),
                            "soft_fitS": bb_nll(kt, nt, pt, Sg)},
                    "groups": {"observed": obs_groups(kt, nt),
                               "b2": groups(lambda q: stats.betabinom.pmf(q, nt, a1, a0), nt),
                               "soft_binom": groups(lambda q: stats.binom.pmf(q, nt, pt), nt),
                               "soft_fitS": groups(lambda q: stats.betabinom.pmf(q, nt, np.clip(pt, EPS, 1) * Sg,
                                                                                 np.clip(1 - pt, EPS, 1) * Sg), nt)},
                    "auroc_disagree": {"b2": auc(p_disagree("b2", bt, nt)),
                                       "b2_mean": auc(p_disagree("soft_bce", logit_m[:, None], nt)),
                                       "soft": auc(p_disagree("soft_bce", st[:, None], nt))},
                    "corr_S_m": float(stats.spearmanr(St, mt).correlation)})
    return rows, labels


def main(dataset, root):
    rows, labels = run(dataset, root)
    d = Path(root) / "phase2" / dataset
    d.mkdir(parents=True, exist_ok=True)
    (d / "spread.json").write_text(json.dumps({"dataset": dataset, "labels": labels, "rows": rows}, indent=1))
    for pair in PAIRS:
        r = [x for x in rows if x["pair"] == pair]
        print(f"{pair}: median S per label (seed 1):",
              {x["label"]: round(x["S_pct"][2], 2) for x in r if x["seed"] == 1})
        for key in r[0]["nll"]:
            print(f"  vote NLL {key:11} {np.mean([x['nll'][key] for x in r]):.4f}")
