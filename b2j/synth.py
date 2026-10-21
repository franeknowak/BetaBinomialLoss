"""Round D, D7: synthetic study (journal_plan/iterations.md). When does B² beat soft BCE, on the mean and on the spread?

World (adapted from the user's earlier test-bench generator): a latent concept u ~ N(0, I_16); features
z = U u + σ_z ε (D = 768, the DINOv3 ViT-B width), what the head sees; a score direction a and an orthogonal clarity
direction d. Mean vote share μ(u) = sigmoid((aᵀu + b) / τ). Clarity drives the per-image spread:
log S(u) = log S̄ + γ dᵀu (γ = 0: one spread for every image). Each image's true vote share θ_i (the part the features
do not explain) and its votes:
  beta         θ_i ~ Beta(μ S, (1 − μ) S), k ~ Binomial(n_i, θ_i)              (B²'s model, correctly specified)
  logitnormal  θ_i = sigmoid(logit μ + σ ε), σ² = ψ'(μ S) + ψ'((1 − μ) S)      (same logit variance, not Beta)
  raters       θ_i as in beta; a fixed panel of 3 raters with biases β_r: vote_r ~ Bernoulli(sigmoid(logit θ_i + β_r))
n_i is fixed or drawn uniformly per image. The real trainer (train.run, frozen-head path) fits soft BCE, B²-r,
vote-count weighted soft BCE and a reference soft BCE trained on 200 votes per image (≈ the hidden θ_i).

Scored against the truth on held-out images ("eval"; synthetic, so the real-data test rule does not apply):
mean — MSE(m, ν(u)), ν(u) = E[k/n | u] (the best any model can do from features), and MSE(m, θ_i); spread — Spearman
of the predicted S with the clarity dᵀu; vote NLL on fresh panels (n_eval = 3, 10, 50; the 3 raters in the raters
world) minus the oracle's (true distribution of k given u); AUROC of m for θ_i > 0.5.

  python -m b2j.synth run --task $SGE_TASK_ID --ntasks N      -> results/journal/synth/<setting>/<model>__seed<s>/
  python -m b2j.synth report                                  -> results/journal/synth/synth_report.md, synth_long.csv
"""
import argparse
import json
import sys
import time
import traceback
import zlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.special import expit, logit, polygamma
from scipy.stats import binom, spearmanr

from b2j import posthoc as P
from b2j.metrics import auroc
from b2j.runs import RESULTS

OUT = RESULTS / "synth"
LATENT, DIM, SIGMA_Z = 16, 768, 0.7
WORLD_SEED = 2026                      # U, a, d and the rater biases are fixed: one world, many samples
N_EVAL, N_DRAWS = 5000, 200            # held-out images; Monte Carlo draws of θ per image for ν(u) and the oracle
N_EVAL_PANELS = (3, 10, 50)
RATER_BIAS = (-1.0, 0.0, 1.0)
SEEDS = range(5)
REF_VOTES = 200                        # the reference model's votes per image (≈ θ_i itself)


@dataclass(frozen=True)
class Setting:
    name: str
    n: int = 3                         # votes per training / validation image (fixed)
    n_range: tuple | None = None       # (lo, hi): n_i uniform on lo..hi instead
    n_train: int = 6000
    gamma: float = 1.0                 # clarity -> log S slope; 0 = one spread for every image
    world: str = "beta"                # beta | logitnormal | raters
    b: float = -0.5
    tau: float = 0.4                   # with S̄ = 5: 29% split 3-vote panels, as Endoscapes (27%) and AMi-Br (22%)
    log_s_bar: float = float(np.log(5.0))
    weighted_baseline: bool = False    # add soft_bce_n (variable-n sweep)


def settings():
    out = []
    for g in (0.0, 1.0):                                                     # S1: votes per image
        out += [Setting(f"S1_n{n}_g{g:g}", n=n, gamma=g) for n in (1, 3, 5, 10, 20, 50)]
    out += [Setting(f"S2_N{N}_n{n}", n=n, n_train=N) for N in (1500, 24000) for n in (3, 10)]   # S2: images
    for g in (0.0, 1.0):                                                     # S3: variable vs fixed n
        out += [Setting(f"S3_var{lo}-{hi}_g{g:g}", n_range=(lo, hi), gamma=g, weighted_baseline=True)
                for lo, hi in ((3, 20), (4, 10))]
        out += [Setting(f"S3_fixed{n}_g{g:g}", n=n, gamma=g, weighted_baseline=True) for n in (11, 7)]
    out += [Setting("S4_logitnormal_n3", world="logitnormal"), Setting("S4_logitnormal_n10", n=10, world="logitnormal"),
            Setting("S4_raters_n3", world="raters")]                         # S4: misspecified worlds
    return out


def models(s: Setting):
    return ("soft_bce", "b2r", "ref200") + (("soft_bce_n",) if s.weighted_baseline else ())


# ---------------------------------------------------------------- the world
def _fixed():
    rng = np.random.default_rng(WORLD_SEED)
    U = rng.normal(size=(DIM, LATENT)) / np.sqrt(LATENT)
    a = rng.normal(size=LATENT)
    a /= np.linalg.norm(a)
    d = rng.normal(size=LATENT)
    d -= (d @ a) * a                                  # clarity ⟂ score: how ambiguous ≠ how positive
    d /= np.linalg.norm(d)
    return U, a, d


def _theta_draws(s: Setting, mu, S, rng, size):
    """size draws of the per-image vote share given u (rater-level shares in the raters world): [N, size]."""
    if s.world == "logitnormal":
        sig = np.sqrt(polygamma(1, mu * S) + polygamma(1, (1 - mu) * S))
        return expit(logit(mu)[:, None] + sig[:, None] * rng.normal(size=(len(mu), size)))
    return rng.beta(mu[:, None] * S[:, None], (1 - mu[:, None]) * S[:, None], size=(len(mu), size))


def _rater_p(theta):
    """[..., 3]: each rater's probability of a yes vote given θ."""
    return expit(logit(np.clip(theta, 1e-9, 1 - 1e-9))[..., None] + np.asarray(RATER_BIAS))


def _pb3(p):
    """Poisson-binomial pmf of the 3 raters' yes count: [..., 4] from p [..., 3]."""
    f = np.ones(p.shape[:-1] + (1,))
    for r in range(3):
        q = p[..., r:r + 1]
        f = np.concatenate([f * (1 - q), np.zeros_like(q)], -1) + np.concatenate([np.zeros_like(q), f * q], -1)
    return f


def sample(s: Setting, N, rng, n_votes=None):
    """N images: features, truths and votes (n_votes overrides the setting's n)."""
    U, a, d = _fixed()
    u = rng.normal(size=(N, LATENT))
    z = (u @ U.T + SIGMA_Z * rng.normal(size=(N, DIM))).astype(np.float32)
    mu = expit((u @ a + s.b) / s.tau).clip(1e-6, 1 - 1e-6)
    clar = u @ d
    S = np.exp(s.log_s_bar + s.gamma * clar)
    theta = _theta_draws(s, mu, S, rng, 1)[:, 0]
    if s.world == "raters":
        votes = rng.random((N, 3)) < _rater_p(theta)
        n = np.full(N, 3.0)
        k = votes.sum(1).astype(float)
        theta_eff = _rater_p(theta).mean(-1)
    else:
        if n_votes is not None:
            n = np.full(N, float(n_votes))
        elif s.n_range is not None:
            n = rng.integers(s.n_range[0], s.n_range[1] + 1, size=N).astype(float)
        else:
            n = np.full(N, float(s.n))
        k = rng.binomial(n.astype(int), theta).astype(float)
        theta_eff = theta
    return {"z": z, "mu": mu, "S": S, "clarity": clar, "theta": theta_eff, "theta_raw": theta, "k": k, "n": n}


def draw_split(s: Setting, seed):
    """Training, validation and evaluation images of one seed (the seed redraws the data, not the world)."""
    rng = np.random.default_rng([seed, zlib.crc32(s.name.encode())])
    tr = sample(s, s.n_train, rng)
    va = sample(s, max(500, s.n_train // 3), rng)
    ev = sample(s, N_EVAL, rng)
    return tr, va, ev


def truths(s: Setting, ev, seed):
    """ν(u) = E[k/n | u] and the oracle vote pmf on fresh panels, by Monte Carlo over θ given u (exact in beta)."""
    rng = np.random.default_rng([seed, 7, zlib.crc32(s.name.encode())])
    mu, S = ev["mu"], ev["S"]
    th = _theta_draws(s, mu, S, rng, N_DRAWS)
    panels, oracle_nll = {}, {}
    if s.world == "raters":
        p = _rater_p(th)                                           # [N, draws, 3]
        nu = p.mean((1, 2))
        pr = _rater_p(ev["theta_raw"])
        k3 = (rng.random(pr.shape) < pr).sum(-1).astype(float)
        pmf = _pb3(p).mean(1)                                      # [N, 4]
        panels[3] = k3
        oracle_nll[3] = -np.log(np.take_along_axis(pmf, k3.astype(int)[:, None], 1)[:, 0])
    else:
        nu = mu if s.world == "beta" else th.mean(1)
        for ne in N_EVAL_PANELS:
            k = rng.binomial(ne, ev["theta"]).astype(float)
            panels[ne] = k
            if s.world == "beta":
                oracle_nll[ne] = P.vote_nll(mu, S, k, np.full_like(k, ne))
            else:
                lp = binom.logpmf(k[:, None], ne, th)
                mx = lp.max(1, keepdims=True)
                oracle_nll[ne] = -(mx[:, 0] + np.log(np.exp(lp - mx).mean(1)))
    return nu, panels, oracle_nll


# ---------------------------------------------------------------- training
def run_dir(s: Setting, model, seed, root=OUT):
    return Path(root) / s.name / f"{model}__seed{seed}"


def specs():
    return [(s, m, seed) for s in settings() for seed in SEEDS for m in models(s)]


def train_one(s: Setting, model, seed, root=OUT, device="cpu"):
    import torch
    from b2j import train as T
    torch.set_num_threads(2)
    d = run_dir(s, model, seed, root)
    if (d / "summary.json").exists():
        return "skipped"
    tr, va, ev = draw_split(s, seed)
    if model == "ref200":                                           # ≈ the hidden θ_i: 200 votes per image
        r = np.random.default_rng([seed, 200])
        for x in (tr, va):
            x["n"] = np.full_like(x["n"], REF_VOTES)
            x["k"] = r.binomial(REF_VOTES, x["theta"]).astype(float)
    variant = "soft_bce" if model == "ref200" else model
    ids = lambda N, p: np.array([f"{p}{i}" for i in range(N)])
    sp = lambda x, p, lab=True: T.Split(x["z"], ids(len(x["z"]), p), x["k"][:, None] if lab else None,
                                        x["n"][:, None] if lab else None)
    cfg = T.TrainConfig(variant, seed=seed, num_workers=0)
    T.run(cfg, sp(tr, "tr"), sp(va, "va"), sp(ev, "ev", False), d, device=device, verbose=False)
    (d / "setting.json").write_text(json.dumps({**asdict(s), "model": model, "seed": seed}))
    return "done"


# ---------------------------------------------------------------- scoring against the truth
def score(s: Setting, seed, root=OUT):
    """Rows (one per model and checkpoint) for one setting and seed."""
    _, va, ev = draw_split(s, seed)
    nu, panels, oracle = truths(s, ev, seed)
    rows = []
    for model in models(s):
        d = run_dir(s, model, seed, root)
        if not (d / "summary.json").exists():
            continue
        variant = "soft_bce" if model == "ref200" else model
        for ck in "AB":
            z = np.load(d / f"test_{ck}.npz")
            out = z["out"]
            m, S = P.panel(variant, out, z["scale"])
            m, S = m[:, 0], S[:, 0]
            r = {"setting": s.name, "model": model, "seed": seed, "checkpoint": ck,
                 "mse_nu": float(((m - nu) ** 2).mean()), "mse_theta": float(((m - ev["theta"]) ** 2).mean()),
                 "auroc": auroc(m, (ev["theta"] > 0.5).astype(float)),
                 "spearman_S_clarity": (float(spearmanr(S, ev["clarity"])[0])
                                        if model == "b2r" and np.std(S) > 0 else np.nan)}
            for ne, k in panels.items():
                r[f"excess_nll_n{ne}"] = float((P.vote_nll(m, S, k, np.full_like(k, ne)) - oracle[ne]).mean())
            rows.append(r)
    return rows


def report(root=OUT):
    import pandas as pd
    rows = []
    for s in settings():
        for seed in SEEDS:
            rows += score(s, seed, root)
    df = pd.DataFrame(rows)
    df.to_csv(Path(root) / "synth_long.csv", index=False)
    md = ["# D7 — synthetic study (validation-free: scored against the known truth)", "",
          "Mean recovery: MSE(m, ν) with ν = E[k/n | features]; Δ = MSE(soft BCE) − MSE(B²-r) (positive = B²-r closer), "
          "mean ± sd over 5 seeds (paired: same data per seed), and the share of soft BCE's few-votes penalty "
          "(its MSE − the 200-vote reference's) that B²-r recovers. Spread: Spearman of B²-r's S with the true "
          "clarity (which drives S when γ > 0). Vote NLL: excess over the oracle on fresh panels, Global-S vs B²-r.",
          ""]
    for ck in ("B", "A"):
        t = df[df.checkpoint == ck]
        md += [f"## checkpoint {ck}", "",
               "| setting | MSE soft BCE | MSE B²-r | Δ (± sd) | B² better (seeds) | share of penalty recovered | "
               "MSE 200-vote ref | Spearman S~clarity | excess NLL n=3 GS / B² | n=10 GS / B² | n=50 GS / B² | "
               "AUROC soft / B² | MSE vote-weighted |",
               "|" + "---|" * 13]
        for s in settings():
            g = t[t.setting == s.name].pivot(index="seed", columns="model")
            if g.empty or "b2r" not in g["mse_nu"] or "soft_bce" not in g["mse_nu"]:
                continue
            dlt = g["mse_nu"]["soft_bce"] - g["mse_nu"]["b2r"]
            pen = g["mse_nu"]["soft_bce"].mean() - g["mse_nu"]["ref200"].mean() if "ref200" in g["mse_nu"] else np.nan
            nll = lambda ne: (f"{g[f'excess_nll_n{ne}']['soft_bce'].mean():.4f} / {g[f'excess_nll_n{ne}']['b2r'].mean():.4f}"
                              if f"excess_nll_n{ne}" in g and g[f"excess_nll_n{ne}"]["b2r"].notna().any() else "–")
            wb = f"{g['mse_nu']['soft_bce_n'].mean():.5f}" if "soft_bce_n" in g["mse_nu"] else "–"
            md.append(f"| {s.name} | {g['mse_nu']['soft_bce'].mean():.5f} | {g['mse_nu']['b2r'].mean():.5f} | "
                      f"{dlt.mean():+.5f} ± {dlt.std():.5f} | {(dlt > 0).sum()}/{len(dlt)} | "
                      f"{dlt.mean() / pen:+.0%} | "
                      f"{g['mse_nu']['ref200'].mean() if 'ref200' in g['mse_nu'] else np.nan:.5f} | "
                      f"{g['spearman_S_clarity']['b2r'].mean():+.2f} | {nll(3)} | {nll(10)} | {nll(50)} | "
                      f"{g['auroc']['soft_bce'].mean():.3f} / {g['auroc']['b2r'].mean():.3f} | {wb} |")
        md.append("")
    out = Path(root) / "synth_report.md"
    out.write_text("\n".join(md))
    return out


def describe(s: Setting, N=20000, seed=0):
    """Vote histogram and spread of the world at the setting's n (for calibration against real datasets)."""
    x = sample(s, N, np.random.default_rng(seed))
    n = x["n"].astype(int)
    split = ((x["k"] > 0) & (x["k"] < n)).mean()
    return {"prevalence_majority": float((2 * x["k"] > n).mean()), "split_share": float(split),
            "S_p10_p90": np.quantile(x["S"], [0.1, 0.9]).round(2).tolist(),
            "theta_hist": np.histogram(x["theta"], bins=5, range=(0, 1))[0].tolist()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "report", "describe", "list"))
    ap.add_argument("--task", type=int, default=1)
    ap.add_argument("--ntasks", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "list":
        print(len(specs()), "runs in", len(settings()), "settings")
    elif a.cmd == "describe":
        for s in settings():
            print(s.name, describe(s))
    elif a.cmd == "report":
        print(report().read_text())
    else:
        todo = specs()[a.task - 1::a.ntasks]
        failed = []
        for i, (s, m, seed) in enumerate(todo):
            t0 = time.time()
            try:
                st = train_one(s, m, seed)
                print(f"[{i + 1}/{len(todo)}] {s.name} {m} seed{seed}: {st} in {time.time() - t0:.0f}s", flush=True)
            except Exception:
                traceback.print_exc()
                failed.append((s.name, m, seed))
        print(f"finished: {len(todo) - len(failed)} ok, {len(failed)} failed {failed}", flush=True)
        sys.exit(1 if failed else 0)
