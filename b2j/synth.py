"""Round D, D7: synthetic study (journal_plan/iterations.md). When does B² beat soft BCE, on the mean and on the concentration?

World (adapted from the user's earlier test-bench generator): a latent concept u ~ N(0, I_16); features
z = U u + σ_z ε (D = 768, the DINOv3 ViT-B width), what the head sees; a score direction a and an orthogonal clarity
direction d. Mean vote share μ(u) = sigmoid((aᵀu + b) / τ). Clarity drives the per-image concentration:
log S(u) = log S̄ + γ dᵀu (γ = 0: one concentration for every image). Each image's true vote share θ_i (the part the features
do not explain) and its votes:
  beta         θ_i ~ Beta(μ S, (1 − μ) S), k ~ Binomial(n_i, θ_i)              (B²'s model, correctly specified)
  logitnormal  θ_i = sigmoid(c + σ ε), (c, σ) per image such that E θ = μ and Var θ = μ(1 − μ)/(S + 1)
               (the Beta's mean and variance, not Beta; panels of ≤ 2 votes cannot tell the two apart). Fixed
               2026-10-22: the first version centred at logit μ and matched the logit variance, which scrambled the
               mean at extreme μ (runs moved to results/journal/synth_invalid_logitnormal/).
  raters       θ_i as in beta; a pool of 20 raters with personal biases β_r ~ N(0, 1) (Endoscapes-like spread of rater
               rates); each image is labelled by its own 3 raters drawn from the pool: vote_r ~ Bernoulli(sigmoid(logit θ_i + β_r))
               (crowdsourcing-like: CIFAR-10H, FER+)
  raters_fixed θ_i as in beta; the same 3 raters with biases (−1, 0, 1) label every image (Endoscapes, AMi-Br, MHIST)
n_i is fixed or drawn uniformly per image. The real trainer (train.run, frozen-head path) fits soft BCE, B²-r,
vote-count weighted soft BCE and a reference soft BCE trained on 200 votes per image (≈ the hidden θ_i).

Scored against the truth on held-out images ("eval"; synthetic, so the real-data test rule does not apply):
mean — MSE(m, ν(u)), ν(u) = E[k/n | u] (≈ the floor from features; z is noisy, so the exact floor is E[ν | z]), and MSE(m, θ_i); concentration — Spearman
of the predicted S with the clarity dᵀu; vote NLL on fresh panels (n_eval = 3, 10, 50; the 3 raters in the raters
world: fresh panels of 3 and 10 pool raters; the 3 raters in raters_fixed) minus the oracle's (true distribution of k given u; exact or quadrature except the pool world's count oracle, Monte
Carlo over θ and panels with N_DRAWS_POOL draws); concentration ablation — vote NLL with one S fitted on the evaluation panels
minus with the per-image S (the D0b test on real data); AUROC of m for θ_i > 0.5.
In the rater worlds θ_i above is the pool-average rater's share, mean_r sigmoid(logit θ_i + β_r).

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
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import expit, log_expit, logit, logsumexp
from scipy.stats import binom, spearmanr

from b2j import posthoc as P
from b2j.metrics import auroc
from b2j.runs import RESULTS

OUT = RESULTS / "synth"
LATENT, DIM, SIGMA_Z = 16, 768, 0.7
WORLD_SEED = 2026                      # U, a, d and the rater biases are fixed: one world, many samples
N_EVAL, N_DRAWS = 5000, 200            # held-out images; Monte Carlo draws of θ per image for ν(u) (rater worlds)
N_DRAWS_POOL = 4000                    # pool world: draws of (θ, panel) per image for the count oracle (bias ≈ 1/draws)
EPS_GRID = np.linspace(-8.5, 8.5, 401)  # logitnormal: quadrature nodes of ε ~ N(0, 1) (trapezoid)
N_EVAL_PANELS = (3, 10, 50)
RATER_POOL, RATER_SD = 20, 1.0        # raters world: pool size and sd of the biases (logit scale)
RATER_PANELS = (3, 10)                 # raters world: fresh evaluation panels (distinct pool raters)
FIXED_BIAS = (-1.0, 0.0, 1.0)          # raters_fixed world: one panel for every image
SEEDS = range(5)
REF_VOTES = 200                        # the reference model's votes per image (≈ θ_i itself)


@dataclass(frozen=True)
class Setting:
    name: str
    n: int = 3                         # votes per training / validation image (fixed)
    n_range: tuple | None = None       # (lo, hi): n_i uniform on lo..hi instead
    n_train: int = 6000
    gamma: float = 1.0                 # clarity -> log S slope; 0 = one concentration for every image
    world: str = "beta"                # beta | logitnormal | raters | raters_fixed
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
            Setting("S4_raters_n3", world="raters_fixed"), Setting("S4_pool_n3_g1", world="raters"), Setting("S4_pool_n3_g0", world="raters", gamma=0.0)]  # S4
    out += [Setting(f"S5_n{n}_g{g:g}", n=n, gamma=g) for n in (3, 7) for g in GAMMA_SWEEP]   # S5: how much must S vary?
    out += [Setting(f"S5_raters_n3_g{g:g}", world="raters_fixed", gamma=g) for g in GAMMA_SWEEP]
    return out


GAMMA_SWEEP = (0.25, 0.5)   # added 2026-10-22 after the review (γ = 0 and 1 are in S1 / S3 / S4)
SWEEP_ROWS = (("S1_n3_g0", "S5_n3_g0.25", "S5_n3_g0.5", "S1_n3_g1"),
              ("S3_fixed7_g0", "S5_n7_g0.25", "S5_n7_g0.5", "S3_fixed7_g1"),
              ("S5_raters_n3_g0.25", "S5_raters_n3_g0.5", "S4_raters_n3"))


RATER_MODELS = ("soft_bce_ba", "b2rba")    # D8: rater worlds only (they need the rater of each vote)


def models(s: Setting):
    return (("soft_bce", "b2r", "ref200") + (("soft_bce_n",) if s.weighted_baseline else ())
            + (RATER_MODELS if s.world in ("raters", "raters_fixed") else ()))


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


def _ln_moments(c, ls):
    """logit E θ and logit ρ (ρ = Var θ / (E θ (1 − E θ))) of θ = sigmoid(c + e^ls ε), by quadrature; stable when
    saturated. Uses Var θ = E θ (1 − E θ) − E[θ (1 − θ)]."""
    lw = -EPS_GRID ** 2 / 2
    lw = lw - logsumexp(lw)
    x = np.asarray(c)[..., None] + np.exp(np.asarray(ls))[..., None] * EPS_GRID
    lp, lq = log_expit(x), log_expit(-x)
    l1, l0, lpq = logsumexp(lw + lp, -1), logsumexp(lw + lq, -1), logsumexp(lw + lp + lq, -1)
    one_minus_rho = np.clip(np.exp(lpq - l1 - l0), 1e-300, 1 - 1e-15)
    return l1 - l0, np.log1p(-one_minus_rho) - np.log(one_minus_rho)


@lru_cache(maxsize=1)
def _ln_table():
    cg, lsg = np.linspace(-40, 40, 801), np.linspace(-5, np.log(300), 300)
    a, b = _ln_moments(cg[:, None] + 0 * lsg, lsg + 0 * cg[:, None])
    return cg, lsg, a, b


def ln_params(mu, S, newton=30):
    """(c, σ) per image with E θ = μ and Var θ = μ(1 − μ)/(S + 1) for θ = sigmoid(c + σ ε): a start from a table
    (c for the mean at each σ, then σ for ρ), then Newton steps on (c, log σ)."""
    cg, lsg, A, B = _ln_table()
    lmu, lrho = logit(mu), -np.log(S)                              # logit ρ = logit 1/(S + 1) = −log S
    cj = np.stack([np.interp(lmu, A[:, j], cg) for j in range(len(lsg))], 1)
    i = np.clip(np.searchsorted(cg, cj) - 1, 0, len(cg) - 2)
    f, J = (cj - cg[i]) / (cg[1] - cg[0]), np.arange(len(lsg))[None, :]
    rj = np.maximum.accumulate(B[i, J] * (1 - f) + B[i + 1, J] * f, 1)          # ρ rises with σ (start only)
    ls = np.array([np.interp(lrho[n], rj[n], lsg) for n in range(len(mu))])
    c = np.array([np.interp(ls[n], lsg, cj[n]) for n in range(len(mu))])
    h = 1e-4
    for _ in range(newton):
        a, b = _ln_moments(c, ls)
        if np.abs(a - lmu).max() < 1e-7 and np.abs(b - lrho).max() < 1e-7:
            break
        ac, bc = _ln_moments(c + h, ls)
        as_, bs = _ln_moments(c, ls + h)
        jac = np.stack([np.stack([(ac - a) / h, (as_ - a) / h], -1), np.stack([(bc - b) / h, (bs - b) / h], -1)], -2)
        d = np.linalg.solve(jac, np.stack([a - lmu, b - lrho], -1)[..., None])[..., 0]
        c, ls = c - d[:, 0], ls - d[:, 1]
    a, b = _ln_moments(c, ls)
    if np.abs(a - lmu).max() > 1e-4 or np.abs(b - lrho).max() > 1e-4:
        raise RuntimeError("logit-normal moment matching did not converge")
    return c, np.exp(ls)


def _theta_draws(s: Setting, mu, S, rng, size):
    """size draws of the per-image vote share given u (rater-level shares in the raters world): [N, size]."""
    if s.world == "logitnormal":
        c, sig = ln_params(mu, S)
        return expit(c[:, None] + sig[:, None] * rng.normal(size=(len(mu), size)))
    return rng.beta(mu[:, None] * S[:, None], (1 - mu[:, None]) * S[:, None], size=(len(mu), size))


def _rater_bias():
    """The pool's biases (fixed with the world), centred so the pool-average rater has no bias on the logit scale."""
    b = np.random.default_rng(WORLD_SEED + 1).normal(0, RATER_SD, RATER_POOL)
    return b - b.mean()


def _rater_p(theta, raters=None, fixed=False):
    """[..., R]: each rater's probability of a yes vote given θ; raters [..., R] pool indices (None: the whole pool;
    fixed: the raters_fixed panel)."""
    b = np.asarray(FIXED_BIAS) if fixed else _rater_bias()
    lt = logit(np.clip(theta, 1e-9, 1 - 1e-9))[..., None]
    return expit(lt + (b if raters is None else b[raters]))


def _panels(rng, shape, size):
    """shape + (size,) distinct pool raters per panel, drawn at random."""
    return np.argsort(rng.random(shape + (RATER_POOL,)), -1)[..., :size]


def _pb(p):
    """Poisson-binomial pmf of the raters' yes count: [..., R + 1] from p [..., R]."""
    f = np.ones(p.shape[:-1] + (1,))
    for r in range(p.shape[-1]):
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
    v = None                                          # ordered votes [N, R, 1] (rater worlds; NaN = not on the panel)
    if s.world == "raters_fixed":
        votes = rng.random((N, 3)) < _rater_p(theta, fixed=True)
        n = np.full(N, 3.0)
        k = votes.sum(1).astype(float)
        theta_eff = _rater_p(theta, fixed=True).mean(-1)
        v = votes.astype(float)[..., None]
    elif s.world == "raters":
        r = rng.random((N, 3))                         # (draw order as before D8: votes' uniforms, then the panels)
        idx = _panels(rng, (N,), 3)
        votes = r < _rater_p(theta, idx)
        n = np.full(N, 3.0)
        k = votes.sum(1).astype(float)
        theta_eff = _rater_p(theta).mean(-1)
        v = np.full((N, RATER_POOL, 1), np.nan)
        np.put_along_axis(v[..., 0], idx, votes.astype(float), 1)
    else:
        if n_votes is not None:
            n = np.full(N, float(n_votes))
        elif s.n_range is not None:
            n = rng.integers(s.n_range[0], s.n_range[1] + 1, size=N).astype(float)
        else:
            n = np.full(N, float(s.n))
        k = rng.binomial(n.astype(int), theta).astype(float)
        theta_eff = theta
    return {"z": z, "mu": mu, "S": S, "clarity": clar, "theta": theta_eff, "theta_raw": theta, "k": k, "n": n, "v": v}


def draw_split(s: Setting, seed):
    """Training, validation and evaluation images of one seed (the seed redraws the data, not the world)."""
    rng = np.random.default_rng([seed, zlib.crc32(s.name.encode())])
    tr = sample(s, s.n_train, rng)
    va = sample(s, max(500, s.n_train // 3), rng)
    ev = sample(s, N_EVAL, rng)
    return tr, va, ev


def truths(s: Setting, ev, seed):
    """ν(u) = E[k/n | u] and the oracle NLL of fresh panels: exact (beta), quadrature (logitnormal, the ordered votes
    and the fixed panel's count), Monte Carlo over (θ, panel) for the pool's count (own generator, N_DRAWS_POOL)."""
    rng = np.random.default_rng([seed, 7, zlib.crc32(s.name.encode())])
    rng_o = np.random.default_rng([seed, 8, zlib.crc32(s.name.encode())])
    mu, S = ev["mu"], ev["S"]
    th = _theta_draws(s, mu, S, rng, N_DRAWS) if s.world != "logitnormal" else None   # (logitnormal: not used)
    panels, oracle_nll, ordered = {}, {}, {}

    def pattern_nll(beta, v):
        """Oracle −log P(the ordered votes v [N, R, 1] | u): the B²-rba likelihood at the true μ, S, β (quadrature)."""
        return P.ordered_nll(mu[:, None], S[:, None], np.asarray(beta)[None, :], v)[:, 0]

    if s.world == "raters_fixed":
        p = _rater_p(th, fixed=True)                               # [N, draws, 3]
        nu = p.mean((1, 2))
        pr = _rater_p(ev["theta_raw"], fixed=True)
        y3 = (rng.random(pr.shape) < pr).astype(float)
        k3 = y3.sum(-1)
        pmf = P.rater_count_pmf(mu[:, None], S[:, None], np.asarray(FIXED_BIAS)[None, :], y3[..., None])[:, 0]
        panels[3] = k3
        oracle_nll[3] = -np.log(np.take_along_axis(pmf, k3.astype(int)[:, None], 1)[:, 0])
        ordered[3] = (y3[..., None], pattern_nll(FIXED_BIAS, y3[..., None]))
    elif s.world == "raters":
        nu = _rater_p(th).mean((1, 2))
        for ne in RATER_PANELS:
            idx = _panels(rng, (len(th),), ne)
            pr = _rater_p(ev["theta_raw"], idx)
            y = (rng.random(pr.shape) < pr).astype(float)
            k = y.sum(-1)
            _panels(rng, th.shape, ne)                               # (kept: the draw order of the panels above)
            pmf = np.concatenate([_pb(_rater_p(_theta_draws(s, mu[sl], S[sl], rng_o, N_DRAWS_POOL),
                                               _panels(rng_o, (len(sl), N_DRAWS_POOL), ne))).mean(1)
                                  for sl in np.array_split(np.arange(len(mu)), 50)])   # mixture over θ and panels
            panels[ne] = k
            oracle_nll[ne] = -np.log(np.take_along_axis(pmf, k.astype(int)[:, None], 1)[:, 0])
            v = np.full((len(th), RATER_POOL, 1), np.nan)            # ordered votes of the known panel
            np.put_along_axis(v[..., 0], idx, y, 1)
            ordered[ne] = (v, pattern_nll(_rater_bias(), v))
    else:
        nu = mu                                                      # both worlds: E θ = μ (logitnormal by construction)
        if s.world == "logitnormal":
            c, sig = ln_params(mu, S)
            lw = -EPS_GRID ** 2 / 2
            lw = lw - logsumexp(lw)
            th_q = expit(c[:, None] + sig[:, None] * EPS_GRID)       # [N, nodes]
        for ne in N_EVAL_PANELS:
            k = rng.binomial(ne, ev["theta"]).astype(float)
            panels[ne] = k
            if s.world == "beta":
                oracle_nll[ne] = P.vote_nll(mu, S, k, np.full_like(k, ne))
            else:
                oracle_nll[ne] = -logsumexp(lw + binom.logpmf(k[:, None], ne, th_q), -1)
    return nu, panels, oracle_nll, ordered


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
    keep_v = model in RATER_MODELS
    sp = lambda x, p, lab=True: T.Split(x["z"], ids(len(x["z"]), p), x["k"][:, None] if lab else None,
                                        x["n"][:, None] if lab else None, x["v"] if (lab and keep_v) else None)
    cfg = T.TrainConfig(variant, seed=seed, num_workers=0)
    T.run(cfg, sp(tr, "tr"), sp(va, "va"), sp(ev, "ev", False), d, device=device, verbose=False)
    (d / "setting.json").write_text(json.dumps({**asdict(s), "model": model, "seed": seed}))
    return "done"


# ---------------------------------------------------------------- scoring against the truth
def score(s: Setting, seed, root=OUT):
    """Rows (one per model and checkpoint) for one setting and seed."""
    _, va, ev = draw_split(s, seed)
    nu, panels, oracle, ordered = truths(s, ev, seed)
    true_beta = np.asarray(FIXED_BIAS) if s.world == "raters_fixed" else _rater_bias()
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
                 "mse_mu": float(((m - ev["mu"]) ** 2).mean()),
                 "auroc": auroc(m, (ev["theta"] > 0.5).astype(float)),
                 "spearman_S_clarity": (float(spearmanr(S, ev["clarity"])[0])
                                        if model in ("b2r", "b2rba") and np.std(S) > 0 else np.nan),
                 "spearman_S_absm": (float(spearmanr(S, np.abs(logit(np.clip(m, 1e-9, 1 - 1e-9))))[0])
                                     if model in ("b2r", "b2rba") and np.std(S) > 0 else np.nan)}
            for ne, k in panels.items():
                r[f"excess_nll_n{ne}"] = float((P.vote_nll(m, S, k, np.full_like(k, ne)) - oracle[ne]).mean())
            beta = P.rater_offsets(variant, out) if model in RATER_MODELS else None
            if beta is not None:
                r["beta_corr"] = float(np.corrcoef(beta[0], true_beta)[0, 1])
                r["beta_mae"] = float(np.abs(beta[0] - true_beta).mean())
            for ne, (v, o) in ordered.items():               # D8: who voted what, on fresh panels of known raters
                r[f"excess_onll_n{ne}"] = float((P.ordered_nll(m[:, None], S[:, None], beta, v)[:, 0] - o).mean())
            if model in ("b2r", "b2rba"):                    # concentration ablation (D0b on real data), 3-vote panels
                r["d_nll_concentration_n3"] = concentration_ablation(m, S, beta, *((ordered[3][0], None) if model == "b2rba"
                                                                     else (None, panels[3])))
            rows.append(r)
    return rows


def concentration_ablation(m, S, beta=None, v=None, k=None):
    """NLL per panel with one S (fitted on these panels, as on real data) minus with the per-image S: > 0 = the
    per-image concentration is informative. Ordered votes v with rater biases β (B²-rba), else the 3-vote counts k."""
    from scipy.optimize import minimize_scalar
    nll = ((lambda SS: P.ordered_nll(m[:, None], np.broadcast_to(SS, m.shape)[:, None], beta, v)[:, 0]) if v is not None
           else (lambda SS: P.vote_nll(m, np.broadcast_to(SS, m.shape), k, np.full_like(k, 3.0))))
    best = minimize_scalar(lambda ls: nll(np.exp(ls)).mean(), bounds=(-3, 12), method="bounded").fun
    return float(best - nll(S).mean())


def _ms(x, f=".4f"):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return "–" if not len(x) else f"{x.mean():{f}} ± {x.std(ddof=1) if len(x) > 1 else 0:{f}}"


def report(root=OUT):
    import pandas as pd
    rows = []
    for s in settings():
        for seed in SEEDS:
            rows += score(s, seed, root)
    df = pd.DataFrame(rows)
    df.to_csv(Path(root) / "synth_long.csv", index=False)
    md = ["# D7 — synthetic study (validation-free: scored against the known truth)", "",
          "Mean recovery: MSE(m, ν) with ν = E[k/n | latent]; Δ = MSE(soft BCE) − MSE(B²-r) (positive = B²-r closer). "
          "All ± are sd over 5 seeds; differences are paired (same data per seed). Concentration: Spearman of B²-r's S with "
          "the true clarity (which drives S when γ > 0), and the concentration ablation (NLL per 3-vote panel with one S "
          "fitted on the evaluation panels minus with the per-image S; > 0 = per-image S informative; the D0b test on "
          "real data). Vote NLL: excess over the oracle on fresh panels, Global-S / B²-r, and Δ = Global-S − B²-r.", ""]
    for ck in ("B", "A"):
        t = df[df.checkpoint == ck]
        md += [f"## checkpoint {ck}", "",
               "| setting | MSE soft BCE | MSE B²-r | Δ MSE | B² better (seeds) | MSE 200-vote ref | Spearman S~clarity | "
               "concentration ablation n=3 | excess NLL n=3 GS / B² (Δ) | n=10 | n=50 | AUROC soft / B² | MSE vote-weighted |",
               "|" + "---|" * 13]
        for s in settings():
            g = t[t.setting == s.name].pivot(index="seed", columns="model")
            if g.empty or "b2r" not in g["mse_nu"] or "soft_bce" not in g["mse_nu"]:
                continue
            dlt = g["mse_nu"]["soft_bce"] - g["mse_nu"]["b2r"]
            col = lambda c, mdl: g[c][mdl] if c in g and mdl in g[c] else pd.Series(dtype=float)

            def nll(ne):
                a, b = col(f"excess_nll_n{ne}", "soft_bce"), col(f"excess_nll_n{ne}", "b2r")
                return "–" if not b.notna().any() else f"{a.mean():.4f} / {b.mean():.4f} ({_ms(a - b)})"
            md.append(f"| {s.name} | {_ms(col('mse_nu', 'soft_bce'), '.5f')} | {_ms(col('mse_nu', 'b2r'), '.5f')} | "
                      f"{_ms(dlt, '+.5f')} | {(dlt > 0).sum()}/{len(dlt)} | {_ms(col('mse_nu', 'ref200'), '.5f')} | "
                      f"{_ms(col('spearman_S_clarity', 'b2r'), '+.2f')} | {_ms(col('d_nll_concentration_n3', 'b2r'), '+.4f')} | "
                      f"{nll(3)} | {nll(10)} | {nll(50)} | "
                      f"{_ms(col('auroc', 'soft_bce'), '.3f')} / {_ms(col('auroc', 'b2r'), '.3f')} | "
                      f"{_ms(col('mse_nu', 'soft_bce_n'), '.5f')} |")
        md.append("")
    tb = df[df.checkpoint == "B"]
    md += ["## S5 — how strongly must the concentration vary to be seen? (checkpoint B)", "",
           "γ = slope of log S on clarity (log S sd = γ; γ = 1: S from ≈ 1.4 to 18, p10–p90). Real-data reference "
           "(D0b, B²-r, ΔNLL per image and label): Endoscapes +0.001 ± 0.000, AMi-Br −0.001 ± 0.001 (3 votes), "
           "MHIST +0.046 ± 0.003 (7 votes). Rater rows: the fixed 3-rater panel with biases (−1, 0, 1); B²-rba's "
           "ablation uses its ordered-vote likelihood.", "",
           "| setting | γ | model | Spearman S~clarity | concentration ablation n=3 | Δ excess NLL n=3 (GS − B²) | n=10 |",
           "|---|---|---|---|---|---|---|"]
    by = {x.name: x for x in settings()}
    for grp in SWEEP_ROWS:
        for name in grp:
            g = tb[tb.setting == name].pivot(index="seed", columns="model")
            if g.empty or "b2r" not in g["mse_nu"]:
                continue
            for mdl in ("b2r", "b2rba"):
                if mdl not in g["mse_nu"]:
                    continue
                dn = lambda ne: (_ms(g[f"excess_nll_n{ne}"]["soft_bce"] - g[f"excess_nll_n{ne}"]["b2r"], "+.4f")
                                 if mdl == "b2r" and f"excess_nll_n{ne}" in g and g[f"excess_nll_n{ne}"]["b2r"].notna().any()
                                 else "–")
                md.append(f"| {name} | {by[name].gamma:g} | {mdl} | {_ms(g['spearman_S_clarity'][mdl], '+.2f')} | "
                          f"{_ms(g['d_nll_concentration_n3'][mdl], '+.4f')} | {dn(3)} | {dn(10)} |")
    md.append("")
    rt = df[df.setting.isin([x.name for x in settings() if x.world in ("raters", "raters_fixed")])]
    if not rt.empty and "beta_mae" in rt:
        md += ["## D8 — rater worlds: B²-rba and soft BCE-ba (checkpoint B)", "",
               "Ordered-vote excess NLL over the oracle on fresh panels of known raters (who voted what); for models that "
               "treat raters as interchangeable every ordering of the count is equally likely. β: mean abs error of the "
               "learned rater biases. MSE(m, μ): μ = the typical (unbiased) rater's mean share. Spearman S~|logit m|: does "
               "S track how extreme the mean is (where bias-driven dispersion would sit) rather than clarity.", "",
               "| setting | model | excess ordered NLL n=3 | n=10 | excess count NLL n=3 | Spearman S~clarity | "
               "Spearman S~abs logit m | concentration ablation n=3 | mean abs(β̂ − β) | MSE(m, μ) | MSE(m, ν) | AUROC |",
               "|" + "---|" * 12]
        for name in dict.fromkeys(rt.setting):
            for model in ("soft_bce", "b2r", "soft_bce_ba", "b2rba"):
                g = rt[(rt.setting == name) & (rt.model == model) & (rt.checkpoint == "B")]
                if g.empty:
                    continue
                c = lambda col, f=".4f": "–" if col not in g else _ms(g[col], f)
                md.append(f"| {name} | {model} | {c('excess_onll_n3')} | {c('excess_onll_n10')} | {c('excess_nll_n3')} | "
                          f"{c('spearman_S_clarity', '+.2f')} | {c('spearman_S_absm', '+.2f')} | "
                          f"{c('d_nll_concentration_n3', '+.4f')} | {c('beta_mae', '.3f')} | "
                          f"{c('mse_mu', '.5f')} | {c('mse_nu', '.5f')} | {c('auroc', '.3f')} |")
            g = rt[(rt.setting == name) & (rt.checkpoint == "B")].pivot(index="seed", columns="model")
            if "b2rba" in g["mse_nu"] and "b2r" in g["mse_nu"]:
                d = lambda col, f: _ms(g[col]["b2rba"] - g[col]["b2r"], f)
                md.append(f"| {name} | paired B²-rba − B²-r | {d('excess_onll_n3', '+.4f')} | – | {d('excess_nll_n3', '+.4f')} | "
                          f"{d('spearman_S_clarity', '+.2f')} | {d('spearman_S_absm', '+.2f')} | "
                          f"{d('d_nll_concentration_n3', '+.4f')} | – | {d('mse_mu', '+.5f')} | – | – |")
        md.append("")
    out = Path(root) / "synth_report.md"
    out.write_text("\n".join(md))
    return out


def describe(s: Setting, N=20000, seed=0):
    """Vote histogram and concentration of the world at the setting's n (for calibration against real datasets)."""
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
