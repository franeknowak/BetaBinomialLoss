"""E: Fisher info for log S vs n. F/G: population optimum under weights/KL/alpha>=1.
H: finite-sample identifiability of S. I: annotator heterogeneity / correlation."""
import numpy as np
from scipy.stats import betabinom, binom
from scipy.special import betaln, digamma, expit, logit
from scipy.optimize import minimize
rng = np.random.default_rng(1)

def logpmf(k, n, m, S):
    return betabinom.logpmf(k, n, m * S, (1 - m) * S)

print("== E: expected Fisher information about log S from ONE item with n votes")
print("   (I_logS; 1/I = asymptotic variance of log S-hat per item; n=1 gives 0 = unidentifiable)")
for m in [0.15, 0.5]:
    for S in [2, 10, 50]:
        row = []
        for n in [1, 2, 3, 5, 10, 50]:
            ks = np.arange(n + 1); h = 1e-4
            d = (logpmf(ks, n, m, S * np.exp(h)) - logpmf(ks, n, m, S * np.exp(-h))) / (2 * h)
            I = np.sum(np.exp(logpmf(ks, n, m, S)) * d ** 2)
            row.append(f"n={n}:{I:.4f}")
        print(f"  m={m} S={S}: " + "  ".join(row))
print("  -> items needed for SE(log S)=0.5 (i.e. S within ~x1.6):  N = 4/I")
for m in [0.15]:
    for S in [2, 10, 50]:
        out = []
        for n in [3, 10, 50]:
            ks = np.arange(n + 1); h = 1e-4
            d = (logpmf(ks, n, m, S * np.exp(h)) - logpmf(ks, n, m, S * np.exp(-h))) / (2 * h)
            I = np.sum(np.exp(logpmf(ks, n, m, S)) * d ** 2)
            out.append(f"n={n}: N~{4 / I:.0f}")
        print(f"  m={m} S={S}: " + "  ".join(out))

# ---------------------------------------------------------------- F/G
W = {'C1': (3.1985, 0.5926)}
PI, NU = 0.1563, 2.0

def kl_beta(a1, a0, p1, p0):
    S, Sp = a1 + a0, p1 + p0
    return (betaln(p1, p0) - betaln(a1, a0) + (a1 - p1) * digamma(a1) + (a0 - p0) * digamma(a0) + (Sp - S) * digamma(S))

def true_q(m, S_true, n=3):
    ks = np.arange(n + 1)
    if np.isinf(S_true):
        return binom.pmf(ks, n, m)
    return betabinom.pmf(ks, n, m * S_true, (1 - m) * S_true)

def fit(q, weights=None, lam=0.0, prior=(PI * NU, (1 - PI) * NU), amin=0.0, n=3):
    ks = np.arange(n + 1)
    w = np.ones(n + 1) if weights is None else np.where(ks >= 2, weights[0], weights[1])
    def obj(z):
        a1, a0 = amin + np.exp(z[0]), amin + np.exp(z[1])
        v = -np.sum(q * w * betabinom.logpmf(ks, n, a1, a0))
        if lam > 0:
            v += lam * kl_beta(a1, a0, prior[0], prior[1])
        return v
    best = None
    for z0 in [(0, 0), (2, 4), (-2, 1), (4, 6), (6, 8)]:
        r = minimize(obj, z0, method='Nelder-Mead', options=dict(xatol=1e-8, fatol=1e-12, maxiter=20000))
        if best is None or r.fun < best.fun:
            best = r
    a1, a0 = amin + np.exp(best.x[0]), amin + np.exp(best.x[1])
    return a1, a0

configs = [
    ("pure NLL, alpha>0", dict()),
    ("pure NLL, alpha>=1 (code head)", dict(amin=1.0)),
    ("weights only, alpha>=1", dict(amin=1.0, weights=W['C1'])),
    ("KL(prior) only, alpha>=1", dict(amin=1.0, lam=0.05)),
    ("CODE: weights+KL(prior)+alpha>=1", dict(amin=1.0, weights=W['C1'], lam=0.05)),
    ("CODE but KL uniform Beta(1,1)", dict(amin=1.0, weights=W['C1'], lam=0.05, prior=(1.0, 1.0))),
]
print("\n== F/G: population-optimal Beta for a group of indistinguishable items (C1 weights, lambda=0.05, nu=2)")
print("   truth: theta ~ Beta(m*S_true,(1-m)*S_true) [S_true=inf: all items share theta=m]; report fitted m, S, Pmaj vs true P(k>=2)")
for S_true in [np.inf, 20, 5, 1.5]:
    for m in [0.02, 0.15, 0.33, 0.5, 0.7, 0.95]:
        q = true_q(m, S_true)
        ptrue = q[2] + q[3]
        print(f" S_true={S_true:>4} m={m:.2f}  true P(k>=2)={ptrue:.3f}")
        for name, kw in configs:
            a1, a0 = fit(q, **kw)
            S = a1 + a0
            pm = betabinom.sf(1, 3, a1, a0)
            Sstr = f"{S:9.1f}" if S < 1e5 else "      inf"
            print(f"    {name:36s} m={a1 / S:.3f}  S={Sstr}  Pmaj={pm:.3f}")

# ---------------------------------------------------------------- H
print("\n== H: finite-sample MLE of S for a group of N items, n votes each (m=0.15)")
def mle_S(k, n):
    def obj(z):
        m = expit(z[0]); S = np.exp(z[1])
        return -np.sum(logpmf(k, n, m, S))
    best = None
    for z0 in [(logit(max(min(k.mean() / n, .99), .01)), np.log(5)), (0, np.log(50)), (0, np.log(1))]:
        r = minimize(obj, z0, method='Nelder-Mead', options=dict(maxiter=4000))
        if best is None or r.fun < best.fun: best = r
    return min(np.exp(best.x[1]), 1e4)
for S_true in [2, 10, 50]:
    for n in [3, 10, 50]:
        for N in [30, 200]:
            est = []
            for rep in range(200):
                th = rng.beta(.15 * S_true, .85 * S_true, N)
                k = rng.binomial(n, th)
                est.append(mle_S(k, n))
            est = np.array(est)
            print(f"  S_true={S_true:3d} n={n:2d} N={N:4d}: S-hat q10/50/90 = {np.percentile(est, 10):8.1f} {np.median(est):8.1f} {np.percentile(est, 90):8.1f}   P(S-hat>=1e3)={np.mean(est >= 1e3):.2f}")

# ---------------------------------------------------------------- I
print("\n== I: what non-iid annotators do to the fitted S and to Pmaj (3 fixed annotators, items theta~Beta(m S_true))")
def sim_votes(N, m, S_true, offsets, corr_sd=0.0):
    th = rng.beta(m * S_true, (1 - m) * S_true, N) if np.isfinite(S_true) else np.full(N, m)
    th = np.clip(th, 1e-4, 1 - 1e-4)
    lo = logit(th)[:, None] + np.array(offsets)[None, :]
    if corr_sd > 0:  # shared item-level shock to all annotators -> correlation beyond theta
        lo = lo + rng.normal(0, corr_sd, (N, 1))
    p = expit(lo)
    votes = rng.random(p.shape) < p
    return votes.sum(1), votes
for m, S_true in [(0.3, 5), (0.3, np.inf), (0.15, 2)]:
    for name, off, cs in [("iid equal annotators", (0, 0, 0), 0),
                           ("one biased annotator (+1.5 logit)", (1.5, 0, 0), 0),
                           ("skill spread (-1,0,+1 logit)", (-1, 0, 1), 0),
                           ("shared item shock sd=1.5 (correlated)", (0, 0, 0), 1.5)]:
        k, v = sim_votes(200000, m, S_true, off, cs)
        mh = k.mean() / 3
        Sh = mle_S(k[:20000], 3)
        ptrue = np.mean(k >= 2)
        pfit = betabinom.sf(1, 3, mh * Sh, (1 - mh) * Sh)
        print(f"  m={m} S_true={S_true}: {name:40s} fitted S={Sh:8.1f}  P(k>=2) true={ptrue:.3f} BB-implied={pfit:.3f}  per-annotator pos rates={np.round(v.mean(0), 3)}")
