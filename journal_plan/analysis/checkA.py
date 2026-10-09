"""A: verify code against closed forms. B: large-S expansion. C: Pmaj threshold equivalence. D: ranking."""
import sys, numpy as np, torch
sys.path.insert(0, '/mnt/user-data/uploads/BetaBinomialLoss')
from scripts.bbkl_loss import beta_binomial_nll_per_sample, dirichlet_kl_per_sample, evidential_bb_loss
from scripts.metrics import p_k_ge_2_from_a
from scipy.stats import betabinom, binom
from scipy.special import betaln, digamma, gammaln
torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)

print("== A1: code NLL vs scipy betabinom")
a0 = rng.uniform(1, 50, 2000); a1 = rng.uniform(1, 50, 2000); k = rng.integers(0, 4, 2000)
alpha = torch.tensor(np.stack([a0, a1], 1)); y = torch.tensor(k / 3.0)
code = beta_binomial_nll_per_sample(alpha, y).numpy()
ref = -betabinom.logpmf(k, 3, a1, a0)
print("max abs diff", np.abs(code - ref).max())
# check y_soft rounding robust to float 1/3, 2/3 (float32 path as in dataset)
y32 = torch.tensor([0., 1/3, 2/3, 1.], dtype=torch.float32)
print("k from float32 soft labels:", torch.round(y32 * 3).long().tolist())

print("== A2: Pmaj closed form vs betabinom.sf(1)")
P = p_k_ge_2_from_a(torch.tensor(a0), torch.tensor(a1)).numpy()
print("max abs diff", np.abs(P - betabinom.sf(1, 3, a1, a0)).max())

print("== A3: Dirichlet KL (code) vs closed-form Beta KL")
def kl_beta(a1, a0, p1, p0):
    S, Sp = a1 + a0, p1 + p0
    return (betaln(p1, p0) - betaln(a1, a0) + (a1 - p1) * digamma(a1) + (a0 - p0) * digamma(a0)
            + (Sp - S) * digamma(S))
pr = (0.8437 * 2, 0.1563 * 2)  # (a0prior, a1prior) as built in build_loss.py
klc = dirichlet_kl_per_sample(alpha, prior_alpha=torch.tensor(pr)).numpy()
print("max abs diff", np.abs(klc - kl_beta(a1, a0, pr[1], pr[0])).max())
print("prior used: a1prior=%.4f a0prior=%.4f (a1prior<1 => unreachable since model alpha>=1)" % (pr[1], pr[0]))

print("== A4: lambda actually used")
import inspect
print("evidential_bb_loss default lambda_reg =", inspect.signature(evidential_bb_loss).parameters['lambda_reg'].default,
      "; build_loss.py does not pass lambda -> always 0.05")

print("== B: large-S expansion. NLL_BB - 3*softBCE(y=k/3, m) - (-log C(3,k)) vs -c/S")
def nll_bb(k, m, S, n=3):
    return -betabinom.logpmf(k, n, m * S, (1 - m) * S)
def c_coef(k, m, n=3):
    return k * (k - 1) / (2 * m) + (n - k) * (n - k - 1) / (2 * (1 - m)) - n * (n - 1) / 2
for m in [0.1, 0.4]:
    for kk in range(4):
        row = []
        for S in [10, 100, 1000, 10000]:
            y_ = kk / 3
            softbce = -(y_ * np.log(m) + (1 - y_) * np.log(1 - m))
            resid = nll_bb(kk, m, S) - (3 * softbce - np.log([1, 3, 3, 1][kk]))
            row.append(f"S={S}: resid*S={resid * S:+.3f}")
        print(f" m={m} k={kk}  -c={-c_coef(kk, m):+.3f} | " + "  ".join(row))

print("== C: is Pmaj>0.5 <=> E[theta]>0.5 for all S?")
bad = 0
for S in np.logspace(np.log10(2), 4, 200):
    for m in np.linspace(0.01, 0.99, 197):
        if m * S < 1 or (1 - m) * S < 1:  # respect alpha>=1 as in code
            continue
        p = betabinom.sf(1, 3, m * S, (1 - m) * S)
        if (p > 0.5) != (m > 0.5) and abs(m - 0.5) > 1e-9:
            bad += 1
print("violations:", bad)
print("Pmaj at fixed m=0.3 vs S:", {S: round(betabinom.sf(1, 3, .3 * S, .7 * S), 4) for S in [3.4, 5, 10, 30, 100, 1e4]})
print("Pmaj at fixed m=0.7 vs S:", {S: round(betabinom.sf(1, 3, .7 * S, .3 * S), 4) for S in [3.4, 5, 10, 30, 100, 1e4]})
