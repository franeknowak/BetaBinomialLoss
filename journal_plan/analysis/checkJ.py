"""J: what S looks like in the saved runs (val at best epoch and test)."""
import json, glob, numpy as np
from scipy.optimize import brentq
from scipy.stats import spearmanr, betabinom
from sklearn.metrics import roc_auc_score, average_precision_score
B = '/mnt/user-data/uploads/BetaBinomialLoss/results/'
def pmaj(m, S):
    a1 = m * S; return 3 * a1 * (a1 + 1) / (S * (S + 1)) - 2 * a1 * (a1 + 1) * (a1 + 2) / (S * (S + 1) * (S + 2))
def inv_m(p, S):
    if p <= 1e-12: return 0.0
    if p >= 1 - 1e-12: return 1.0
    return brentq(lambda mm: pmaj(mm, S) - p, 1e-12, 1 - 1e-12)
def bacc(score, yb, t=.5):
    pr = score >= t; return .5 * ((pr & yb).sum() / yb.sum() + (~pr & ~yb).sum() / (~yb).sum())
files = {'frozen SwinV2 +KL(prior) [base]': 'results_26_05_26/ft/frozen_bbl_0_results.json',
         'frozen SwinV2 no KL': 'results_26_05_26/ablation/nokl_bbl_0_results.json',
         'frozen SwinV2 KL uniform': 'results_26_05_26/ablation/kl_no_prior_bbl_0_results.json',
         'frozen DINOv3 +KL': 'results_26_05_26/encoder/dinov3_bbl_0_results.json',
         'frozen SwinV2 single-frame': 'results_26_05_26/temporal/notemp_bbl_0_results.json',
         'E2E SwinV2 (stages>=2) +KL': 'results_26_05_26/ft/e2e_bbl_0_results.json',
         'results/frozen_bbl_0 (root, newer)': 'frozen_bbl_0_results.json'}
for name, f in files.items():
    R = json.load(open(B + f))
    tk = [k for k in R if k.endswith('Test')][0]
    be = int(tk.split()[1])
    print(f"\n### {name}   best epoch={be}, epochs run={sum(1 for x in R if x.endswith('Val'))}")
    print("   val avg_uncert_pos/neg by epoch:", [(i + 1, round(2 / R[f'Epoch {i+1} Val']['avg_uncert_pos'], 1), round(2 / R[f'Epoch {i+1} Val']['avg_uncert_neg'], 1)) for i in range(sum(1 for x in R if x.endswith('Val')))], "(shown as 2/mean(u), ~S)")
    for split, key in [('val', f'Epoch {be} Val'), ('test', tk)]:
        T = R[key]['saved']; k = np.rint(np.array(T['labels']) * 3).astype(int)
        for ci, c in enumerate(['C1', 'C2', 'C3']):
            S = 2 / np.array(T[c]['uncerts']); P = np.array(T[c]['probs']); kc = k[:, ci]; yb = kc >= 2
            m = np.array([inv_m(p, s) for p, s in zip(P, S)])
            dis = (kc == 1) | (kc == 2)
            med = [np.median(S[kc == j]) if (kc == j).any() else np.nan for j in range(4)]
            # NLL of observed votes under the predicted BB, vs under binomial with same mean (S->inf)
            nll = -betabinom.logpmf(kc, 3, m * S, (1 - m) * S).mean()
            nll_inf = -np.mean(np.log(np.clip(np.array([__import__('scipy').stats.binom.pmf(kk, 3, mm) for kk, mm in zip(kc, m)]), 1e-12, 1)))
            print(f"   {split:4s} {c}: median S by k=0..3: {np.round(med, 1)}  S range[{S.min():.2f},{np.percentile(S,99):.0f}]"
                  f"  AUROC(low S -> split vote)={roc_auc_score(dis, -S):.3f}  AUROC(low S -> error)={roc_auc_score((P>=.5)!=yb, -S):.3f}"
                  f"\n              AP Pmaj={average_precision_score(yb, P):.4f} AP m={average_precision_score(yb, m):.4f}  BACC Pmaj={bacc(P, yb):.4f} m={bacc(m, yb):.4f}"
                  f"  mean Pmaj={P.mean():.3f} vs frac(k>=2)={yb.mean():.3f}; mean m={m.mean():.3f} vs mean k/3={kc.mean()/3:.3f}  NLL BB={nll:.3f} vs S->inf={nll_inf:.3f}")
