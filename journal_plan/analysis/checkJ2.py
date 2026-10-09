import json, numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.optimize import brentq
from scipy.stats import betabinom, binom, spearmanr
from sklearn.metrics import roc_auc_score
B='/mnt/user-data/uploads/BetaBinomialLoss/'
A=json.load(open(B+'config/reformatted_annotations_frames.json'))
lut={}
for sp in A:
    for v in A[sp].values():
        a=v['annotations']; lut[(sp,a['video_id'],a['frame_id'])]=np.rint(np.array(a['ds'])*3).astype(int)
def pmaj(m,S):
    a1=m*S; return 3*a1*(a1+1)/(S*(S+1))-2*a1*(a1+1)*(a1+2)/(S*(S+1)*(S+2))
def inv_m(p,S):
    if p<=1e-12: return 1e-9
    if p>=1-1e-12: return 1-1e-9
    return brentq(lambda mm: pmaj(mm,S)-p,1e-12,1-1e-12)
files={'frozen SwinV2 +KL(prior)':'results/results_26_05_26/ft/frozen_bbl_0_results.json',
       'frozen SwinV2 no KL':'results/results_26_05_26/ablation/nokl_bbl_0_results.json',
       'frozen DINOv3 +KL':'results/results_26_05_26/encoder/dinov3_bbl_0_results.json',
       'E2E SwinV2 +KL (ep1)':'results/results_26_05_26/ft/e2e_bbl_0_results.json'}
for name,f in files.items():
    R=json.load(open(B+f)); tk=[k for k in R if k.endswith('Test')][0]
    print(f"\n### {name}")
    for split,key in [('val',f"Epoch {tk.split()[1]} Val"),('test',tk)]:
        T=R[key]['saved']
        K=np.stack([lut[(split,int(v),int(fr))] for v,fr in zip(T['vid_ids'],T['frame_ids'])])
        for ci,c in enumerate(['C1','C2','C3']):
            S=2/np.array(T[c]['uncerts']); P=np.array(T[c]['probs']); k=K[:,ci]; yb=k>=2
            m=np.array([inv_m(p,s) for p,s in zip(P,S)])
            split_vote=(k==1)|(k==2)
            conf=np.abs(m-0.5)
            err=(m>=.5)!=yb
            # does S carry info about split votes beyond the mean? AUROC within strata of m
            bins=np.quantile(m,np.linspace(0,1,6)); idx=np.clip(np.digitize(m,bins[1:-1]),0,4)
            within=[roc_auc_score(split_vote[idx==b],-S[idx==b]) for b in range(5) if 0<split_vote[idx==b].mean()<1]
            nll=-betabinom.logpmf(k,3,m*S,(1-m)*S).mean(); nll_inf=-binom.logpmf(k,3,m).mean()
            # oracle: NLL with model mean but S fitted globally per criterion
            Sg=np.logspace(0,3,300); nllg=[-betabinom.logpmf(k,3,m*s,(1-m)*s).mean() for s in Sg]
            print(f"  {split:4s} {c}: med S by k=0..3 {np.round([np.median(S[k==j]) for j in range(4)],1)} | AUROC split-vote: lowS={roc_auc_score(split_vote,-S):.3f} "
                  f"vs |m-.5|small={roc_auc_score(split_vote,-conf):.3f} vs m={roc_auc_score(split_vote,m):.3f}; within-m-quintile lowS={np.mean(within):.3f} | "
                  f"AUROC error: lowS={roc_auc_score(err,-S):.3f} |m-.5|={roc_auc_score(err,-conf):.3f} | rho(S,m)={spearmanr(S,m)[0]:+.2f} | "
                  f"NLL pred={nll:.3f} S->inf={nll_inf:.3f} bestconstS={min(nllg):.3f}@S={Sg[int(np.argmin(nllg))]:.1f}")
