import json, numpy as np
exec(open('checkJ2.py').read().split("files={")[0])
files={'frozen SwinV2 +KL':'results/results_26_05_26/ft/frozen_bbl_0_results.json','frozen SwinV2 no KL':'results/results_26_05_26/ablation/nokl_bbl_0_results.json','frozen DINOv3 +KL':'results/results_26_05_26/encoder/dinov3_bbl_0_results.json','E2E SwinV2 +KL':'results/results_26_05_26/ft/e2e_bbl_0_results.json'}
for name,f in files.items():
    R=json.load(open(B+f)); tk=[k for k in R if k.endswith('Test')][0]; T=R[tk]['saved']
    out=[]
    for c in ['C1','C2','C3']:
        S=2/np.array(T[c]['uncerts']); P=np.array(T[c]['probs']); m=np.array([inv_m(p,s) for p,s in zip(P,S)])
        a1=m*S; a0=(1-m)*S
        out.append(f"{c}: alpha1<1.1 {np.mean(a1<1.1):.2f}, alpha0<1.1 {np.mean(a0<1.1):.2f}, either {np.mean((a1<1.1)|(a0<1.1)):.2f}; median S/(1/min(m,1-m))={np.median(S*np.minimum(m,1-m)):.2f}")
    print(name,'(test)\n   '+'\n   '.join(out))
