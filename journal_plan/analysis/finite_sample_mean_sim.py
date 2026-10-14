import numpy as np, torch
torch.set_default_dtype(torch.float64)
def run(N, seed, het=1.5, n=3):
    g=np.random.default_rng(seed)
    x=g.normal(size=N); z=2*x-1.5+het*g.normal(size=N); th=1/(1+np.exp(-z)); k=g.binomial(n,th)
    X=torch.tensor(x); K=torch.tensor(k,dtype=torch.float64); f=K/n
    xs=np.linspace(-3,3,601); zz=2*xs[:,None]-1.5+het*np.random.default_rng(0).normal(size=(1,4000))
    target=(1/(1+np.exp(-zz))).mean(1)                       # true E[theta|x]
    out={}
    for loss in ("soft","b2"):
        p=torch.zeros(3,requires_grad=True); opt=torch.optim.LBFGS([p],max_iter=500,line_search_fn="strong_wolfe")
        def cl():
            opt.zero_grad(); m=torch.sigmoid(p[0]+p[1]*X)
            if loss=="soft": L=-(f*torch.log(m)+(1-f)*torch.log(1-m)).mean()
            else:
                S=torch.exp(p[2]); a1,a0=m*S,(1-m)*S
                L=-(torch.lgamma(K+a1)+torch.lgamma(n-K+a0)-torch.lgamma(n+S)-torch.lgamma(a1)-torch.lgamma(a0)+torch.lgamma(S)).mean()
            L.backward(); return L
        opt.step(cl)
        m=1/(1+np.exp(-(p[0].item()+p[1].item()*xs))); out[loss]=np.mean((m-target)**2)
    return out
for N in (100,300,1000,5000):
    r=[run(N,s) for s in range(200)]
    s=np.array([d["soft"] for d in r]); b=np.array([d["b2"] for d in r])
    print(f"N={N:5d}  MSE(m vs true) soft={s.mean():.5f} b2={b.mean():.5f}  B2 better in {np.mean(b<s)*100:.0f}% of repeats; mean |m_soft-m_b2| gap ratio={np.mean(np.abs(b-s))/s.mean():.2f}")
