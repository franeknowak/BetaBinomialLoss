"""KL-prior ablation of B² (user, 2026-11-23): B² plus λ · KL(predicted Beta ‖ prior Beta) per image and label.

  python -m b2j.ablation_kl <dataset> [--variants uniform moment] [--seeds 1 2 3 4 5]

Loss per (image, label): Beta-Binomial NLL of (k, n) + LAMBDA · KL( Beta(α₁, α₀) ‖ Beta(p₁, p₀) ); batch mean, sum over
labels, as for B². λ = 0.05 (the MICCAI value). Head, ε, AdamW, schedule as for B²; no +1 on the alphas, no class
weights. Two priors, one per label, from the training votes only:
  uniform   Beta(1, 1)
  moment    method of moments for the Beta-Binomial with unequal n (Kleinman 1973): π = mean of k/n;
            Var(k/n) = π(1 − π) [ρ + (1 − ρ) E(1/n)], solved for ρ = 1/(ν + 1); prior Beta(π ν, (1 − π) ν)
Settings: B²'s phase-1 learning rate and weight-decay level for the dataset (results/tuning/<dataset>/selection.json),
seeds 1-5. Checkpoint: lowest validation value of the variant's full loss (NLL + λ KL), saved as "b2kl". Outputs are
(α₀, α₁) as for B², so every B² metric applies. Writes results/ablation_kl/final/<dataset>/b2_kl_<variant>/seed<s>/.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from b2j.losses import alphas, beta_binomial_nll, beta_kl
from b2j.train import load_embeddings, save_final, train_run

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
LAMBDA = 0.05
CKPT = "b2kl"
VARIANTS = ("uniform", "moment")


def moment_prior(k, n):
    """[K, 2] = (p0, p1) per label from training votes k, n [N, K] (method of moments, unequal n)."""
    k, n = k.double(), n.double()
    f = k / n
    pi = f.mean(0)
    v = f.var(0, unbiased=True)
    h = (1.0 / n).mean(0)
    rho = ((v / (pi * (1 - pi))) - h) / (1 - h)
    if not ((rho > 0) & (rho < 1)).all():
        raise ValueError(f"moment prior: rho outside (0, 1): {rho.tolist()}")
    nu = 1.0 / rho - 1.0
    return torch.stack([(1 - pi) * nu, pi * nu], dim=-1)


def make_prior(variant, k, n):
    if variant == "uniform":
        return torch.ones(k.shape[1], 2, dtype=torch.float64)
    if variant == "moment":
        return moment_prior(k, n)
    raise ValueError(variant)


class B2KLLoss(nn.Module):
    """B² NLL + λ KL to a fixed prior per label; sum over heads of the batch mean. prior: [K, 2] = (p0, p1)."""

    def __init__(self, prior, lam=LAMBDA, eps=1e-3):
        super().__init__()
        self.lam, self.eps = lam, eps
        self.register_buffer("prior", torch.as_tensor(prior, dtype=torch.float64))

    def per_sample(self, out, k, n):
        a0, a1 = alphas(out.double(), self.eps)
        p = self.prior.to(a0.device)
        kl = beta_kl(a0, a1, p[:, 0].expand_as(a0), p[:, 1].expand_as(a1))
        return (beta_binomial_nll(a0, a1, k.double(), n.double()) + self.lam * kl).to(out.dtype)     # [B, K]

    def forward(self, out, k, n):
        return self.per_sample(out, k, n).mean(0).sum()


def run(dataset, variants=VARIANTS, seeds=(1, 2, 3, 4, 5), root=RESULTS, settings_root=RESULTS, cache=None,
        epochs=25, data=None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s = json.loads((Path(settings_root) / "tuning" / dataset / "selection.json").read_text())["b2"]
    lr, level = s["lr"], s["wd_level"]
    print(f"{dataset}: B² phase-1 settings lr {lr:g}, level {level:g}; lambda {LAMBDA}", flush=True)
    if data is None:
        data = load_embeddings(dataset, device=dev) if cache is None else load_embeddings(dataset, cache, dev)
    data, labels = data
    _, ktr, ntr, _ = data["train"]
    for variant in variants:
        prior = make_prior(variant, ktr.cpu(), ntr.cpu())
        print(f"{dataset} b2_kl_{variant}: prior (p0, p1) per label "
              + ", ".join(f"{lab} ({a:.4f}, {b:.4f})" for lab, (a, b) in zip(labels, prior.tolist())), flush=True)
        loss_fn = B2KLLoss(prior).to(dev)
        for seed in seeds:
            d = Path(root) / "ablation_kl" / "final" / dataset / f"b2_kl_{variant}" / f"seed{seed}"
            if (d / "run.json").exists():
                print(f"skip {d} (done)", flush=True)
                continue
            r, states = train_run(dataset, data, labels, "b2", lr, level, seed, dev, epochs, loss_fn=loss_fn, ckpt=CKPT)
            r.update({"variant": f"b2_kl_{variant}", "kl_lambda": LAMBDA, "prior": prior.tolist(),
                      "settings_from": "phase-1 B² selection"})
            if r["status"] != "ok":
                d.mkdir(parents=True, exist_ok=True)
                (d / "run.json").write_text(json.dumps(r, indent=1))
                print(f"{dataset} b2_kl_{variant} seed {seed}: DIVERGED ({r['error']})", flush=True)
                continue
            save_final(d, dataset, data, labels, r, states, dev)
            c = r["checkpoints"][CKPT]
            print(f"{dataset} b2_kl_{variant} seed {seed}: checkpoint @ epoch {c['epoch']:g} ({r['seconds']} s)",
                  flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    run(a.dataset, variants=tuple(a.variants), seeds=tuple(a.seeds))
