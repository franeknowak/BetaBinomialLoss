"""Rater-bias variants of B² and soft BCE (datasets where the rater of every vote is known: Endoscapes, AMi-Br).

Model, per image and label: a "typical rater" probability θ, and one learned bias β_r per rater and label (centred over
the panel, so the panel average is the typical rater). Rater r votes yes with probability σ(logit θ + β_r).
  B² + rater bias:       θ ~ Beta(α₁, α₀), (α₀, α₁) = softplus(head) + 1e-3 as in B²; loss = −log P(the ordered votes),
                         the integral over θ by quadrature on the logit scale (logit_grid).
  soft BCE + rater bias: θ = σ(head logit); loss = BCE of each vote against σ(logit + β_r), averaged over the raters
                         (equal to soft BCE when β = 0).
Training as in phase 1 (AdamW, batch 64, 25 epochs, cosine decay per step, 4 evaluations per epoch, same learning rate
and weight-decay level as the loss's phase-1 run on the dataset). The rater biases are logit-scale intercepts, not
network weights: their own step size BETA_LR (same cosine factor) and no weight decay (decay would pull the raters
together). Checkpoint: lowest validation value of the model's own loss.

Evaluation uses the predicted distribution of the vote count k for the known panel: P(k) = ∫ p(θ) PoissonBinomial(k;
σ(logit θ + β_1), …, σ(logit θ + β_R)) dθ, scored with the phase-1 metrics (metrics.compute_pmf); the predicted
fraction is E[k]/n.

  python -m b2j.rater_bias <endoscapes|amibr> [--seeds 1 2 3 4 5] [--losses b2 soft_bce] [--out rater_bias]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from b2j import metrics as M
from b2j.data import load_table
from b2j.heads import MultiHead
from b2j.losses import alphas
from b2j.schedule import BATCH, EPOCHS, EVALS_PER_EPOCH, eval_steps, lr_factors, steps_per_epoch, wd_for_level
from b2j.train import _commit, load_embeddings

DATASETS = ("endoscapes", "amibr")
LOSSES = {"b2": 2, "soft_bce": 1}          # loss -> head outputs per label
BETA_LR = 1e-2                             # step size of the rater biases (logit units); as in the October D8 runs
QUAD_HALF = 129                            # quadrature nodes per half-grid (257 in all)
QUAD_T = 30.0                              # |logit θ| beyond which every vote is certain (tails in closed form)


# ---------------------------------------------------------------- model
class RaterBiasModel(nn.Module):
    """MultiHead plus one bias per label and rater, β [K, R], centred over the raters."""

    def __init__(self, in_dim, num_heads, width, num_raters):
        super().__init__()
        self.head = MultiHead(in_dim, num_heads, width)
        self.bias = nn.Parameter(torch.zeros(num_heads, num_raters))

    def beta(self):
        return self.bias - self.bias.mean(1, keepdim=True)

    def forward(self, z):
        return self.head(z)


# ---------------------------------------------------------------- likelihood
def logit_grid(a0, a1, half=QUAD_HALF, T=QUAD_T):
    """Quadrature of θ ~ Beta(a1, a0) on t = logit θ, where the density ∝ exp(a1 logσ(t) + a0 logσ(−t)). Nodes cluster
    around t* = logit(a1/(a0 + a1)) with the Laplace width s = 1/sqrt(S m (1 − m)); each half [−T, t*], [t*, T] has its
    own sinh-stretched trapezoid rule. Beyond ±T, logσ(t) ≈ t (left) or −t (right), so the tails have closed forms.
    a0, a1 [...] torch. Returns t [..., J], log weight incl. the unnormalised density [..., J], and the log mass of the
    left and right tails [...]; every use is self-normalised."""
    log_a0, log_a1 = torch.log(a0), torch.log(a1)
    ts = torch.clamp(log_a1 - log_a0, -T + 1e-6, T - 1e-6)[..., None]
    s = torch.exp(0.5 * (torch.log(a0 + a1) - log_a0 - log_a1))[..., None]
    cl, cr = torch.arcsinh((ts + T) / s), torch.arcsinh((T - ts) / s)
    h = torch.linspace(0.0, 1.0, half, dtype=a0.dtype, device=a0.device)
    w = torch.full_like(h, 1.0 / (half - 1))
    w[0] = w[-1] = 0.5 / (half - 1)
    t = torch.cat([ts - s * torch.sinh(cl * h), ts + s * torch.sinh(cr * h)], -1)
    jac = torch.cat([s * cl * torch.cosh(cl * h), s * cr * torch.cosh(cr * h)], -1)
    lw = torch.log(torch.cat([w, w]) * jac) + a1[..., None] * F.logsigmoid(t) + a0[..., None] * F.logsigmoid(-t)
    return t, lw, -a1 * T - log_a1, -a0 * T - log_a0


def b2_ordered_nll(a0, a1, beta, y):
    """−log P(ordered votes y) under θ ~ Beta(a1, a0), vote_r ~ Bernoulli(σ(logit θ + β_r)). a0, a1 [B, K];
    beta [K, R]; y [B, K, R] in {0, 1}. Returns [B, K]."""
    t, lw, lt_l, lt_r = logit_grid(a0, a1)
    tb = t[..., None] + beta[None, :, None, :]                                   # [B, K, J, R]
    yy = y[..., None, :]
    lf = (yy * F.logsigmoid(tb) + (1 - yy) * F.logsigmoid(-tb)).sum(-1)
    ninf = torch.full_like(lt_l, -torch.inf)
    all_no, all_yes = y.sum(-1) == 0, (1 - y).sum(-1) == 0
    num = torch.logsumexp(torch.cat([lw + lf, torch.where(all_no, lt_l, ninf)[..., None],
                                     torch.where(all_yes, lt_r, ninf)[..., None]], -1), -1)
    den = torch.logsumexp(torch.cat([lw, lt_l[..., None], lt_r[..., None]], -1), -1)
    return -(num - den)


def soft_bce_votes(z, beta, y):
    """Mean over raters of the BCE of each vote against σ(z + β_r). z [B, K]; beta [K, R]; y [B, K, R]. Returns [B, K]."""
    return F.binary_cross_entropy_with_logits(z[..., None] + beta[None], y, reduction="none").mean(-1)


def per_sample(loss, out, beta, y):
    if loss == "b2":
        a0, a1 = alphas(out.double())
        return b2_ordered_nll(a0, a1, beta.double(), y.double()).to(out.dtype)
    return soft_bce_votes(out[..., 0], beta, y)


def loss_value(loss, out, beta, y):
    return per_sample(loss, out, beta, y).mean(0).sum()


# ---------------------------------------------------------------- predicted count distribution
def poisson_binomial(p):
    """p [..., R] -> P(k = 0..R) [..., R + 1] for independent Bernoulli(p_r)."""
    pmf = torch.zeros(p.shape[:-1] + (p.shape[-1] + 1,), dtype=p.dtype, device=p.device)
    pmf[..., 0] = 1.0
    for r in range(p.shape[-1]):
        pr = p[..., r:r + 1]
        pmf = torch.cat([pmf[..., :1] * (1 - pr), pmf[..., 1:] * (1 - pr) + pmf[..., :-1] * pr], -1)
    return pmf


def count_pmf(loss, out, beta):
    """P(k = 0..R) for the known panel [N, K, R + 1], float64. out [N, K, w]; beta [K, R]."""
    out, beta = out.double(), beta.double()
    R = beta.shape[1]
    if loss == "soft_bce":
        return poisson_binomial(torch.sigmoid(out[..., 0][..., None] + beta[None]))
    a0, a1 = alphas(out)
    t, lw, lt_l, lt_r = logit_grid(a0, a1)
    logw = torch.cat([lw, lt_l[..., None], lt_r[..., None]], -1)
    w = torch.softmax(logw, -1)                                                  # [N, K, J + 2]
    pb = poisson_binomial(torch.sigmoid(t[..., None] + beta[None, :, None, :]))  # [N, K, J, R + 1]
    pmf = (w[..., :-2, None] * pb).sum(-2)
    pmf[..., 0] += w[..., -2]                                                    # left tail: every vote no
    pmf[..., R] += w[..., -1]                                                    # right tail: every vote yes
    return pmf


# ---------------------------------------------------------------- data
def load_data(dataset, device="cpu"):
    """Phase-1 embeddings and splits, plus each image's ordered votes y [N, K, R] (0/1) aligned to the cache ids."""
    data, labels = load_embeddings(dataset, device=device)
    table = load_table(dataset)
    if table.votes is None:
        raise ValueError(f"{dataset} has no rater identities")
    row = {str(i): r for r, i in enumerate(table.ids)}
    out = {}
    for s, (X, k, n, ids) in data.items():
        v = table.votes[[row[str(i)] for i in ids]]                              # [N, R, K]
        if not np.isfinite(v).all():
            raise ValueError("missing votes: the rater-bias runs assume every rater voted on every image")
        y = torch.as_tensor(v.transpose(0, 2, 1), dtype=torch.float32, device=device)
        if not torch.equal(y.sum(-1), k):
            raise ValueError(f"{dataset} {s}: votes do not sum to the cached k")
        out[s] = (X, k, n, ids, y)
    return out, labels


# ---------------------------------------------------------------- training
def _outputs(model, X, chunk=4096):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + chunk]) for i in range(0, len(X), chunk)])


def _full_loss(loss, model, X, y, chunk=4096):
    """Own loss over a whole split, = the batch-mean loss summed over labels."""
    with torch.no_grad():
        out, beta = _outputs(model, X, chunk), model.beta()
        tot = sum(per_sample(loss, out[i:i + chunk], beta, y[i:i + chunk]).double().sum(0)
                  for i in range(0, len(X), chunk))
        return float((tot / len(X)).sum())


def train_run(dataset, data, labels, loss, lr, wd_level, seed, device="cpu", epochs=EPOCHS):
    """One rater-bias run. Returns (run dict, head+bias state at the checkpoint on CPU or None if diverged)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    Xtr, ktr, ntr, _, ytr = data["train"]
    Xva, _, _, _, yva = data["val"]
    N, K = ktr.shape
    R = ytr.shape[-1]
    wd = wd_for_level(wd_level, lr, N, epochs)
    torch.manual_seed(seed)
    model = RaterBiasModel(Xtr.shape[1], K, LOSSES[loss], R).to(device)
    opt = torch.optim.AdamW([{"params": model.head.parameters(), "lr": lr, "weight_decay": wd, "base": lr},
                             {"params": [model.bias], "lr": BETA_LR, "weight_decay": 0.0, "base": BETA_LR}])
    gen = torch.Generator().manual_seed(seed)
    spe = steps_per_epoch(N)
    factors = lr_factors(epochs * spe)
    evals = eval_steps(N, epochs)
    curve, best, state = [], [float("inf"), None], None
    t0 = time.time()
    status, error = "ok", None
    try:
        step = 0
        for _ in range(epochs):
            perm = torch.randperm(N, generator=gen).to(device)
            for b in range(spe):
                idx = perm[b * BATCH:(b + 1) * BATCH]
                for g in opt.param_groups:
                    g["lr"] = g["base"] * float(factors[step])
                model.train()
                lv = loss_value(loss, model(Xtr[idx]), model.beta(), ytr[idx])
                if not torch.isfinite(lv):
                    raise FloatingPointError(f"non-finite training loss at step {step}")
                opt.zero_grad(set_to_none=True)
                lv.backward()
                gnorm = torch.sqrt(sum((p.grad.double() ** 2).sum() for p in model.parameters())).item()
                if not np.isfinite(gnorm):
                    raise FloatingPointError(f"non-finite gradient at step {step}")
                opt.step()
                step += 1
                if len(curve) < len(evals) and step == evals[len(curve)]:
                    tr, va = _full_loss(loss, model, Xtr, ytr), _full_loss(loss, model, Xva, yva)
                    e = len(curve) + 1
                    curve.append({"eval": e, "epoch": e / EVALS_PER_EPOCH, "step": step, "train_loss": tr,
                                  "val_loss": va, "grad_norm": gnorm, "beta": model.beta().detach().cpu().tolist()})
                    if not (np.isfinite(tr) and np.isfinite(va)):
                        raise FloatingPointError(f"non-finite evaluation loss at evaluation {e}")
                    if va < best[0]:
                        best = [va, e]
                        state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
    except FloatingPointError as err:
        status, error, best, state = "diverged", str(err), [None, None], None
    run = {"dataset": dataset, "loss": loss, "variant": "rater_bias", "seed": seed, "labels": labels, "lr": lr,
           "beta_lr": BETA_LR, "wd_level": wd_level, "wd": wd, "epochs": epochs, "batch": BATCH, "n_train": N,
           "n_raters": R, "commit": _commit(), "seconds": round(time.time() - t0, 1), "status": status, "error": error,
           "best_val_loss": best[0], "best_val_loss_eval": best[1],
           "checkpoint": None if best[1] is None else {"eval": best[1], "epoch": best[1] / EVALS_PER_EPOCH},
           "curve": curve}
    return run, state


def save_final(out_dir, data, labels, run, state, device="cpu"):
    """Head + bias weights, val/test outputs (head outputs as in phase 1, plus β) and the metrics on train/val/test."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loss = run["loss"]
    X0, _, _, _, y0 = data["train"]
    model = RaterBiasModel(X0.shape[1], len(labels), LOSSES[loss], y0.shape[-1]).to(device)
    model.load_state_dict(state)
    torch.save(state, out_dir / "head_own.pt")
    beta = model.beta().detach()
    images, mets = [], {}
    for s in ("train", "val", "test"):
        X, k, n, ids, _ = data[s]
        out = _outputs(model, X)
        pmf = torch.cat([count_pmf(loss, out[i:i + 4096], beta) for i in range(0, len(X), 4096)]).cpu().numpy()
        kk, nn_ = k.cpu().numpy(), n.cpu().numpy()
        mets[s] = M.compute_pmf(pmf, kk, nn_, labels)
        if s == "train":
            continue
        raw = torch.stack(alphas(out.double()), -1) if loss == "b2" else out.double()
        o = raw.cpu().numpy()
        images += [{"id": str(ids[i]), "split": s, "out": o[i].tolist(), "k": [int(v) for v in kk[i]],
                    "n": [int(v) for v in nn_[i]]} for i in range(len(ids))]
    (out_dir / "outputs_own.json").write_text(json.dumps(
        {"dataset": run["dataset"], "loss": loss, "variant": "rater_bias", "seed": run["seed"], "labels": labels,
         "outputs": ["alpha0", "alpha1"] if loss == "b2" else ["logit"], "beta": beta.cpu().tolist(),
         "images": images}))
    run["checkpoint"]["metrics"] = mets
    (out_dir / "run.json").write_text(json.dumps(run, indent=1))


def run(dataset, seeds=(1, 2, 3, 4, 5), losses=tuple(LOSSES), root=None, out="rater_bias", epochs=EPOCHS):
    from b2j.experiment import RESULTS, _device
    root = Path(root) if root is not None else RESULTS
    sel = json.loads((root / "tuning" / dataset / "selection.json").read_text())
    dev = _device()
    data, labels = load_data(dataset, device=dev)
    print(dataset, {s: len(v[3]) for s, v in data.items()}, flush=True)
    for loss in losses:
        s = sel[loss]
        for seed in seeds:
            d = root / out / dataset / loss / f"seed{seed}"
            if (d / "run.json").exists():
                print(f"skip {d} (done)", flush=True)
                continue
            r, state = train_run(dataset, data, labels, loss, s["lr"], s["wd_level"], seed, dev, epochs)
            if r["status"] != "ok":
                d.mkdir(parents=True, exist_ok=True)
                (d / "run.json").write_text(json.dumps(r, indent=1))
                print(f"{dataset} {loss} seed {seed}: DIVERGED ({r['error']})", flush=True)
                continue
            save_final(d, data, labels, r, state, dev)
            print(f"{dataset} {loss} seed {seed}: lr {s['lr']:g} level {s['wd_level']:g} wd {r['wd']:.4g}, "
                  f"checkpoint epoch {r['checkpoint']['epoch']:g}, β {np.round(r['curve'][r['best_val_loss_eval'] - 1]['beta'], 2).tolist()} "
                  f"({r['seconds']} s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=DATASETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--losses", nargs="+", choices=list(LOSSES), default=list(LOSSES))
    ap.add_argument("--out", default="rater_bias")
    a = ap.parse_args()
    run(a.dataset, seeds=tuple(a.seeds), losses=tuple(a.losses), out=a.out)
