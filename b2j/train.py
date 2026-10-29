"""One training run of a classifier head on cached embeddings (PLAN §1.3-1.9).

AdamW, batch 64 (shuffled by the seed), 25 epochs, cosine decay per step from the learning rate to 0, no warmup,
no gradient clipping, no early stopping. At each of the 100 evaluations (4 per epoch): a full pass over train and
val in eval mode (dropout off) gives train loss, val loss and val BACC (macro), plus the gradient norm of the last
step and the learning rate. A NaN/inf loss or gradient stops the run: it is returned with status "diverged", the curve
up to that point and no checkpoints (agreed: diverged settings are excluded from selection).

Checkpoints (raw best evaluation, ties -> the earlier one): "bacc" = highest val BACC (every loss); "nll" = lowest
val loss for B² (its Beta-Binomial NLL); "softbce" = lowest val loss for soft BCE; "hardbce" = lowest val loss for
hard BCE. `best_val_loss` (lowest val loss,
any loss) is what tuning selects on.

Tuning mode returns run.json content only. Final mode also writes, per checkpoint: head_<ckpt>.pt (head weights),
outputs_<ckpt>.json (val + test: id, split, outputs per label = logit for the BCE losses, (α₀, α₁) for B², k, n) and phase-1 metrics on train/val/test.
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from b2j import metrics as M
from b2j.heads import MultiHead
from b2j.losses import alphas, make_loss
from b2j.schedule import BATCH, EPOCHS, EVALS_PER_EPOCH, eval_steps, lr_factors, steps_per_epoch, wd_for_level

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "cache" / "embeddings"
EXTRA_CKPT = {"b2": "nll", "soft_bce": "softbce", "hard_bce": "hardbce"}   # second checkpoint (lowest val loss)
OUTPUT_NAMES = {"hard_bce": ["logit"], "soft_bce": ["logit"], "b2": ["alpha0", "alpha1"]}   # saved per label


def load_embeddings(dataset, cache=CACHE, device="cpu"):
    """{split: (X [N, 768], k [N, K], n [N, K], ids [N])} as float32 tensors on `device`, plus the label names."""
    z = np.load(Path(cache) / f"{dataset}.npz")
    data = {}
    for s in ("train", "val", "test"):
        m = z["split"] == s
        data[s] = tuple(torch.as_tensor(a[m], dtype=torch.float32, device=device) for a in (z["emb"], z["k"], z["n"])) \
            + (z["ids"][m],)
    return data, [str(x) for x in z["labels"]]


def _outputs(model, X, chunk=8192):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + chunk]) for i in range(0, len(X), chunk)])


def _commit():
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


def train_run(dataset, data, labels, loss_name, lr, wd_level, seed, device="cpu", epochs=EPOCHS):
    """Train one head. Returns (run dict, {checkpoint: head state_dict on CPU}); run["status"] is "ok" or "diverged"."""
    torch.backends.cuda.matmul.allow_tf32 = False
    Xtr, ktr, ntr, _ = data["train"]
    Xva, kva, nva, _ = data["val"]
    N, K = ktr.shape
    loss_fn, width = make_loss(loss_name)
    wd = wd_for_level(wd_level, lr, N, epochs)
    torch.manual_seed(seed)
    model = MultiHead(Xtr.shape[1], K, width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    gen = torch.Generator().manual_seed(seed)
    spe = steps_per_epoch(N)
    factors = lr_factors(epochs * spe)
    evals = eval_steps(N, epochs)
    ckpt_names = ["bacc", EXTRA_CKPT[loss_name]]
    best = {c: None for c in ckpt_names}                    # (value, eval index)
    states, curve = {}, []
    best_loss = [float("inf"), None]
    t0 = time.time()

    def optimise():
        step = 0
        for epoch in range(epochs):
            perm = torch.randperm(N, generator=gen).to(device)
            for b in range(spe):
                idx = perm[b * BATCH:(b + 1) * BATCH]
                lr_t = lr * float(factors[step])
                for g in opt.param_groups:
                    g["lr"] = lr_t
                model.train()
                loss = loss_fn(model(Xtr[idx]), ktr[idx], ntr[idx])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at step {step}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = torch.sqrt(sum((p.grad.double() ** 2).sum() for p in model.parameters())).item()
                if not np.isfinite(gnorm):
                    raise FloatingPointError(f"non-finite gradient at step {step}")
                opt.step()
                step += 1
                if len(curve) < len(evals) and step == evals[len(curve)]:
                    evaluate(step, lr_t, gnorm)

    def evaluate(step, lr_t, gnorm):
        with torch.no_grad():
            out_tr, out_va = _outputs(model, Xtr), _outputs(model, Xva)
            tr_loss, va_loss = loss_fn(out_tr, ktr, ntr).item(), loss_fn(out_va, kva, nva).item()
        va_bacc = M.macro_bacc(loss_name, out_va, kva, nva)
        e = len(curve) + 1
        curve.append({"eval": e, "epoch": e / EVALS_PER_EPOCH, "step": step, "lr": lr_t,
                      "train_loss": tr_loss, "val_loss": va_loss, "val_bacc": va_bacc, "grad_norm": gnorm})
        if not (np.isfinite(tr_loss) and np.isfinite(va_loss)):
            raise FloatingPointError(f"non-finite evaluation loss at evaluation {e}")
        if not np.isfinite(va_bacc):
            raise FloatingPointError(f"validation BACC is not finite at evaluation {e}")
        if va_loss < best_loss[0]:
            best_loss[:] = [va_loss, e]
        for c in ckpt_names:
            v = va_bacc if c == "bacc" else -va_loss                # higher is better; strict -> earlier wins ties
            if best[c] is None or v > best[c][0]:
                best[c] = (v, e)
                states[c] = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}

    status, error = "ok", None
    try:
        optimise()
    except FloatingPointError as err:
        status, error = "diverged", str(err)
        best, states, best_loss = {c: None for c in ckpt_names}, {}, [None, None]
    run = {"dataset": dataset, "loss": loss_name, "seed": seed, "labels": labels, "lr": lr, "wd_level": wd_level,
           "wd": wd, "epochs": epochs, "batch": BATCH, "steps_per_epoch": spe, "evals_per_epoch": EVALS_PER_EPOCH,
           "n_train": N, "commit": _commit(), "seconds": round(time.time() - t0, 1), "status": status, "error": error,
           "best_val_loss": best_loss[0], "best_val_loss_eval": best_loss[1],
           "checkpoints": {c: {"eval": best[c][1], "epoch": best[c][1] / EVALS_PER_EPOCH, "step": evals[best[c][1] - 1]}
                           for c in ckpt_names if best[c] is not None},
           "curve": curve}
    return run, states


def _num(x):
    x = float(x)
    return int(x) if x.is_integer() else x


def save_final(out_dir, dataset, data, labels, run, states, device="cpu"):
    """Final mode: per checkpoint, head weights, val/test outputs and phase-1 metrics on train/val/test."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loss_name = run["loss"]
    _, width = make_loss(loss_name)
    for c, state in states.items():
        model = MultiHead(data["train"][0].shape[1], len(labels), width).to(device)
        model.load_state_dict(state)
        torch.save(state, out_dir / f"head_{c}.pt")
        images, mets = [], {}
        for s in ("train", "val", "test"):
            X, k, n, ids = data[s]
            out = _outputs(model, X)
            mets[s] = M.compute(loss_name, out, k, n, labels, dataset)
            if s == "train":
                continue
            raw = torch.stack(alphas(out.double()), -1) if loss_name == "b2" else out.double()   # alphas / logits
            o, kk, nn_ = raw.cpu().numpy(), k.cpu().numpy(), n.cpu().numpy()
            images += [{"id": str(ids[i]), "split": s, "out": o[i].tolist(),
                        "k": [_num(v) for v in kk[i]], "n": [_num(v) for v in nn_[i]]} for i in range(len(ids))]
        (out_dir / f"outputs_{c}.json").write_text(json.dumps(
            {"dataset": dataset, "loss": loss_name, "seed": run["seed"], "checkpoint": c, "labels": labels,
             "outputs": OUTPUT_NAMES[loss_name], "images": images}))
        run["checkpoints"][c]["metrics"] = mets
    (out_dir / "run.json").write_text(json.dumps(run, indent=1))
