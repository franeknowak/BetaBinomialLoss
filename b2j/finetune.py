"""Fine-tuning ablation (supplement): DINOv3 ViT-B/16 adapted with LoRA, Endoscapes, one loss per job.

  python -m b2j.finetune <dataset> --loss b2 [--root results/lora]     tuning (lr grid + edge rule), then final seeds

Agreed with the user (2026-11-23):
- LoRA on all 12 blocks, on the query and value projections (timm's fused qkv: the q and v slices; k untouched),
  rank 8, alpha 16 (scale alpha / rank = 2), LoRA dropout 0.1. A ~ Kaiming-uniform, B = 0, so training starts from the
  pretrained encoder exactly. Everything else in the encoder stays frozen.
- Head and embedding as in phase 1: [CLS] after the final LayerNorm -> the same MLP head per label.
- Training augmentation, after the 480 centre crop: rotation by a uniform angle in [-10, 10] degrees, then the
  largest centre square free of empty corners for that angle (side 480 / (cos + sin), minus a 2 px margin), then a
  random resized crop of 85-100% of the area with aspect ratio 9:10-10:9, resized to 256 (bicubic). No flips.
  Validation / test: the phase-1 transform (480 centre crop, 256 bicubic).
- AdamW, one learning rate for LoRA and head, weight decay 0.01, batch 64, 15 epochs, cosine decay per step to 0, no
  warmup; bf16 autocast for the encoder (losses in float64 as in phase 1).
- Tuning: learning rates 1e-4, 5e-4, 1e-3, seed 0; choice by the lowest centred-moving-average (5 evaluations)
  validation value of the loss's own loss, with the grid-extension rule at the edge (1-5-10 pattern). Final: seeds 1-5.
- Checkpoint: lowest raw validation value of the own loss (as phase 1); 4 evaluations per epoch.
Evaluations are full passes over validation in eval mode; the logged train loss is the mean of the training batches
since the previous evaluation (a full pass over the training images at every evaluation would cost more than the
training itself). Final runs save outputs_<ckpt>.json (val + test) in the phase-1 format and metrics on val / test.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from b2j import metrics as M
from b2j.data import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE, ImageSet, load_table, transform
from b2j.experiment import SMOOTH_WINDOW, next_lr, smoothed_min
from b2j.heads import MultiHead
from b2j.losses import alphas, make_loss
from b2j.schedule import BATCH, EVALS_PER_EPOCH, eval_steps, lr_factors, steps_per_epoch
from b2j.train import CKPT, OUTPUT_NAMES, _commit, _num

REPO = Path(__file__).resolve().parents[1]
MODEL = "vit_base_patch16_dinov3.lvd1689m"
RANK, ALPHA, LORA_DROPOUT = 8, 16, 0.1
LRS = (1e-4, 5e-4, 1e-3)
WD, EPOCHS = 0.01, 15
MAX_ANGLE, CROP_SCALE, CROP_RATIO, CROP_MARGIN = 10.0, (0.85, 1.0), (0.9, 1 / 0.9), 2
FINAL_SEEDS, TUNING_SEED = (1, 2, 3, 4, 5), 0


# ---------------------------------------------------------------- LoRA
class LoRAqv(nn.Module):
    """A frozen fused qkv Linear plus low-rank updates on its q and v output slices: y = W x + s [B_q A_q x, 0, B_v A_v x]."""

    def __init__(self, base: nn.Linear, rank=RANK, alpha=ALPHA, dropout=LORA_DROPOUT):
        super().__init__()
        d_in, d_out = base.in_features, base.out_features
        if d_out != 3 * d_in:
            raise ValueError(f"expected a fused qkv layer, got {d_in} -> {d_out}")
        self.base, self.d, self.scale = base, d_in, alpha / rank
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.drop = nn.Dropout(dropout)
        self.a_q, self.a_v = nn.Linear(d_in, rank, bias=False), nn.Linear(d_in, rank, bias=False)
        self.b_q, self.b_v = nn.Linear(rank, d_in, bias=False), nn.Linear(rank, d_in, bias=False)
        for a in (self.a_q, self.a_v):
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
        for b in (self.b_q, self.b_v):
            nn.init.zeros_(b.weight)

    def forward(self, x):
        h = self.drop(x)
        dq, dv = self.b_q(self.a_q(h)) * self.scale, self.b_v(self.a_v(h)) * self.scale
        return self.base(x) + torch.cat([dq, torch.zeros_like(dq), dv], dim=-1)


def lora_encoder(pretrained=True, model_name=MODEL):
    """DINOv3 with every encoder weight frozen and LoRA on q and v in every block. Returns (model, number of blocks)."""
    import timm
    enc = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    for p in enc.parameters():
        p.requires_grad_(False)
    for blk in enc.blocks:
        attn = blk.attn
        if getattr(attn, "q_bias", None) is not None or attn.qkv is None:
            raise ValueError("LoRA wrapper expects attention computed as self.qkv(x) (no separate q/k/v biases)")
        attn.qkv = LoRAqv(attn.qkv)
    return enc, len(enc.blocks)


class FineTuneModel(nn.Module):
    def __init__(self, encoder, num_labels, width):
        super().__init__()
        self.encoder = encoder
        self.head = MultiHead(encoder.num_features, num_labels, width)

    def forward(self, x):
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            z = self.encoder.forward_features(x)[:, 0]          # [CLS] after the final LayerNorm, as phase 1
        return self.head(z.float())

    def trainable_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if ".a_q." in k or ".a_v." in k or ".b_q." in k or ".b_v." in k or k.startswith("head.")}


# ---------------------------------------------------------------- augmentation
def corner_free_side(side, angle_deg):
    """Side of the largest axis-aligned centre square inside a side x side square rotated by angle_deg."""
    t = math.radians(abs(angle_deg))
    return side / (math.cos(t) + math.sin(t))


class RotateCrop:
    """Rotate by a uniform angle in [-max_angle, max_angle], then centre-crop the corner-free square (minus a margin)."""

    def __init__(self, max_angle=MAX_ANGLE, margin=CROP_MARGIN):
        self.max_angle, self.margin = max_angle, margin

    def __call__(self, img):
        from torchvision.transforms import functional as TF
        from torchvision.transforms import InterpolationMode
        angle = random.uniform(-self.max_angle, self.max_angle)
        side = min(img.size)
        img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0)
        return TF.center_crop(img, int(math.floor(corner_free_side(side, angle))) - self.margin)


def train_transform(dataset, size=None):
    from torchvision import transforms as T
    size = INPUT_SIZE[dataset] if size is None else size
    steps = [T.CenterCrop(480)] if dataset == "endoscapes" else []
    steps += [RotateCrop(),
              T.RandomResizedCrop(size, scale=CROP_SCALE, ratio=CROP_RATIO,
                                  interpolation=T.InterpolationMode.BICUBIC, antialias=True),
              T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)


def _worker_init(wid):
    s = torch.initial_seed() % 2 ** 32
    random.seed(s)
    np.random.seed(s)


# ---------------------------------------------------------------- one run
def _outputs(model, loader, device):
    model.eval()
    outs, ks, ns, idx = [], [], [], []
    with torch.no_grad():
        for img, k, n, i in loader:
            outs.append(model(img.to(device, non_blocking=True)).float().cpu())
            ks.append(k), ns.append(n), idx.append(i)
    return torch.cat(outs), torch.cat(ks), torch.cat(ns), torch.cat(idx)


def train_run(sets, labels, loss_name, lr, seed, device, epochs=EPOCHS, workers=8, model_name=MODEL, pretrained=True,
              batch=BATCH):
    """sets: {"train": ImageSet (augmenting), "val": ImageSet}. Returns (run dict, trainable state of the checkpoint)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    loss_fn, width = make_loss(loss_name)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    enc, n_blocks = lora_encoder(pretrained, model_name)
    model = FineTuneModel(enc, len(labels), width).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WD)
    N = len(sets["train"])
    gen = torch.Generator().manual_seed(seed)
    tr = torch.utils.data.DataLoader(sets["train"], batch_size=batch, shuffle=True, generator=gen, drop_last=False,
                                     num_workers=workers, worker_init_fn=_worker_init, pin_memory=device != "cpu",
                                     persistent_workers=workers > 0)
    va = torch.utils.data.DataLoader(sets["val"], batch_size=128, shuffle=False, num_workers=workers,
                                     pin_memory=device != "cpu")
    spe = steps_per_epoch(N, batch)
    factors = lr_factors(epochs * spe)
    evals = eval_steps(N, epochs, batch=batch)
    ck = CKPT[loss_name]
    best, state, curve, since = None, None, [], []
    status, error, t0, step = "ok", None, time.time(), 0
    try:
        for epoch in range(epochs):
            for img, k, n, _ in tr:
                lr_t = lr * float(factors[step])
                for g in opt.param_groups:
                    g["lr"] = lr_t
                model.train()
                loss = loss_fn(model(img.to(device, non_blocking=True)), k.to(device), n.to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at step {step}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = torch.sqrt(sum((p.grad.double() ** 2).sum() for p in params)).item()
                if not np.isfinite(gnorm):
                    raise FloatingPointError(f"non-finite gradient at step {step}")
                opt.step()
                since.append(loss.item())
                step += 1
                if len(curve) < len(evals) and step == evals[len(curve)]:
                    out, kv, nv, _ = _outputs(model, va, device)
                    with torch.no_grad():
                        va_loss = loss_fn(out, kv, nv).item()
                    va_bacc = M.macro_bacc(loss_name, out, kv, nv)
                    e = len(curve) + 1
                    curve.append({"eval": e, "epoch": e / EVALS_PER_EPOCH, "step": step, "lr": lr_t,
                                  "train_loss": float(np.mean(since)), "val_loss": va_loss, "val_bacc": va_bacc,
                                  "grad_norm": gnorm, "seconds": round(time.time() - t0, 1)})
                    since = []
                    if not np.isfinite(va_loss):
                        raise FloatingPointError(f"non-finite validation loss at evaluation {e}")
                    if best is None or va_loss < best[0]:
                        best, state = (va_loss, e), model.trainable_state()
                    print(f"  eval {e}/{len(evals)} epoch {e / EVALS_PER_EPOCH:g}: val loss {va_loss:.5f} "
                          f"({curve[-1]['seconds']} s)", flush=True)
    except FloatingPointError as err:
        status, error, best, state = "diverged", str(err), None, None
    run = {"dataset": "endoscapes", "loss": loss_name, "seed": seed, "labels": labels, "lr": lr, "wd": WD,
           "epochs": epochs, "batch": batch, "steps_per_epoch": spe, "evals_per_epoch": EVALS_PER_EPOCH, "n_train": N,
           "lora": {"blocks": n_blocks, "targets": ["q", "v"], "rank": RANK, "alpha": ALPHA, "dropout": LORA_DROPOUT},
           "augmentation": {"rotation_deg": MAX_ANGLE, "corner_free_crop_margin_px": CROP_MARGIN,
                            "crop_scale": list(CROP_SCALE), "crop_ratio": list(CROP_RATIO)},
           "precision": "bf16 autocast (encoder)", "n_trainable": sum(p.numel() for p in params),
           "commit": _commit(), "seconds": round(time.time() - t0, 1), "status": status, "error": error,
           "best_val_loss": None if best is None else best[0], "best_val_loss_eval": None if best is None else best[1],
           "checkpoints": {} if best is None else
           {ck: {"eval": best[1], "epoch": best[1] / EVALS_PER_EPOCH, "step": evals[best[1] - 1]}},
           "curve": curve}
    return run, state


def save_final(out_dir, run, state, sets_eval, labels, device, model_name=MODEL, pretrained=True, workers=8):
    """Outputs (val + test, phase-1 format) and metrics of the checkpoint; trainable weights as head_<ckpt>.pt."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loss_name, ck = run["loss"], CKPT[run["loss"]]
    _, width = make_loss(loss_name)
    enc, _ = lora_encoder(pretrained, model_name)
    model = FineTuneModel(enc, len(labels), width).to(device)
    missing = model.load_state_dict(state, strict=False)
    if missing.unexpected_keys or any(".a_" in k or ".b_" in k or k.startswith("head.") for k in missing.missing_keys):
        raise ValueError(f"checkpoint does not match the model: {missing}")
    torch.save(state, out_dir / f"head_{ck}.pt")
    images, mets = [], {}
    for s in ("val", "test"):
        ds = sets_eval[s]
        out, k, n, idx = _outputs(model, torch.utils.data.DataLoader(ds, batch_size=128, num_workers=workers), device)
        mets[s] = M.compute(loss_name, out, k, n, labels)
        raw = torch.stack(alphas(out.double()), -1) if loss_name == "b2" else out.double()
        o, kk, nn_ = raw.numpy(), k.numpy(), n.numpy()
        ids = ds.table.ids
        images += [{"id": str(ids[int(i)]), "split": s, "out": o[j].tolist(), "k": [_num(v) for v in kk[j]],
                    "n": [_num(v) for v in nn_[j]]} for j, i in enumerate(idx)]
    (out_dir / f"outputs_{ck}.json").write_text(json.dumps(
        {"dataset": "endoscapes", "loss": loss_name, "seed": run["seed"], "checkpoint": ck, "labels": labels,
         "outputs": OUTPUT_NAMES[loss_name], "images": images}))
    run["checkpoints"][ck]["metrics"] = mets
    (out_dir / "run.json").write_text(json.dumps(run, indent=1))


# ---------------------------------------------------------------- tuning, selection, final
def splits(dataset):
    table = load_table(dataset)
    idx = {s: np.flatnonzero(table.split == s) for s in ("train", "val", "test")}
    aug, plain = ImageSet(table, dataset, train_transform(dataset)), ImageSet(table, dataset, transform(dataset))
    labels = {"endoscapes": ["C1", "C2", "C3"]}[dataset]
    return {"train": aug.subset(idx["train"]), "val": plain.subset(idx["val"]),
            "test": plain.subset(idx["test"])}, labels


def select(root, dataset, loss):
    every = [json.loads(p.read_text()) for p in (Path(root) / "tuning" / dataset / loss).glob("*/run.json")]
    runs = [r for r in every if r["status"] == "ok"]
    for r in runs:
        r["smoothed_val_loss"], r["smoothed_eval"] = smoothed_min(r["curve"])
    runs.sort(key=lambda r: (r["smoothed_val_loss"], r["lr"]))
    r, tried = runs[0], sorted({x["lr"] for x in runs})
    edge = "low" if r["lr"] == tried[0] else "high" if r["lr"] == tried[-1] else None
    sel = {"lr": r["lr"], "smooth_window": SMOOTH_WINDOW, "smoothed_val_loss": r["smoothed_val_loss"],
           "smoothed_epoch": r["smoothed_eval"] / EVALS_PER_EPOCH, "best_epoch": r["best_val_loss_eval"] / EVALS_PER_EPOCH,
           "lr_at_edge": edge, "lrs_tried": tried, "n_runs": len(runs), "n_diverged": len(every) - len(runs)}
    print(f"{dataset} {loss}: lr {r['lr']:g}, smoothed val loss {r['smoothed_val_loss']:.5f} at epoch "
          f"{sel['smoothed_epoch']:g}/{r['epochs']}" + (f"  ** lr at {edge} edge **" if edge else ""), flush=True)
    (Path(root) / "tuning" / dataset / loss / "selection.json").write_text(json.dumps(sel, indent=1))
    return sel


def main(dataset, loss, root, workers=8, max_ext=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sets, labels = splits(dataset)
    lrs, ext = list(LRS), 0
    while True:
        for lr in lrs:
            d = Path(root) / "tuning" / dataset / loss / f"lr{lr:g}"
            if (d / "run.json").exists():
                continue
            print(f"{dataset} {loss} tuning lr {lr:g}", flush=True)
            run, _ = train_run(sets, labels, loss, lr, TUNING_SEED, device, workers=workers)
            d.mkdir(parents=True, exist_ok=True)
            (d / "run.json").write_text(json.dumps(run, indent=1))
            print(f"{dataset} {loss} lr {lr:g}: {run['status']}, best val loss {run['best_val_loss']} at eval "
                  f"{run['best_val_loss_eval']} ({run['seconds']} s)", flush=True)
        sel = select(root, dataset, loss)
        if not sel["lr_at_edge"]:
            break
        nxt = next_lr(sel["lr"], sel["lr_at_edge"])
        if nxt in sel["lrs_tried"] or ext >= max_ext:
            raise SystemExit(f"{dataset} {loss}: lr still at the {sel['lr_at_edge']} edge; final runs not started")
        ext += 1
        print(f"{dataset} {loss}: grid extension to lr {nxt:g}", flush=True)
        lrs = [nxt]
    for seed in FINAL_SEEDS:
        d = Path(root) / "final" / dataset / loss / f"seed{seed}"
        if (d / "run.json").exists():
            continue
        run, state = train_run(sets, labels, loss, sel["lr"], seed, device, workers=workers)
        if run["status"] != "ok":
            d.mkdir(parents=True, exist_ok=True)
            (d / "run.json").write_text(json.dumps(run, indent=1))
            print(f"{dataset} {loss} seed {seed}: DIVERGED ({run['error']})", flush=True)
            continue
        save_final(d, run, state, sets, labels, device, workers=workers)
        print(f"{dataset} {loss} seed {seed}: checkpoint @ epoch {run['checkpoints'][CKPT[loss]]['epoch']:g} "
              f"({run['seconds']} s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["endoscapes"])
    ap.add_argument("--loss", required=True, choices=list(CKPT))
    ap.add_argument("--root", default=str(REPO / "results" / "lora"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main(a.dataset, a.loss, a.root, a.workers)
