"""Module c: one trainer for every model variant, on cached embeddings (frozen encoder) or on images (fine-tuning).

The trainer evaluates on validation only, evals_per_epoch times per epoch, over a short fixed budget with no
early stopping (plan amendment 2026-10-14 #5). At each evaluation it fits the variant's spread scale
(posthoc.fit_spread) and records two selection criteria:
  A = validation mAP of the discrimination score m against the majority vote;
  B = validation vote NLL per vote after the spread step (Global-S for hard and soft BCE, c for B²).
  DUP has no panel prediction: A and B are both its validation loss.
Each checkpoint is the evaluation with the best centred moving average (window cfg.smooth) of its criterion;
the kept weights are the real state at that evaluation.
At each evaluation it also scores a fixed subset of training images (as many as validation; eval mode, dropout
off) with the same metrics and the spread fitted on validation at that evaluation (train_map,
train_nll_per_vote / train_dup_bce): train and validation curves on one footing (05_plan amendment 2026-10-16 #6).
The subset is drawn with its own generator, so training is unchanged.
After the last epoch it writes, for A and B: validation and test raw head outputs, the fitted scale and the head
weights. Test labels never reach the trainer (Split.k is None for test), no test metric is computed or printed,
and no encoder weights or per-epoch checkpoints are written (storage policy, plan amendment 2026-10-14 #3).
See journal_plan/specs/c_cache_trainer.md.
"""
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit
from sklearn.metrics import average_precision_score

from b2j import posthoc
from b2j.data import ImageSet, LabelTable
from b2j.losses import STUCK_LOGIT, split_target
from b2j.variants import VARIANTS, build


BETA_LR = 1e-2                        # D8: Adam step size of the rater biases (logit units), not tuned


@dataclass
class TrainConfig:
    variant: str
    lr: float = 3e-4                  # head
    wd: float = 1e-2
    epochs: int = 15
    batch_size: int = 64
    evals_per_epoch: int = 4          # Endoscapes overfits by epoch 2-3: evaluate often, select afterwards
    smooth: int = 3                   # odd window of the centred moving average used for selection (1 = off)
    warmup_epochs: int = 0
    seed: int = 0
    eps: float | None = None          # B²-r positivity override (R0 fallback)
    clip: float = 1.0
    encoder_lr: float | None = None   # set => fine-tune the encoder on images
    amp: bool = True                  # bf16 autocast for the encoder forward (fine-tuning only)
    num_workers: int = 4

    @classmethod
    def finetune(cls, variant, encoder_lr, **kw):
        return cls(variant, encoder_lr=encoder_lr, **{"epochs": 5, "batch_size": 32, "warmup_epochs": 1, **kw})

    def __post_init__(self):
        if self.smooth < 1 or self.smooth % 2 == 0:
            raise ValueError("smooth must be an odd window >= 1")


@dataclass
class Split:
    x: object                       # [N, D] embeddings (np.ndarray) or an ImageSet
    ids: np.ndarray
    k: np.ndarray | None = None     # [N, K]; None for test
    n: np.ndarray | None = None
    v: np.ndarray | None = None     # [N, R, K] ordered votes (rater-aware variants, D8); None for test


def split_from_table(table: LabelTable, x, labelled=True) -> Split:
    """Split for one partition; labelled=False drops k, n and the votes (use for test)."""
    return Split(x, table.ids, table.k if labelled else None, table.n if labelled else None,
                 table.votes if labelled else None)


class Model(nn.Module):
    def __init__(self, head, encoder=None, amp=True):
        super().__init__()
        self.head, self.encoder, self.amp = head, encoder, amp

    def forward(self, x):
        if self.encoder is not None:
            with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=self.amp and x.device.type == "cuda"):
                x = self.encoder(x)
            x = x.float()
        return self.head(x)


# ---------------------------------------------------------------- batching
def _batches(split: Split, batch_size, device, shuffle, gen, num_workers):
    """Yields (x, row indices) on device. Embeddings are indexed directly; images go through a DataLoader."""
    if isinstance(split.x, ImageSet):
        loader = torch.utils.data.DataLoader(split.x, batch_size=batch_size, shuffle=shuffle, generator=gen,
                                             num_workers=num_workers, pin_memory=device.type == "cuda")
        for x, _, _, idx in loader:
            yield x.to(device, non_blocking=True), idx.numpy()
    else:
        N = len(split.ids)
        order = torch.randperm(N, generator=gen).numpy() if shuffle else np.arange(N)
        for s in range(0, N, batch_size):
            idx = order[s:s + batch_size]
            yield split.x[idx], idx


@torch.no_grad()
def predict(model, split: Split, device, batch_size=1024, num_workers=4) -> np.ndarray:
    """Raw head outputs [N, K, d] float32, model in eval mode."""
    model.eval()
    out = np.empty(0)
    for i, (x, idx) in enumerate(_batches(split, batch_size, device, False, None, num_workers)):
        o = model(x).float().cpu().numpy()
        if i == 0:
            out = np.empty((len(split.ids),) + o.shape[1:], np.float32)
        out[idx] = o
    return out


# ---------------------------------------------------------------- validation scoring
def score_split(name, out, k, n, scale, eps=None, prefix="val", v=None) -> dict:
    """mAP of m against the majority vote and vote NLL per vote with a given spread scale (DUP: its BCE).
    Rater-aware variants (D8): NLL of the ordered votes v with their learned rater biases."""
    if posthoc.spread_rule(name) == "dup":
        t = split_target(torch.as_tensor(k), torch.as_tensor(n)).numpy()
        p = np.clip(expit(out[..., 0].astype(np.float64)), 1e-12, 1 - 1e-12)
        return {f"{prefix}_dup_bce": float(-(t * np.log(p) + (1 - t) * np.log1p(-p)).mean())}
    m, S = posthoc.panel(name, out, scale, eps)
    if VARIANTS[name].loss.rater:
        nll = float((posthoc.ordered_nll(m, S, posthoc.rater_offsets(name, out), v) / n).mean())
    else:
        nll = float((posthoc.vote_nll(m, S, k, n) / n).mean())
    y = 2 * k > n
    aps = [average_precision_score(y[:, j], m[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return {f"{prefix}_map": float(np.mean(aps)) if aps else float("nan"), f"{prefix}_nll_per_vote": nll,
            f"{prefix}_median_S": np.median(S, 0).tolist()}


def score_val(name, out, k, n, eps=None, v=None) -> dict:
    """Validation selection quantities for one evaluation. Uses validation labels only."""
    scale = posthoc.fit_spread(name, out, k, n, eps, v)
    sc = score_split(name, out, k, n, scale, eps, v=v)
    if posthoc.spread_rule(name) == "dup":
        return {**sc, "sel_A": -sc["val_dup_bce"], "sel_B": sc["val_dup_bce"], "scale": None}
    return {**sc, "sel_A": sc["val_map"], "sel_B": sc["val_nll_per_vote"],
            "scale": None if scale is None else scale.tolist()}


def train_probe(train: "Split", n_val: int, seed: int) -> "Split":
    """Fixed subset of training rows (min(n_train, n_val)), drawn with its own generator (training RNG untouched)."""
    N = len(train.ids)
    idx = np.sort(np.random.default_rng(10_000 + seed).permutation(N)[: min(N, n_val)])
    x = train.x.subset(idx) if isinstance(train.x, ImageSet) else train.x[idx]
    return Split(x, train.ids[idx], train.k[idx], train.n[idx], None if train.v is None else train.v[idx])


def _lr_lambda(warmup_steps, total_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    return f


def _cpu_state(model):
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def _jsonable(d):
    return {k: v for k, v in d.items() if k not in ("sel_A", "sel_B")}


# ---------------------------------------------------------------- the trainer
def run(cfg: TrainConfig, train: Split, val: Split, test: Split, out_dir, encoder=None, device=None, verbose=True):
    """Train one variant, select A and B on validation, write predictions. Returns the summary."""
    if test.k is not None or test.n is not None:
        raise ValueError("test Split must not carry labels (use split_from_table(..., labelled=False))")
    finetune = cfg.encoder_lr is not None
    if finetune != (encoder is not None):
        raise ValueError("an encoder is given iff cfg.encoder_lr is set")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    if not finetune:   # embeddings live on the device for the whole run
        train, val, test = (Split(torch.as_tensor(np.asarray(s.x), dtype=torch.float32, device=device), s.ids, s.k, s.n,
                                  s.v) for s in (train, val, test))
    probe = train_probe(train, len(val.ids), cfg.seed)
    in_dim = encoder.num_features if finetune else train.x.shape[1]
    k_tr, n_tr = torch.as_tensor(train.k, dtype=torch.float32), torch.as_tensor(train.n, dtype=torch.float32)
    rater = VARIANTS[cfg.variant].loss.rater
    if rater and (train.v is None or val.v is None):
        raise ValueError(f"{cfg.variant} needs the ordered votes (Split.v) for training and validation")
    head, loss_fn = build(cfg.variant, in_dim, k_tr, n_tr, eps=cfg.eps, n_raters=train.v.shape[1] if rater else None)
    v_tr = torch.as_tensor(train.v, dtype=torch.float32, device=device) if rater else None
    model = Model(head, encoder, cfg.amp).to(device)
    loss_fn = loss_fn.to(device)
    k_tr, n_tr = k_tr.to(device), n_tr.to(device)

    if rater:
        # D8: the rater biases are logit-scale intercepts (order 1), not network weights: their own Adam step size
        # (Adam moves a parameter by ≈ lr per step) and no weight decay (decay would assume the raters identical)
        head_params = [p for n_, p in model.head.named_parameters() if n_ != "bias"]
        groups = [{"params": head_params, "lr": cfg.lr},
                  {"params": [model.head.bias], "lr": BETA_LR, "weight_decay": 0.0}]
    else:
        groups = [{"params": model.head.parameters(), "lr": cfg.lr}]
    if finetune:
        groups.append({"params": model.encoder.parameters(), "lr": cfg.encoder_lr})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.wd)
    steps_per_epoch = math.ceil(len(train.ids) / cfg.batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(cfg.warmup_epochs * steps_per_epoch,
                                                              cfg.epochs * steps_per_epoch))
    is_b2 = VARIANTS[cfg.variant].loss.kind == "b2"
    E = cfg.evals_per_epoch
    eval_after = {round(steps_per_epoch * (i + 1) / E) for i in range(E)}      # batch counts within an epoch
    best = {"A": (-np.inf, None, None, None), "B": (np.inf, None, None, None)}   # (smoothed crit, eval, state, scores)
    hist, recent, h = [], deque(maxlen=cfg.smooth // 2 + 1), cfg.smooth // 2  # recent: (eval, state, scores)
    log_path = out_dir / "log.jsonl"
    log_path.write_text("")
    nonfinite_total, t0 = 0, time.time()

    def smoothed(j, c):
        """Centred moving average of the selection criterion over evaluations j-h..j+h (truncated at the ends)."""
        v = np.array([r[c] for r in hist[max(0, j - h): j + h + 1]], dtype=float)
        return np.nan if np.isnan(v).all() else float(np.nanmean(v))

    def consider(j, state, sc):
        for c, better in (("A", lambda a, b: a > b), ("B", lambda a, b: a < b)):
            v = smoothed(j, c)
            if better(v, best[c][0]):
                best[c] = (v, j, state, sc)

    def reset_counters():
        return 0.0, 0, 0, torch.zeros(k_tr.shape[1], device=device), torch.zeros(k_tr.shape[1], device=device)

    loss_sum, n_steps, nonfinite, stuck_num, stuck_den = reset_counters()
    for epoch in range(cfg.epochs):
        model.train()
        for b, (x, idx) in enumerate(_batches(train, cfg.batch_size, device, True, gen, cfg.num_workers), start=1):
            idx_t = torch.as_tensor(idx, device=device)
            kb, nb = k_tr[idx_t], n_tr[idx_t]
            out = model(x)
            loss = loss_fn(out, kb, nb, None if v_tr is None else v_tr[idx_t])
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                nonfinite += 1
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
                opt.step()
                loss_sum, n_steps = loss_sum + loss.item(), n_steps + 1
                if is_b2 and VARIANTS[cfg.variant].loss.param == "alpha":     # alpha_1 logit; not defined for 'ms'
                    pos = kb >= 1
                    stuck_num += ((out.detach()[..., 1] < STUCK_LOGIT) & pos).sum(0)
                    stuck_den += pos.sum(0)
            sched.step()
            if b not in eval_after:
                continue

            # ---- evaluation (validation only)
            t = len(hist)
            val_out = predict(model, val, device, num_workers=cfg.num_workers)
            sc = score_val(cfg.variant, val_out, val.k, val.n, cfg.eps, val.v)
            scale = None if sc["scale"] is None else np.asarray(sc["scale"])
            tr = score_split(cfg.variant, predict(model, probe, device, num_workers=cfg.num_workers), probe.k, probe.n,
                             scale, cfg.eps, prefix="train", v=probe.v)
            rec = {"eval": t, "epoch": round(epoch + b / steps_per_epoch, 3), "train_loss": loss_sum / max(n_steps, 1),
                   "nonfinite_steps": nonfinite, "lr": sched.get_last_lr()[0], **_jsonable(sc), **tr}
            if is_b2:
                rec["train_stuck_share"] = (stuck_num / stuck_den.clamp(min=1)).tolist()
            nonfinite_total += nonfinite
            hist.append({"A": sc["sel_A"], "B": sc["sel_B"]})
            recent.append((t, _cpu_state(model), sc))
            if t >= h:                     # evaluation t completes the window centred on t - h
                consider(*recent[0])
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            if verbose:   # validation only
                print(f"[{cfg.variant}] ep {rec['epoch']:6.2f} loss {rec['train_loss']:.4f} "
                      + " ".join(f"{k} {v:.4f}" for k, v in rec.items()
                                 if k in ("val_map", "val_nll_per_vote", "val_dup_bce")), flush=True)
            loss_sum, n_steps, nonfinite, stuck_num, stuck_den = reset_counters()
            model.train()
    for j, state, sc in list(recent)[1:] if len(hist) > h else list(recent):   # windows at the end of the run
        consider(j, state, sc)

    summary = {"config": asdict(cfg), "finetune": finetune, "n_train": len(train.ids), "n_val": len(val.ids),
               "nonfinite_steps": nonfinite_total, "seconds": round(time.time() - t0, 1)}
    if best["A"][2] is None or best["B"][2] is None:
        raise RuntimeError("no finite validation criterion in any evaluation")
    for c in ("A", "B"):
        crit, j, state, sc = best[c]
        model.load_state_dict(state)
        scale = None if sc["scale"] is None else np.asarray(sc["scale"])
        for name, split in (("val", val), ("test", test)):
            o = predict(model, split, device, num_workers=cfg.num_workers)
            np.savez(out_dir / f"{name}_{c}.npz", ids=split.ids, out=o,
                     scale=np.full(o.shape[1], np.nan) if scale is None else scale)
        torch.save(model.head.state_dict(), out_dir / f"head_{c}.pt")
        summary[c] = {"eval": j, "epoch": json.loads(log_path.read_text().splitlines()[j])["epoch"],
                      "smoothed_criterion": crit, **_jsonable(sc)}
    # model is now at B: memorisation diagnostic (median S on train / on val), B² variants only
    if is_b2:
        scale = np.asarray(best["B"][3]["scale"])
        S_tr = posthoc.panel(cfg.variant, predict(model, train, device, num_workers=cfg.num_workers), scale, cfg.eps)[1]
        S_va = posthoc.panel(cfg.variant, np.load(out_dir / "val_B.npz")["out"], scale, cfg.eps)[1]
        summary["B"]["train_val_S_ratio"] = (np.median(S_tr, 0) / np.median(S_va, 0)).tolist()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
