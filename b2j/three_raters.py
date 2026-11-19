"""Three raters per image: train and validate on 3 raters drawn per image, test against the full panel.

Datasets with more than 3 raters per image: MHIST (7), FER+ (4-10), CIFAR-10H (47-63); all images kept.
For every image, 3 of its n votes are drawn without replacement, once (one fixed draw per image, DRAW_SEED; the same
draw for every loss, setting and seed). The raters of these datasets are not identified per vote, so a draw takes 3
of the image's votes at random (hypergeometric over the vote counts: the votes of the K labels, plus for a single
binary label the n - k "no" votes). Train and val use the drawn counts (k out of n = 3); test keeps the full panel.
Tuning (grid + extension, seed 0), selection and final runs (seeds 1-5) exactly as in phase 1, on the drawn
train / val counts only.

  python -m b2j.three_raters <dataset> [--seeds 1 2 3 4 5]

Writes results/three_raters/<dataset>/draw.npz (ids, split, drawn k, full k and n), results/three_raters/tuning/
<dataset>/... and results/three_raters/final/<dataset>/<loss>/seed<s>/ (as the phase-1 final runs).
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from b2j.experiment import FINAL_SEEDS, RESULTS, _device, final, tune_and_extend
from b2j.train import load_embeddings

DATASETS = ("mhist", "ferplus", "cifar10h")
DRAWN, DRAW_SEED = 3, 0
ROOT = RESULTS / "three_raters"
SPLITS = ("train", "val", "test")


def draw_votes(k, n, rng, r=DRAWN):
    """k, n [N, K] (vote counts) -> k drawn [N, K]: r of each image's votes, without replacement.

    Single-label datasets (votes of the K labels add up to n): one draw over the K labels. A single binary label
    (K = 1): over "yes" (k) and "no" (n - k)."""
    k, n = np.rint(np.asarray(k)).astype(np.int64), np.rint(np.asarray(n)).astype(np.int64)
    if k.shape[1] == 1:
        colors = np.concatenate([k, n - k], 1)
    elif (k.sum(1) == n[:, 0]).all() and (n == n[:, :1]).all():
        colors = k
    else:
        raise ValueError("votes are neither one binary label nor one label per vote")
    if (colors.sum(1) < r).any():
        raise ValueError(f"an image has fewer than {r} votes")
    out = np.stack([rng.multivariate_hypergeometric(c, r) for c in colors])
    return out[:, :k.shape[1]].astype(np.float64)


def three_rater_data(data, seed=DRAW_SEED, r=DRAWN):
    """data as from train.load_embeddings -> (train / val with the drawn counts, test unchanged, draw record)."""
    rng = np.random.default_rng(seed)
    out, rec = {}, {"ids": [], "split": [], "k_drawn": [], "k_full": [], "n_full": []}
    for s in SPLITS:                                     # fixed order -> one fixed draw per image
        X, k, n, ids = data[s]
        kd = draw_votes(k.cpu().numpy(), n.cpu().numpy(), rng, r)
        rec["ids"].append(np.asarray(ids).astype(str))
        rec["split"].append(np.full(len(ids), s))
        rec["k_drawn"].append(kd)
        rec["k_full"].append(k.cpu().numpy())
        rec["n_full"].append(n.cpu().numpy())
        if s == "test":
            out[s] = data[s]
        else:
            kt = torch.as_tensor(kd, dtype=k.dtype, device=k.device)
            out[s] = (X, kt, torch.full_like(kt, float(r)), ids)
    return out, {a: np.concatenate(v) for a, v in rec.items()}


def run(dataset, seeds=FINAL_SEEDS, root=ROOT, epochs=25):
    dev = _device()
    full, labels = load_embeddings(dataset, device=dev)
    data, rec = three_rater_data(full)
    d = Path(root) / dataset
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "draw.npz", labels=np.array(labels), draw_seed=DRAW_SEED, **rec)
    for s in ("train", "val"):
        m = rec["split"] == s
        kd, kf, nf = rec["k_drawn"][m], rec["k_full"][m], rec["n_full"][m]
        print(f"{dataset} {s}: {m.sum()} images; drawn majority = full-panel majority on "
              f"{((kd > 1.5) == (kf > nf / 2)).mean():.3f} of (image, label)", flush=True)
    tune_and_extend(dataset, root=root, epochs=epochs, data=(data, labels))
    final(dataset, seeds=seeds, root=root, epochs=epochs, data=(data, labels))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=DATASETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(FINAL_SEEDS))
    a = ap.parse_args()
    run(a.dataset, seeds=tuple(a.seeds))
