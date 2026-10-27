"""Balanced random re-split of the Endoscapes2023 videos (whole videos; 121 / 40 / 40).

  python -m b2j.resplit_endoscapes [--draws 100000] [--seed 1] [--top 5] [--write <rank>]

Per split, six numbers: the majority-positive rate and the unanimous rate of each CVS criterion (C1-C3), over its
frames. A split's score is the largest gap (max - min over train/val/test) among the six; lower is better.
Frame shares must stay within: train 60-70%, val >= 12.5%, test >= 12.5%, |val - test| <= 5 points. Draws random splits from a fixed seed and
prints the best few; --write saves the chosen one to b2j/splits/endoscapes_resplit.csv (video, split).
The committed split is `--draws 100000 --seed 1 --write 1` (draw 62405; largest gap 2.25 points, official 15.6).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from b2j.data import DATA_ROOT, ENDO_SPLIT, endoscapes_labels

SPLITS = ("train", "val", "test")
SIZES = (121, 40, 40)
OUT = Path(__file__).resolve().parent / "splits" / "endoscapes_resplit.csv"


def video_sums(t):
    """Per video: frames, majority-positive frames per criterion, unanimous frames per criterion."""
    vids = np.unique(t.cluster)
    pos, una = t.k > t.n / 2, (t.k == 0) | (t.k == t.n)
    rows = {v: [0, *[0] * 6] for v in vids}
    for v, p, u in zip(t.cluster, pos, una):
        r = rows[v]
        r[0] += 1
        for c in range(3):
            r[1 + c] += p[c]
            r[4 + c] += u[c]
    return vids, np.array([rows[v] for v in vids], dtype=float)        # [V, 7]


def stats(sums, assign):
    """assign: [V] in {0, 1, 2}. Returns frame shares [3] and the six rates [3, 6]."""
    tot = np.stack([sums[assign == s].sum(0) for s in range(3)])        # [3, 7]
    return tot[:, 0] / tot[:, 0].sum(), tot[:, 1:] / tot[:, :1]


def score(rates):
    return (rates.max(0) - rates.min(0)).max()


def feasible(shares):
    return (0.60 <= shares[0] <= 0.70 and shares[1] >= 0.125 and shares[2] >= 0.125
            and abs(shares[1] - shares[2]) <= 0.05)


def show(name, sums, assign):
    shares, rates = stats(sums, assign)
    print(f"{name}: score {100 * score(rates):.2f} pp (largest gap)")
    print(f"  {'split':5s} {'videos':>6s} {'frames':>7s} {'share':>6s}  majority-positive C1/C2/C3   unanimous C1/C2/C3")
    for s in range(3):
        f = int(sums[assign == s, 0].sum())
        print(f"  {SPLITS[s]:5s} {(assign == s).sum():6d} {f:7d} {100 * shares[s]:5.1f}%  "
              + " / ".join(f"{100 * x:4.1f}" for x in rates[s, :3]) + "      "
              + " / ".join(f"{100 * x:4.1f}" for x in rates[s, 3:]))
    gaps = rates.max(0) - rates.min(0)
    print("  gaps (pp): majority-positive " + " / ".join(f"{100 * g:.1f}" for g in gaps[:3])
          + ", unanimous " + " / ".join(f"{100 * g:.1f}" for g in gaps[3:]))


def main(draws=100000, seed=1, top=5, write=None):
    t = endoscapes_labels(DATA_ROOT / "endoscapes", split="official")
    vids, sums = video_sums(t)
    official = np.array([next(i for i, (lo, hi) in enumerate(ENDO_SPLIT.values()) if lo <= v <= hi) for v in vids])
    show("official split", sums, official)

    rng = np.random.default_rng(seed)
    base = np.repeat(np.arange(3), SIZES)
    found, n_ok = [], 0
    for d in range(draws):
        assign = rng.permutation(base)
        shares, rates = stats(sums, assign)
        if feasible(shares):
            n_ok += 1
            found.append((score(rates), d, assign))
    found.sort(key=lambda x: (x[0], x[1]))
    print(f"\n{draws} draws (seed {seed}): {n_ok} satisfy the frame shares; best {top}:")
    for rank, (_, d, assign) in enumerate(found[:top], 1):
        print()
        show(f"candidate {rank} (draw {d})", sums, assign)
    if write is not None:
        _, d, assign = found[int(write) - 1]
        OUT.parent.mkdir(exist_ok=True)
        pd.DataFrame({"video": vids.astype(int), "split": [SPLITS[a] for a in assign]}).to_csv(OUT, index=False)
        print(f"\nwrote candidate {write} (draw {d}) to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--write", type=int, default=None)
    main(**vars(ap.parse_args()))
