"""Small sample pack of every dataset for the overview notebook (notebooks/datasets_overview.ipynb).

Per dataset writes to <out>/<dataset>/:
  table.npz     the full label table as b2j.data.load_table builds it (ids, k, n, split, cluster; votes where known)
  examples.csv  the rows that have a thumbnail: row, id, split, label, bin, file
  img/<row>.png|jpg  thumbnails (Endoscapes: JPEG, 384 px wide; the others PNG at native size)

Examples: per label and vote bin up to PER_BIN random images (all splits, fixed seed), plus RANDOM random images.
Vote bins: exact k (0..n) where every image has the same small panel (Endoscapes, MHIST, AMi-Br); otherwise the
share k/n in fifths, without k = 0 (CIFAR-10H, FER+: a class nobody chose says nothing about the image).

    python -m b2j.dataset_examples --out results/dataset_examples
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from b2j import data as D

DATASETS = ("endoscapes", "cifar10h", "mhist", "amibr", "ferplus")
LABELS = {"endoscapes": ("C1", "C2", "C3"),
          "cifar10h": ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"),
          "mhist": ("SSA",), "amibr": ("atypical",), "ferplus": D.FERPLUS_EMOTIONS}
PER_BIN, RANDOM = 12, 48
SHARE_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def vote_bins(k, n):
    """[N, K] bin label per image and label: 'k/n' for a fixed small panel, else a share interval ('' = skip)."""
    if np.unique(n).size == 1 and n.flat[0] <= 10:
        return np.array([[f"{int(a)}/{int(b)}" for a, b in zip(ra, rb)] for ra, rb in zip(k, n)], dtype=object)
    share = k / n
    out = np.full(k.shape, "", dtype=object)
    for lo, hi in zip(SHARE_EDGES[:-1], SHARE_EDGES[1:]):
        out[(share > lo) & (share <= hi)] = f"{lo:.0%}-{hi:.0%}"
    return out


def pick(table, labels, seed=0):
    """Rows to export: (row, label, bin); label = '' for the random overview sample."""
    rng = np.random.default_rng(seed)
    bins = vote_bins(table.k, table.n)
    rows = []
    for j, lab in enumerate(labels):
        for b in sorted(set(bins[:, j]) - {""}):
            idx = np.flatnonzero(bins[:, j] == b)
            for r in rng.choice(idx, min(PER_BIN, idx.size), replace=False):
                rows.append((int(r), lab, b))
    for r in rng.choice(len(table.ids), RANDOM, replace=False):
        rows.append((int(r), "", ""))
    return rows


def thumbnail(path, dataset):
    img = D.open_image(path).convert("RGB")
    if dataset == "endoscapes":
        img = img.resize((384, round(384 * img.height / img.width)))
        return img, "jpg"
    return img, "png"


def export(dataset, out):
    t = D.load_table(dataset)
    d = Path(out) / dataset
    (d / "img").mkdir(parents=True, exist_ok=True)
    arrays = dict(ids=t.ids, k=t.k, n=t.n, split=t.split, cluster=t.cluster.astype(str),
                  labels=np.array(LABELS[dataset]))
    if t.votes is not None:
        arrays["votes"] = t.votes
    np.savez_compressed(d / "table.npz", **arrays)
    rows, files = pick(t, LABELS[dataset]), {}
    for r, _, _ in rows:
        if r not in files:
            img, ext = thumbnail(t.paths[r], dataset)
            files[r] = f"img/{r}.{ext}"
            img.save(d / files[r], quality=90) if ext == "jpg" else img.save(d / files[r])
    pd.DataFrame([dict(row=r, id=t.ids[r], split=t.split[r], label=lab, bin=b, file=files[r])
                  for r, lab, b in rows]).to_csv(d / "examples.csv", index=False)
    print(f"{dataset}: {len(t.ids)} images in table, {len(files)} thumbnails, {len(rows)} example rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/dataset_examples")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    a = ap.parse_args()
    for ds in a.datasets:
        export(ds, a.out)


if __name__ == "__main__":
    main()
