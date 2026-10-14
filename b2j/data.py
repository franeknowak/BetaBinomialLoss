"""Module b: label tables (exact per-sample k, n), vote subsampling, ambiguous subset, image datasets.

A LabelTable holds one row per image: k, n as float arrays [N, K], plus a cluster id
(bootstrap resampling unit), the split and the image path. See journal_plan/specs/b_loaders.md.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ENDO_SPLIT = {"train": (1, 120), "val": (121, 161), "test": (162, 201)}   # official video ranges
DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets"   # same on Mac and cluster: <repo>/../datasets/<name>


@dataclass
class LabelTable:
    ids: np.ndarray        # [N] str
    k: np.ndarray          # [N, K] float
    n: np.ndarray          # [N, K] float
    cluster: np.ndarray    # [N] bootstrap unit (video / image)
    split: np.ndarray      # [N] 'train' | 'val' | 'test'
    paths: np.ndarray      # [N] image path
    counts: np.ndarray | None = None   # [N, C] raw category counts (CIFAR-10H) for joint subsampling

    def subset(self, mask):
        return LabelTable(*(None if v is None else v[mask] for v in
                            (self.ids, self.k, self.n, self.cluster, self.split, self.paths, self.counts)))

    def take(self, split):
        return self.subset(self.split == split)


# ---------------------------------------------------------------- Endoscapes2023 (CVS201)
def endoscapes_labels(root) -> LabelTable:
    """Exact k per criterion = sum of the 3 raters' binary votes, from all_metadata.csv (keyframes only)."""
    root = Path(root)
    d = pd.read_csv(root / "all_metadata.csv")
    d = d[d["is_ds_keyframe"]].reset_index(drop=True)
    votes = np.stack([np.array([json.loads(x) for x in d[f"cvs_annotator_{i}"]]) for i in (1, 2, 3)], 1)
    k = votes.sum(1).astype(np.float64)                                       # [N, 3]
    if np.abs(k / 3 - d[["C1", "C2", "C3"]].values).max() > 1e-9:
        raise ValueError("rater votes disagree with the averaged C1-C3 columns")
    split = np.full(len(d), "", dtype=object)
    for s, (lo, hi) in ENDO_SPLIT.items():
        split[(d["vid"] >= lo) & (d["vid"] <= hi)] = s
    ids = (d["vid"].astype(str) + "_" + d["frame"].astype(str)).values.astype(str)   # str, not object: np.load needs no pickle
    paths = np.array([str(root / s / f"{i}.jpg") for s, i in zip(split, ids)])
    return LabelTable(ids, k, np.full_like(k, 3.0), d["vid"].values, split.astype(str), paths)


# ---------------------------------------------------------------- CIFAR-10H
def stratified_split(labels, sizes=(6000, 2000, 2000), seed=0):
    """Split indices stratified by class into train/val/test of the given sizes (fixed seed)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    total = sum(sizes)
    out = np.full(len(labels), "", dtype=object)
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        cut = np.round(np.cumsum(sizes) / total * len(idx)).astype(int)
        for name, part in zip(("train", "val", "test"), np.split(idx, cut[:-1])):
            out[part] = name
    return out.astype(str)


def cifar10h_labels(counts, true_labels, image_dir, seed=0) -> LabelTable:
    """One-vs-rest heads: k_c = votes for class c, n = the image's total votes. counts: [10000, 10]."""
    counts = np.asarray(counts, dtype=np.float64)
    n = np.repeat(counts.sum(1, keepdims=True), counts.shape[1], 1)
    ids = np.array([f"{i:05d}" for i in range(len(counts))])
    paths = np.array([str(Path(image_dir) / f"{i}.png") for i in ids])
    return LabelTable(ids, counts.copy(), n, np.arange(len(counts)),
                      stratified_split(true_labels, seed=seed), paths, counts=counts)


# ---------------------------------------------------------------- MHIST
def mhist_labels(annotations_csv, image_dir, val_frac=0.2, seed=0) -> LabelTable:
    """k = SSA votes out of 7. The vote column name is UNVERIFIED: accepted names below, else fail loudly."""
    d = pd.read_csv(annotations_csv)
    col = next((c for c in d.columns if "ssa" in c.lower() and ("number" in c.lower() or "vote" in c.lower())
                and "majority" not in c.lower()), None)
    if col is None:
        raise ValueError(f"no SSA vote-count column in {list(d.columns)}")
    k = d[[col]].values.astype(np.float64)
    if (k > 7).any() or ((k > 3.5) != (d["Majority Vote Label"].values[:, None] == "SSA")).any():
        raise ValueError("SSA vote counts inconsistent with the majority label")
    split = np.where(d["Partition"].str.lower().values == "test", "test", "train").astype(object)
    rng = np.random.default_rng(seed)
    tr = np.flatnonzero(split == "train")
    split[rng.choice(tr, int(round(val_frac * len(tr))), replace=False)] = "val"
    ids = d["Image Name"].values.astype(str)
    paths = np.array([str(Path(image_dir) / i) for i in ids])
    return LabelTable(ids, k, np.full_like(k, 7.0), np.arange(len(d)), split.astype(str), paths)


# ---------------------------------------------------------------- subsampling and subsets
def subsample(table: LabelTable, n_sub: int, seed: int) -> LabelTable:
    """Draw n_sub raters per image without replacement from its votes (hypergeometric), fixed per seed.

    Multi-class tables (CIFAR-10H) draw jointly from the category counts so the K one-vs-rest
    heads stay consistent; binary tables draw per head from (k, n - k).
    """
    rng = np.random.default_rng(seed)
    if n_sub > table.n.min():
        raise ValueError(f"n_sub={n_sub} exceeds the smallest panel ({table.n.min():.0f})")
    if table.counts is not None:
        counts = np.stack([rng.multivariate_hypergeometric(c.astype(np.int64), n_sub) for c in table.counts])
        k = counts.astype(np.float64)
    else:
        counts = None
        k = rng.hypergeometric(table.k.astype(np.int64), (table.n - table.k).astype(np.int64), n_sub).astype(np.float64)
    return LabelTable(table.ids, k, np.full_like(k, float(n_sub)), table.cluster, table.split, table.paths, counts)


def ambiguous_mask(table: LabelTable, threshold=0.9) -> np.ndarray:
    """Full-panel plurality share < threshold. Multi-class: max category share; binary heads: any head."""
    if table.counts is not None:
        return table.counts.max(1) / table.counts.sum(1) < threshold
    share = np.maximum(table.k, table.n - table.k) / table.n
    return (share < threshold).any(1)


def split_frame_counts(table: LabelTable) -> np.ndarray:
    """Per head: number of frames with 0 < k < n (the U2 feasibility count)."""
    return ((table.k > 0) & (table.k < table.n)).sum(0)


# ---------------------------------------------------------------- images
INPUT_SIZE = {"endoscapes": 384, "cifar10h": 224, "mhist": 224}
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def transform(dataset: str):
    """No augmentation. Endoscapes: centre-crop 480 (as MICCAI) then 384². CIFAR-10H: 32 → 224 bicubic. MHIST: native 224."""
    from torchvision import transforms as T
    steps = [T.CenterCrop(480)] if dataset == "endoscapes" else []
    steps += [T.Resize((INPUT_SIZE[dataset],) * 2, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
              T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)


class ImageSet:
    """torch-style dataset over a LabelTable: returns (image, k, n, index)."""

    def __init__(self, table: LabelTable, dataset: str):
        self.table, self.tf = table, transform(dataset)

    def __len__(self):
        return len(self.table.ids)

    def __getitem__(self, i):
        from PIL import Image
        import torch
        img = self.tf(Image.open(self.table.paths[i]).convert("RGB"))
        return img, torch.from_numpy(self.table.k[i]).float(), torch.from_numpy(self.table.n[i]).float(), i


def endoscapes_rater_table(root, splits=("train", "val")) -> pd.DataFrame:
    """Per-rater positive rate and pairwise Cohen's kappa per criterion (plan §6, fixed-panel limitation)."""
    from sklearn.metrics import cohen_kappa_score
    d = pd.read_csv(Path(root) / "all_metadata.csv")
    d = d[d["is_ds_keyframe"]]
    keep = np.zeros(len(d), bool)
    for s in splits:
        lo, hi = ENDO_SPLIT[s]
        keep |= ((d["vid"] >= lo) & (d["vid"] <= hi)).values
    v = np.stack([np.array([json.loads(x) for x in d[f"cvs_annotator_{i}"]]) for i in (1, 2, 3)], 1)[keep]
    rows = []
    for c in range(3):
        row = {"criterion": f"C{c + 1}"}
        row.update({f"pos_rate_r{r + 1}": v[:, r, c].mean() for r in range(3)})
        row.update({f"kappa_r{a + 1}r{b + 1}": cohen_kappa_score(v[:, a, c], v[:, b, c]) for a, b in ((0, 1), (0, 2), (1, 2))})
        rows.append(row)
    return pd.DataFrame(rows)


def load_table(dataset: str, root=None) -> LabelTable:
    """Full label table of a dataset from the shared layout DATA_ROOT/<dataset> (same on Mac and cluster)."""
    root = Path(root) if root is not None else DATA_ROOT / dataset
    if dataset == "endoscapes":
        return endoscapes_labels(root)
    if dataset == "cifar10h":
        return cifar10h_labels(np.load(root / "cifar10h-counts.npy"), np.load(root / "test_labels.npy"), root / "images")
    if dataset == "mhist":
        return mhist_labels(root / "annotations.csv", root / "images")
    raise ValueError(f"unknown dataset {dataset}")
