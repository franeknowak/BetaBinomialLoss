"""Dataset loading: label tables (exact per-sample k, n) and image datasets.

A LabelTable holds one row per image: k, n as float arrays [N, K], plus a cluster id
(bootstrap resampling unit), the split and the image path; where the rater of each vote is known (Endoscapes,
AMi-Br: the same 3 raters for every image) also the votes themselves [N, R, K].
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
    counts: np.ndarray | None = None   # [N, C] raw category counts (single-label datasets: CIFAR-10H, FER+)
    votes: np.ndarray | None = None    # [N, R, K] each rater's 0/1 vote (NaN = rater did not vote); k = Σ_r votes

    def subset(self, mask):
        return LabelTable(*(None if v is None else v[mask] for v in
                            (self.ids, self.k, self.n, self.cluster, self.split, self.paths, self.counts, self.votes)))

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
    return LabelTable(ids, k, np.full_like(k, 3.0), d["vid"].values, split.astype(str), paths,
                      votes=votes.astype(np.float64))


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


# ---------------------------------------------------------------- AMi-Br
AMIBR_RATERS = ("expert1_atypical", "expert2_atypical", "expert3_atypical")


def slide_split(slides, strata, fracs=(0.6, 0.2, 0.2), seed=0):
    """Assign whole slides to train/val/test, stratified by source dataset (fixed seed). Returns [N] split per row."""
    rng = np.random.default_rng(seed)
    out = np.full(len(slides), "", dtype=object)
    for s in np.unique(strata):
        sl = rng.permutation(np.unique(slides[strata == s]))
        cut = np.round(np.cumsum(fracs) / sum(fracs) * len(sl)).astype(int)
        for name, part in zip(("train", "val", "test"), np.split(sl, cut[:-1])):
            out[np.isin(slides, part) & (strata == s)] = name
    return out.astype(str)


def amibr_labels(root, seed=0) -> LabelTable:
    """AMi-Br (Bertram et al., BVM 2025): k = atypical votes of 3 pathologists per mitotic figure.

    Patches: patches/<atypical|normal>/<dataset>_<uid>.png, foldered by the majority vote. Bootstrap unit and
    split unit = slide (patient); slide names are only unique within a source dataset, so the key is both.
    No official split (the authors' patient-level split is unseeded): 60/20/20 of the slides per source.
    """
    root = Path(root)
    d = pd.read_csv(root / "AMI-BR.csv", index_col=0)
    votes = d[list(AMIBR_RATERS)].astype(bool).values[:, :, None].astype(np.float64)   # [N, 3 experts, 1]
    k = votes.sum(1)
    if ((k >= 2)[:, 0] != d["majority_atypical"].astype(bool).values).any():
        raise ValueError("expert votes disagree with majority_atypical")
    slide = (d["dataset"] + "/" + d["slide"].astype(str)).values.astype(str)
    split = slide_split(slide, d["dataset"].values.astype(str), seed=seed)
    ids = (d["dataset"] + "_" + d["uid"].astype(str)).values.astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate (dataset, uid) ids")
    folder = np.where(d["majority_atypical"].astype(bool).values, "atypical", "normal")
    paths = np.array([str(root / "patches" / f / f"{i}.png") for f, i in zip(folder, ids)])
    return LabelTable(ids, k, np.full_like(k, 3.0), slide, split, paths, votes=votes)


# ---------------------------------------------------------------- FER+
FERPLUS_EMOTIONS = ("neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt")
FERPLUS_SPLIT = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}   # official FER+ usage


def ferplus_labels(root) -> LabelTable:
    """FER+ (Barsoum et al., ICMI 2016): 10 crowd votes per FER2013 face over 8 emotions + "unknown" + "not a face".

    Kept: rows with a unique image name (FER+ removed 173; 4 names occur twice, and a folder of images keeps only
    one of each pair, so all 8 rows are dropped) whose plurality vote is an emotion (drops faces the crowd
    called unknown / not a face). One-vs-rest heads over the 8 emotions; n = the image's emotion votes (unknown / NF
    votes are not part of the panel), so n varies (mostly 6-10). Cluster = image.
    Images: located via zip_index.csv (b2j.prepare_ferplus; faces without an unambiguous original in the zip are
    dropped): read inside the zip while it is present, else from the unpacked copy <root>/<member> (cluster, unpacked
    2026-10-27); without the index (Mac), images/<name>.png.
    """
    root = Path(root)
    d = pd.read_csv(root / "fer2013new.csv")
    votes = d[list(FERPLUS_EMOTIONS) + ["unknown", "NF"]].values.astype(np.float64)
    name = d["Image name"]
    keep = name.notna().values & ~name.duplicated(keep=False).values & (votes.argmax(1) < len(FERPLUS_EMOTIONS))
    d, counts = d[keep].reset_index(drop=True), votes[keep, :len(FERPLUS_EMOTIONS)]
    if (counts.sum(1) == 0).any():
        raise ValueError("kept image without emotion votes")
    ids = d["Image name"].str.replace(".png", "", regex=False).values.astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate image names")
    n = np.repeat(counts.sum(1, keepdims=True), counts.shape[1], 1)
    split = d["Usage"].map(FERPLUS_SPLIT).values.astype(str)
    paths = np.array([str(root / "images" / f"{i}.png") for i in ids])
    t = LabelTable(ids, counts.copy(), n, np.arange(len(ids)), split, paths, counts=counts)
    if (root / "zip_index.csv").exists():
        idx = pd.read_csv(root / "zip_index.csv", dtype=str).set_index("id")
        t = t.subset(np.isin(t.ids, idx.index.values))
        t.paths = np.array([f"{root / idx.at[i, 'archive']}{ZIP_SEP}{idx.at[i, 'member']}"
                            if (root / idx.at[i, "archive"]).exists() else str(root / idx.at[i, "member"])
                            for i in t.ids])
    return t


# ---------------------------------------------------------------- images
INPUT_SIZE = {"endoscapes": 384, "cifar10h": 224, "mhist": 224, "amibr": 224, "ferplus": 224}
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def transform(dataset: str):
    """No augmentation. Endoscapes: centre-crop 480 (as MICCAI) then 384². CIFAR-10H: 32 → 224 bicubic. MHIST: native 224.
    AMi-Br: 128 → 224 bicubic. FER+: grayscale (48 px, or 112 px upscaled in the Kaggle copy) → RGB → 224 bicubic."""
    from torchvision import transforms as T
    steps = [T.CenterCrop(480)] if dataset == "endoscapes" else []
    steps += [T.Resize((INPUT_SIZE[dataset],) * 2, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
              T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)


ZIP_SEP = "::"      # an image inside an archive: "<archive>::<member>"; read in place
_ZIPS: dict = {}


def _zip(archive):
    import os
    import zipfile
    key = (os.getpid(), archive)                     # one handle per process (DataLoader workers fork)
    if key not in _ZIPS:
        _ZIPS[key] = zipfile.ZipFile(archive)
    return _ZIPS[key]


def open_image(path):
    from PIL import Image
    if ZIP_SEP not in path:
        return Image.open(path)
    import io
    archive, member = path.split(ZIP_SEP, 1)
    return Image.open(io.BytesIO(_zip(archive).read(member)))


def image_exists(path) -> bool:
    if ZIP_SEP not in path:
        return Path(path).exists()
    archive, member = path.split(ZIP_SEP, 1)
    if not Path(archive).exists():
        return False
    z = _zip(archive)
    if not hasattr(z, "_b2j_names"):
        z._b2j_names = set(z.namelist())
    return member in z._b2j_names


class ImageSet:
    """torch-style dataset over a LabelTable: returns (image, k, n, index)."""

    def __init__(self, table: LabelTable, dataset: str):
        self.table, self.tf = table, transform(dataset)

    def __len__(self):
        return len(self.table.ids)

    def subset(self, idx):
        out = ImageSet.__new__(ImageSet)
        out.table, out.tf = self.table.subset(idx), self.tf
        return out

    def __getitem__(self, i):
        import torch
        img = self.tf(open_image(self.table.paths[i]).convert("RGB"))
        return img, torch.from_numpy(self.table.k[i]).float(), torch.from_numpy(self.table.n[i]).float(), i


def load_table(dataset: str, root=None) -> LabelTable:
    """Full label table of a dataset from the shared layout DATA_ROOT/<dataset> (same on Mac and cluster)."""
    root = Path(root) if root is not None else DATA_ROOT / dataset
    if dataset == "endoscapes":
        return endoscapes_labels(root)
    if dataset == "cifar10h":
        return cifar10h_labels(np.load(root / "cifar10h-counts.npy"), np.load(root / "test_labels.npy"), root / "images")
    if dataset == "mhist":
        return mhist_labels(root / "annotations.csv", root / "images")
    if dataset == "amibr":
        return amibr_labels(root)
    if dataset == "ferplus":
        return ferplus_labels(root)
    raise ValueError(f"unknown dataset {dataset}")
