"""Unit tests for module b (loaders). Run: python -m pytest tests/test_module_b.py -q

The Endoscapes test runs on the real metadata if ../datasets/endoscapes/all_metadata.csv exists.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from b2j import data as D

ENDO = D.DATA_ROOT / "endoscapes"


@pytest.mark.skipif(not (ENDO / "all_metadata.csv").exists(), reason="Endoscapes metadata not present")
def test_endoscapes_real_counts():
    t = D.endoscapes_labels(ENDO)
    assert {s: (t.split == s).sum() for s in ("train", "val", "test")} == {"train": 6960, "val": 2331, "test": 1799}
    assert set(np.unique(t.k)) <= {0, 1, 2, 3} and (t.n == 3).all()
    tr = t.take("train")
    np.testing.assert_allclose((2 * tr.k > 3).mean(0), [0.1563, 0.1121, 0.1789], atol=5e-4)   # MICCAI PI values
    assert len(np.intersect1d(t.take("train").cluster, t.take("test").cluster)) == 0


def _fake_cifar(N=1000, seed=0):
    rng = np.random.default_rng(seed)
    true = np.repeat(np.arange(10), N // 10)
    counts = np.stack([rng.multinomial(rng.integers(47, 64), np.where(np.arange(10) == c, 0.8, 0.2 / 9)) for c in true])
    return counts, true


def test_stratified_split_sizes_and_balance():
    _, true = _fake_cifar(10000)
    s = D.stratified_split(true)
    assert [(s == x).sum() for x in ("train", "val", "test")] == [6000, 2000, 2000]
    for c in range(10):
        assert ((s == "test") & (true == c)).sum() == 200
    assert (D.stratified_split(true, seed=0) == s).all()


def test_cifar_one_vs_rest_and_joint_subsample():
    counts, true = _fake_cifar()
    t = D.cifar10h_labels(counts, true, "/img")
    assert t.k.shape == (1000, 10) and (t.n == counts.sum(1, keepdims=True)).all()
    for n_sub in (1, 3, 5, 10):
        s = D.subsample(t, n_sub, seed=1)
        assert (s.k.sum(1) == n_sub).all() and (s.k <= t.k).all() and (s.n == n_sub).all()
    assert (D.subsample(t, 3, seed=7).k == D.subsample(t, 3, seed=7).k).all()
    assert (D.subsample(t, 3, seed=7).k != D.subsample(t, 3, seed=8).k).any()


def test_subsample_is_unbiased():
    t = D.LabelTable(np.array(["a"]), np.array([[2.0]]), np.array([[7.0]]), np.array([0]),
                     np.array(["train"]), np.array(["p"]))
    draws = np.array([D.subsample(t, 3, seed=s).k[0, 0] for s in range(4000)])
    assert abs(draws.mean() - 3 * 2 / 7) < 0.03
    with pytest.raises(ValueError):
        D.subsample(t, 8, seed=0)


def test_ambiguous_mask():
    counts = np.array([[50, 1, 0], [30, 20, 1], [45, 5, 0]], dtype=float)
    t = D.cifar10h_labels(counts, [0, 0, 0], "/img")
    assert D.ambiguous_mask(t).tolist() == [False, True, False]
    b = D.LabelTable(np.array(["a", "b"]), np.array([[3.0, 0.0], [3.0, 1.0]]), np.full((2, 2), 3.0),
                     np.arange(2), np.array(["test"] * 2), np.array(["p"] * 2))
    assert D.ambiguous_mask(b).tolist() == [False, True]
    assert D.split_frame_counts(b).tolist() == [0, 1]


def test_mhist_parsing(tmp_path):
    df = pd.DataFrame({"Image Name": ["a.png", "b.png", "c.png", "d.png", "e.png"],
                       "Majority Vote Label": ["SSA", "HP", "HP", "SSA", "HP"],
                       "Number of Annotators who Selected SSA (Out of 7)": [5, 0, 3, 7, 1],
                       "Partition": ["train", "train", "train", "test", "train"]})
    df.to_csv(tmp_path / "annotations.csv", index=False)
    t = D.mhist_labels(tmp_path / "annotations.csv", "/img", val_frac=0.25)
    assert t.k[:, 0].tolist() == [5, 0, 3, 7, 1] and (t.n == 7).all()
    assert (t.split == "test").sum() == 1 and (t.split == "val").sum() == 1
    df.loc[0, "Majority Vote Label"] = "HP"
    df.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError):
        D.mhist_labels(tmp_path / "bad.csv", "/img")


def test_imageset_shapes(tmp_path):
    from PIL import Image
    Image.new("RGB", (854, 480)).save(tmp_path / "x.jpg")
    Image.new("RGB", (32, 32)).save(tmp_path / "y.png")
    for name, p, size in (("endoscapes", "x.jpg", 384), ("cifar10h", "y.png", 224)):
        t = D.LabelTable(np.array(["x"]), np.array([[1.0]]), np.array([[3.0]]), np.array([0]),
                         np.array(["train"]), np.array([str(tmp_path / p)]))
        img, k, n, i = D.ImageSet(t, name)[0]
        assert img.shape == (3, size, size) and k.item() == 1 and n.item() == 3 and i == 0


MHIST = D.DATA_ROOT / "mhist" / "annotations.csv"


@pytest.mark.skipif(not MHIST.exists(), reason="MHIST annotations not present")
def test_mhist_real():
    t = D.mhist_labels(MHIST, D.DATA_ROOT / "mhist" / "images")
    assert {s: (t.split == s).sum() for s in ("train", "val", "test")} == {"train": 1740, "val": 435, "test": 977}
    assert set(np.unique(t.k)) == set(range(8)) and (t.n == 7).all()
    assert (t.k > 3.5).sum() == 990                       # SSA majority count (MHIST paper: 990 SSA / 2162 HP)


C10H = D.DATA_ROOT / "cifar10h"


@pytest.mark.skipif(not (C10H / "cifar10h-counts.npy").exists(), reason="CIFAR-10H counts not present")
def test_cifar10h_real():
    c, y = np.load(C10H / "cifar10h-counts.npy"), np.load(C10H / "test_labels.npy")
    assert c.shape == (10000, 10) and c.sum(1).min() == 47 and c.sum(1).max() == 63
    assert (c.argmax(1) == y).mean() == pytest.approx(0.9921, abs=1e-4)      # human plurality vs CIFAR label
    t = D.cifar10h_labels(c, y, C10H / "images")
    assert {s: (t.split == s).sum() for s in ("train", "val", "test")} == {"train": 6000, "val": 2000, "test": 2000}
    for s in ("train", "val", "test"):
        assert (np.bincount(y[t.split == s]) == {"train": 600, "val": 200, "test": 200}[s]).all()
    assert D.ambiguous_mask(t).sum() == 1211


@pytest.mark.skipif(not (ENDO / "all_metadata.csv").exists(), reason="Endoscapes metadata not present")
def test_ids_are_plain_strings():
    """Object-dtype ids cannot be read back from .npz without pickle (caches and prediction files)."""
    assert D.endoscapes_labels(ENDO).ids.dtype.kind == "U"


def _fake_amibr(root, rng):
    rows = []
    for ds, n_sl in (("MIDOG21", 10), ("TUPAC16", 5)):
        for s in range(n_sl):
            for u in range(4):
                v = rng.random(3) < 0.3
                rows.append({"slide": f"{s:03d}.tiff", "dataset": ds, "uid": s * 10 + u, "x": 0, "y": 0,
                             **{c: bool(x) for c, x in zip(D.AMIBR_RATERS, v)}, "majority_atypical": bool(v.sum() >= 2)})
    pd.DataFrame(rows).to_csv(root / "AMI-BR.csv")
    return pd.DataFrame(rows)


def test_amibr_parsing(tmp_path):
    df = _fake_amibr(tmp_path, np.random.default_rng(0))
    t = D.amibr_labels(tmp_path)
    assert (t.n == 3).all() and t.k[:, 0].tolist() == df[list(D.AMIBR_RATERS)].sum(1).tolist()
    # same slide name in both sources stays two slides; every slide lies in one split
    assert len(np.unique(t.cluster)) == 15
    for c in np.unique(t.cluster):
        assert len(np.unique(t.split[t.cluster == c])) == 1
    # stratified by source: each source contributes to every split
    for ds in ("MIDOG21", "TUPAC16"):
        assert set(t.split[np.char.startswith(t.ids, ds)]) == {"train", "val", "test"}
    assert t.paths[0].endswith(("atypical/MIDOG21_0.png", "normal/MIDOG21_0.png"))
    assert (np.char.find(t.paths, "/atypical/") >= 0).tolist() == df.majority_atypical.tolist()
    df.loc[0, "majority_atypical"] = not df.loc[0, "majority_atypical"]
    df.to_csv(tmp_path / "AMI-BR.csv")
    with pytest.raises(ValueError):
        D.amibr_labels(tmp_path)


AMIBR = D.DATA_ROOT / "amibr" / "AMI-BR.csv"


@pytest.mark.skipif(not AMIBR.exists(), reason="AMi-Br not present (cluster only)")
def test_amibr_real():
    t = D.amibr_labels(AMIBR.parent)
    assert len(t.ids) == 3720 and len(np.unique(t.cluster)) == 202
    assert np.bincount(t.k[:, 0].astype(int)).tolist() == [2496, 392, 420, 412]
    assert all(__import__("os").path.exists(p) for p in t.paths)
