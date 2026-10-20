"""Unit tests for module c (embedding cache, trainer) and the post-hoc spread fit it uses.

Run: python -m pytest tests/test_module_c.py -q   (CPU; synthetic data; a stub encoder replaces DINOv3)
"""
import json

import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy.stats import betabinom, binom

from b2j import embed as E
from b2j import posthoc as P
from b2j import train as T
from b2j.data import ImageSet, LabelTable
from b2j.variants import VARIANTS


# ---------------------------------------------------------------- post-hoc
def test_vote_nll_matches_scipy():
    k, n = np.array([0, 1, 2, 3, 5]), np.array([3, 3, 3, 3, 7])
    m, S = 0.3, 2.5
    np.testing.assert_allclose(P.vote_nll(m, S, k, n), -betabinom.logpmf(k, n, m * S, (1 - m) * S), rtol=1e-10)
    np.testing.assert_allclose(P.vote_nll(m, np.inf, k, n), -binom.logpmf(k, n, m), rtol=1e-10)
    np.testing.assert_allclose(P.vote_nll(m, 1e9, k, n), P.vote_nll(m, np.inf, k, n), rtol=1e-6)


def test_fit_scale_recovers_spread():
    rng = np.random.default_rng(0)
    N, S = 20000, 4.0
    m = rng.uniform(0.1, 0.9, (N, 2))
    k = rng.binomial(3, rng.beta(m * S, (1 - m) * S)).astype(float)
    n = np.full_like(k, 3.0)
    s = P.fit_scale(m, np.ones_like(m), k, n)                 # global S*
    np.testing.assert_allclose(s, S, rtol=0.15)
    c = P.fit_scale(m, np.full_like(m, 2.0), k, n)            # c with S0 = 2 -> c = 2
    np.testing.assert_allclose(c, S / 2, rtol=0.15)
    assert np.isinf(P.fit_scale(m, np.ones_like(m), k[:, :] * 0, np.ones_like(k))).all()   # n = 1: tie-break inf


def test_spread_rules_and_eps():
    assert {v: P.spread_rule(v) for v in VARIANTS} == {
        "hard_bce": "global", "hard_bce_w": "global", "soft_bce": "global", "b2r": "c", "b2r_w": "c",
        "b2r_kl": "c", "b2m": "none", "b2r_linear": "c", "dup": "dup", "b2ms": "c", "b2ms_shrink0.03": "c", "b2ms_shrink0.3": "c", "b2ms_shrink3": "c",
        "b2ms_var0.03": "c", "b2ms_var0.3": "c", "b2ms_var3": "c", "soft_bce_n": "global"}
    out = np.full((1, 1, 2), -50.0)
    assert P.mean_spread("b2r", out)[1][0, 0] == pytest.approx(2e-3)
    assert P.mean_spread("b2r", out, eps=0.1)[1][0, 0] == pytest.approx(0.2)
    assert P.mean_spread("b2m", out, eps=0.1)[1][0, 0] == pytest.approx(2 + 2e-6)     # B²-M ignores the override
    m, S = P.panel("soft_bce", np.zeros((4, 2, 1)), scale=np.array([3.0, 5.0]))
    assert (m == 0.5).all() and (S == [3.0, 5.0]).all()


# ---------------------------------------------------------------- trainer on synthetic embeddings
def synthetic(N=600, D=16, K=2, n=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D)).astype(np.float32)
    w = rng.normal(size=(D, K))
    m = 1 / (1 + np.exp(-(X @ w)))
    k = rng.binomial(n, rng.beta(m * 3, (1 - m) * 3)).astype(np.float64)
    return X, k, np.full_like(k, float(n))


def splits(N=600, **kw):
    X, k, n = synthetic(N, **kw)
    ids = np.array([f"i{i}" for i in range(N)])
    a, b = int(0.6 * N), int(0.8 * N)
    return (T.Split(X[:a], ids[:a], k[:a], n[:a]), T.Split(X[a:b], ids[a:b], k[a:b], n[a:b]),
            T.Split(X[b:], ids[b:]))


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_every_variant_trains_and_writes(variant, tmp_path, capsys):
    tr, va, te = splits()
    s = T.run(T.TrainConfig(variant, epochs=4, lr=1e-2), tr, va, te, tmp_path, device="cpu")
    for c in "AB":
        for part, sp in (("val", va), ("test", te)):
            z = np.load(tmp_path / f"{part}_{c}.npz")
            assert z["out"].shape == (len(sp.ids), 2, VARIANTS[variant].loss.out_dim) and (z["ids"] == sp.ids).all()
            assert np.isfinite(z["out"]).all()
        assert (tmp_path / f"head_{c}.pt").exists()
        assert 0 <= s[c]["eval"] < 16 and 0 < s[c]["epoch"] <= 4
    log = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    ep = [r["epoch"] for r in log]                  # 4 evaluations per epoch, at whole batches
    assert len(log) == 16 and ep[3] == 1.0 and ep[-1] == 4.0 and np.all(np.diff(ep) > 0)
    # test silence: nothing named test in any logged or printed quantity
    text = (tmp_path / "log.jsonl").read_text() + (tmp_path / "summary.json").read_text() + capsys.readouterr().out
    assert "test" not in text
    assert not list(tmp_path.glob("*encoder*"))


def _smoothed(v, h):
    return np.array([np.nanmean(v[max(0, j - h): j + h + 1]) for j in range(len(v))])


@pytest.mark.parametrize("smooth", [1, 3, 5])
def test_checkpoints_match_log(tmp_path, smooth):
    tr, va, te = splits()
    s = T.run(T.TrainConfig("b2r", epochs=3, lr=1e-2, smooth=smooth), tr, va, te, tmp_path, device="cpu", verbose=False)
    log = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    h = smooth // 2
    assert s["A"]["eval"] == int(np.argmax(_smoothed(np.array([r["val_map"] for r in log]), h)))
    assert s["B"]["eval"] == int(np.argmin(_smoothed(np.array([r["val_nll_per_vote"] for r in log]), h)))
    # the kept state is the real state at that evaluation: saved val_B outputs reproduce its logged (unsmoothed) NLL
    z = np.load(tmp_path / "val_B.npz")
    m, S = P.panel("b2r", z["out"], z["scale"])
    assert (P.vote_nll(m, S, va.k, va.n) / va.n).mean() == pytest.approx(log[s["B"]["eval"]]["val_nll_per_vote"], rel=1e-5)
    assert s["B"]["val_map"] > 0.7                  # learnable synthetic task
    assert len(s["B"]["train_val_S_ratio"]) == 2 and len(log[0]["train_stuck_share"]) == 2


def test_smoothing_rejects_lone_spike():
    """With a lone spike, the raw best and the smoothed best differ; smoothing picks the plateau."""
    v = np.array([0.50, 0.55, 0.95, 0.50, 0.70, 0.74, 0.73, 0.60])
    assert int(np.argmax(v)) == 2 and int(np.argmax(_smoothed(v, 1))) == 5
    with pytest.raises(ValueError):
        T.TrainConfig("b2r", smooth=2)


def test_deterministic(tmp_path):
    tr, va, te = splits()
    outs = []
    for r in ("a", "b"):
        T.run(T.TrainConfig("soft_bce", epochs=3, seed=7), tr, va, te, tmp_path / r, device="cpu", verbose=False)
        outs.append(np.load(tmp_path / r / "val_B.npz")["out"])
    np.testing.assert_array_equal(*outs)


def test_refuses_test_labels(tmp_path):
    tr, va, te = splits()
    with pytest.raises(ValueError):
        T.run(T.TrainConfig("b2r", epochs=1), tr, va, T.Split(te.x, te.ids, np.zeros((len(te.ids), 2)), None),
              tmp_path, device="cpu")
    with pytest.raises(ValueError):    # encoder given without encoder_lr
        T.run(T.TrainConfig("b2r", epochs=1), tr, va, te, tmp_path, encoder=nn.Identity(), device="cpu")


# ---------------------------------------------------------------- images, fine-tuning, cache (stub encoder)
class StubEncoder(nn.Module):
    num_features = 8

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 8, 7, stride=8), nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


@pytest.fixture
def image_table(tmp_path):
    from PIL import Image
    rng = np.random.default_rng(0)
    N = 24
    ids = np.array([f"{i:05d}" for i in range(N)])
    for i in ids:
        Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(tmp_path / f"{i}.png")
    k = rng.integers(0, 4, (N, 1)).astype(float)
    split = np.array(["train"] * 12 + ["val"] * 6 + ["test"] * 6)
    return LabelTable(ids, k, np.full_like(k, 3.0), np.arange(N), split, np.array([str(tmp_path / f"{i}.png") for i in ids]))


def test_finetune_path_and_feature_cache(image_table, tmp_path, monkeypatch):
    t = image_table
    sp = {s: T.split_from_table(t.take(s), ImageSet(t.take(s), "cifar10h"), labelled=s != "test")
          for s in ("train", "val", "test")}
    cfg = T.TrainConfig.finetune("b2r", encoder_lr=1e-4, epochs=2, batch_size=4, num_workers=0)
    enc = StubEncoder()
    w0 = enc.net[0].weight.detach().clone()
    s = T.run(cfg, sp["train"], sp["val"], sp["test"], tmp_path / "run", encoder=enc, device="cpu", verbose=False)
    assert s["finetune"] and not torch.equal(enc.net[0].weight, w0)                # encoder was trained (state at B)
    assert np.load(tmp_path / "run" / "test_B.npz")["out"].shape == (6, 1, 2)
    log = [json.loads(l) for l in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    assert all("train_map" in r and "train_nll_per_vote" in r for r in log)       # training-set probe on images
    # feature cache from a trained encoder, then a frozen head on it
    model = T.Model(None, enc)
    monkeypatch.setattr(E, "CACHE_DIR", tmp_path / "cache")
    path = E.build_cache(t, "cifar10h", torch.device("cpu"), tag="ft_test", encoder=model.encoder, num_workers=0)
    shuffled = t.subset(np.random.default_rng(1).permutation(len(t.ids)))
    X = E.load_cache(shuffled, path)
    direct = E.embed(model.encoder, ImageSet(shuffled, "cifar10h"), torch.device("cpu"), num_workers=0)
    np.testing.assert_allclose(X, direct, rtol=1e-5, atol=1e-6)                    # aligned by id
    assert "k" not in np.load(path).files                                          # no labels in the cache
    with pytest.raises(KeyError):
        E.load_cache(LabelTable(np.array(["nope"]), *(getattr(t, f)[:1] for f in ("k", "n", "cluster", "split", "paths"))), path)
    fr = {s: T.split_from_table(t.take(s), E.load_cache(t.take(s), path), labelled=s != "test") for s in ("train", "val", "test")}
    T.run(T.TrainConfig("b2r", epochs=2), fr["train"], fr["val"], fr["test"], tmp_path / "head", device="cpu", verbose=False)
    assert (tmp_path / "head" / "val_B.npz").exists()
