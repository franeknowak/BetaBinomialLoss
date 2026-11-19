"""Tests for the three-rater experiment and the automatic grid extension.
Run: python -m pytest tests/test_three_raters.py -q
"""
import json

import numpy as np
import pytest
import torch

from b2j import experiment as E
from b2j import three_raters as R
from b2j import train as T
from tests.test_pipeline import _fake_cache


def test_draw_binary_label_takes_three_votes_without_replacement():
    rng = np.random.default_rng(0)
    k = np.array([[0.0], [7.0], [3.0], [1.0]] * 500)
    n = np.full_like(k, 7.0)
    d = R.draw_votes(k, n, rng)
    assert d.shape == k.shape and ((d >= 0) & (d <= 3)).all()
    assert (d[k[:, 0] == 0] == 0).all() and (d[k[:, 0] == 7] == 3).all()
    assert (d[k[:, 0] == 1] <= 1).all()                                       # one "yes" vote cannot be drawn twice
    assert d[k[:, 0] == 3].mean() == pytest.approx(3 * 3 / 7, abs=0.08)       # hypergeometric mean r k / n


def test_draw_single_label_dataset_keeps_one_label_per_vote():
    rng = np.random.default_rng(1)
    k = np.array([[2.0, 1, 1, 0], [0, 0, 4, 0], [1, 1, 1, 1]] * 300)
    n = np.repeat(k.sum(1, keepdims=True), 4, 1)
    d = R.draw_votes(k, n, rng)
    assert (d.sum(1) == 3).all() and (d <= k).all()
    assert (d[1::3] == [0, 0, 3, 0]).all()


def test_draw_rejects_multilabel_votes_and_too_few_votes():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        R.draw_votes(np.array([[1.0, 1.0]]), np.array([[3.0, 3.0]]), rng)     # two labels, votes don't add up to n
    with pytest.raises(ValueError):
        R.draw_votes(np.array([[1.0]]), np.array([[2.0]]), rng)


def test_three_rater_data_draws_train_val_only_and_is_fixed(tmp_path):
    cache = _fake_cache(tmp_path, K=1, raters=7)
    data, labels = T.load_embeddings("toy", cache=cache)
    a, rec = R.three_rater_data(data)
    b, _ = R.three_rater_data(data)
    for s in ("train", "val"):
        assert (a[s][2] == 3).all() and torch.equal(a[s][1], b[s][1])        # same draw every time
        assert torch.equal(a[s][0], data[s][0])
        assert (a[s][1] <= data[s][1]).all() and (3 - a[s][1] <= data[s][2] - data[s][1]).all()
    assert all(torch.equal(x, y) for x, y in zip(a["test"][:3], data["test"][:3]))   # test: the full panel
    assert len(rec["ids"]) == sum(len(data[s][3]) for s in data) and (rec["n_full"] == 7).all()


def test_next_lr_follows_1_5_10_pattern():
    assert [E.next_lr(x, "high") for x in (1e-3, 5e-3, 1e-2)] == [5e-3, 1e-2, 5e-2]
    assert [E.next_lr(x, "low") for x in (5e-5, 1e-5, 5e-6)] == [1e-5, 5e-6, 1e-6]


def test_tune_and_extend_steps_until_off_the_edge(tmp_path, monkeypatch):
    """Fake losses: the best learning rate is 1e-2 for b2 and inside the grid for the other two."""
    calls = []
    best = {"b2": 1e-2, "soft_bce": 1e-4, "hard_bce": 5e-4}

    def fake_tune(dataset, lrs=E.LRS, levels=E.WD_LEVELS, losses=tuple(best), root=None, epochs=25, data=None, cache=None):
        calls.append((tuple(lrs), tuple(losses)))
        for loss in losses:
            for lr in lrs:
                for lv in levels:
                    d = E.tuning_dir(dataset, loss, lr, lv, root)
                    d.mkdir(parents=True, exist_ok=True)
                    v = abs(np.log(lr / best[loss])) + (1 - lv)
                    (d / "run.json").write_text(json.dumps({
                        "lr": lr, "wd_level": lv, "wd": 0.0, "evals_per_epoch": 4, "epochs": 25,
                        "best_val_loss": v, "best_val_loss_eval": 1, "status": "ok",
                        "curve": [{"eval": 1, "val_loss": v}]}))

    monkeypatch.setattr(E, "tune", fake_tune)
    sel = E.tune_and_extend("toy", root=tmp_path, data=({}, []))
    assert calls == [(E.LRS, tuple(best)), ((5e-3,), ("b2",)), ((1e-2,), ("b2",)), ((5e-2,), ("b2",))]
    assert sel["b2"]["lr"] == 1e-2 and not any(s["lr_at_edge"] for s in sel.values())


def test_three_raters_run_end_to_end(tmp_path, monkeypatch):
    """Small grid (2 lrs, 1 level, 1 epoch) on random data: the lr may sit at the edge, so final runs with allow_edge."""
    import functools
    cache = _fake_cache(tmp_path, K=1, raters=7)
    monkeypatch.setattr(R, "load_embeddings", lambda ds, device="cpu": T.load_embeddings(ds, cache=cache, device=device))

    def small(dataset, root, epochs, data):
        E.tune(dataset, lrs=(1e-4, 1e-3), levels=(1.0,), root=root, epochs=epochs, data=data)
        return E.select(dataset, root)

    monkeypatch.setattr(R, "tune_and_extend", small)
    monkeypatch.setattr(R, "final", functools.partial(E.final, allow_edge=True))
    root = tmp_path / "res"
    R.run("toy", seeds=(1,), root=root, epochs=1)
    z = np.load(root / "toy" / "draw.npz")
    assert set(z["split"]) == {"train", "val", "test"} and (z["k_drawn"] <= z["k_full"]).all()
    out = json.loads((root / "final" / "toy" / "b2" / "seed1" / "outputs_nll.json").read_text())
    n = {s: {i["n"][0] for i in out["images"] if i["split"] == s} for s in ("val", "test")}
    assert n == {"val": {3}, "test": {7}}
    m = z["split"] == "val"
    k_val = {i["id"]: i["k"][0] for i in out["images"] if i["split"] == "val"}
    assert all(k_val[str(i)] == k for i, k in zip(z["ids"][m], z["k_drawn"][m, 0]))
