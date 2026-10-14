"""Unit tests for module f (run specs, splits, launcher paths). CPU; fake features; stub encoder for images.

Run: python -m pytest tests/test_module_f.py -q
"""
import json

import numpy as np
import pytest
import torch.nn as nn

from b2j import embed as E
from b2j import runs as R
from b2j.blocks import BLOCKS
from b2j.data import DATA_ROOT, LabelTable, load_table

HAVE_C10H = (DATA_ROOT / "cifar10h" / "cifar10h-counts.npy").exists()


def test_names_unique_and_smoke_block():
    specs = BLOCKS["smoke"]()
    assert len({(s.dataset, s.name) for s in specs}) == len(specs) == 36
    assert {s.regime for s in specs} == {"frozen", "finetune", "rehead"}


@pytest.mark.skipif(not HAVE_C10H, reason="CIFAR-10H counts not present")
def test_tables_subsample_part_and_test_full_panel():
    full = load_table("cifar10h")
    tr, va, te = R.tables(R.RunSpec("cifar10h", "b2r", n_sub=3), full)
    assert (tr.n == 3).all() and (va.n == 3).all() and (te.n >= 47).all()          # test keeps the full panel
    assert np.array_equal(te.k, full.take("test").k)
    _, va_full, _ = R.tables(R.RunSpec("cifar10h", "b2r", n_sub=3, val_full_panel=True), full)
    assert (va_full.n >= 47).all()
    a, _, _ = R.tables(R.RunSpec("cifar10h", "b2r", part="A", seed=2), full)
    b, _, _ = R.tables(R.RunSpec("cifar10h", "b2r", part="B", seed=2), full)
    assert len(a.ids) + len(b.ids) == 6000 and not set(a.ids) & set(b.ids)
    assert abs(len(a.ids) / 6000 - 0.8) < 0.01


@pytest.mark.skipif(not HAVE_C10H, reason="CIFAR-10H counts not present")
def test_frozen_run_and_skip(tmp_path, monkeypatch):
    full = load_table("cifar10h")
    monkeypatch.setattr(E, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    rng = np.random.default_rng(0)
    np.savez(E.cache_path("cifar10h"), ids=full.ids[::-1], X=rng.normal(size=(len(full.ids), 16)).astype(np.float32))
    spec = R.RunSpec("cifar10h", "b2r", n_sub=5, epochs=1, block="t")
    s = R.execute(spec, root=tmp_path / "res", device="cpu", verbose=False)
    out = spec.out_dir(tmp_path / "res")
    assert json.loads((out / "spec.json").read_text())["n_sub"] == 5 and s["n_train"] == 6000
    assert np.load(out / "test_B.npz")["out"].shape == (2000, 10, 2)
    (out / "log.jsonl").write_text("")                     # a finished run is skipped, not retrained
    R.execute(spec, root=tmp_path / "res", device="cpu", verbose=False)
    assert (out / "log.jsonl").read_text() == ""


class Stub(nn.Module):
    num_features = 8

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 8, 7, stride=8), nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


def test_finetune_and_rehead_paths(tmp_path, monkeypatch):
    from PIL import Image
    rng = np.random.default_rng(0)
    N = 40
    ids = np.array([f"{i:05d}" for i in range(N)])
    for i in ids:
        Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(tmp_path / f"{i}.png")
    k = rng.integers(0, 6, (N, 1)).astype(float)
    table = LabelTable(ids, k, np.full_like(k, 5.0), np.arange(N), np.array(["train"] * 24 + ["val"] * 8 + ["test"] * 8),
                       np.array([str(tmp_path / f"{i}.png") for i in ids]))
    monkeypatch.setattr(R, "load_table", lambda ds: table)
    monkeypatch.setattr(E, "load_encoder", lambda ds: Stub())
    kw = dict(encoder_lr=1e-4, epochs=1, block="t")
    s = R.execute(R.RunSpec("cifar10h", "soft_bce", regime="finetune", **kw), root=tmp_path / "res", device="cpu", verbose=False, num_workers=0)
    assert s["finetune"] and s["n_train"] == 24
    spec = R.RunSpec("cifar10h", "b2r", regime="rehead", **kw)
    s = R.execute(spec, root=tmp_path / "res", device="cpu", verbose=False, num_workers=0)
    out = spec.out_dir(tmp_path / "res")
    assert s["finetune"]["n_train"] + s["rehead"]["n_train"] == 24                     # parts A and B
    assert (out / "features_ftB.npz").exists() and (out / "rehead" / "test_B.npz").exists()
    assert not list(out.rglob("*encoder*"))
    with pytest.raises(ValueError):
        R.execute(R.RunSpec("cifar10h", "b2r", regime="finetune", block="t2"), root=tmp_path / "res", device="cpu")


@pytest.mark.skipif(not (DATA_ROOT / "endoscapes" / "all_metadata.csv").exists(), reason="Endoscapes metadata not present")
def test_smoke_report_on_fake_block(tmp_path, monkeypatch):
    from b2j import report_smoke as RS
    full = load_table("endoscapes")
    monkeypatch.setattr(E, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    np.savez(E.cache_path("endoscapes"), ids=full.ids,
             X=np.random.default_rng(0).normal(size=(len(full.ids), 16)).astype(np.float32))
    for v in ("b2r", "soft_bce"):
        R.execute(R.RunSpec("endoscapes", v, epochs=1, block="smoke"), root=tmp_path / "res", device="cpu", verbose=False)
    # epochs=1 changes the run name; the report looks for the default-name runs, so link them
    for v in ("b2r", "soft_bce"):
        src = tmp_path / "res" / "smoke" / "endoscapes" / f"{v}__frozen__lr0.0003_wd0.01__ep1__seed0"
        src.rename(src.with_name(f"{v}__frozen__lr0.0003_wd0.01__seed0"))
    bd = tmp_path / "res" / "smoke"
    table = RS.rows(RS.load_runs(bd))
    assert len(table) == 2 and all(r["n_evals"] == 4 for r in table)
    res, boot = RS.module_e_check(bd, B=50)
    assert set(res) == {"b2r", "soft_bce"} and np.isfinite(res["b2r"]["nll_per_vote"])
    assert boot.lo <= boot.estimate <= boot.hi
    assert RS.eps_rule([{"run": "endoscapes/b2r__frozen", "nonfinite": 0, "stuck_max": 0.2}]) == ["endoscapes/b2r__frozen"]
    assert RS.eps_rule([{"run": "endoscapes/b2m__frozen", "nonfinite": 3, "stuck_max": 0.2}]) == []


def test_analysis_refuses_test_before_r3(tmp_path):
    from b2j import analysis as A
    with pytest.raises(PermissionError):
        A.collect("smoke", "endoscapes", split="test", root=tmp_path)
    with pytest.raises(PermissionError):
        A.figures("smoke", split="test", root=tmp_path)
