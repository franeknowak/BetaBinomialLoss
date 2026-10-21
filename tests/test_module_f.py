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
    assert len({(s.dataset, s.name) for s in specs}) == len(specs) == 34
    assert {s.regime for s in specs} == {"frozen", "finetune"}


@pytest.mark.skipif(not HAVE_C10H, reason="CIFAR-10H counts not present")
def test_tables_subsample_fraction_and_test_full_panel():
    full = load_table("cifar10h")
    tr, va, te = R.tables(R.RunSpec("cifar10h", "b2r", n_sub=3), full)
    assert (tr.n == 3).all() and (va.n == 3).all() and (te.n >= 47).all()          # test keeps the full panel
    assert np.array_equal(te.k, full.take("test").k)
    _, va_full, _ = R.tables(R.RunSpec("cifar10h", "b2r", n_sub=3, val_full_panel=True), full)
    assert (va_full.n >= 47).all()
    f10, f30 = (R.tables(R.RunSpec("cifar10h", "b2r", train_frac=f, seed=2), full)[0] for f in (0.1, 0.3))
    assert (len(f10.ids), len(f30.ids)) == (600, 1800) and set(f10.ids) <= set(f30.ids)      # nested
    other = R.tables(R.RunSpec("cifar10h", "soft_bce", train_frac=0.1, seed=2), full)[0]
    assert np.array_equal(other.ids, f10.ids)                                            # same rows for every variant
    assert not np.array_equal(R.tables(R.RunSpec("cifar10h", "b2r", train_frac=0.1, seed=3), full)[0].ids, f10.ids)
    assert "frac0.1" in R.RunSpec("cifar10h", "b2r", train_frac=0.1).name
    assert R.RunSpec.from_json({"dataset": "x", "variant": "b2r", "part": "B"}) is None
    assert R.RunSpec.from_json({"dataset": "x", "variant": "b2r", "part": "all"}) == R.RunSpec("x", "b2r")


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


def test_finetune_path(tmp_path, monkeypatch):
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
    assert not list((tmp_path / "res").rglob("*encoder*"))
    with pytest.raises(ValueError):
        R.execute(R.RunSpec("cifar10h", "b2r", regime="finetune", block="t2"), root=tmp_path / "res", device="cpu")
    with pytest.raises(ValueError):
        R.execute(R.RunSpec("cifar10h", "b2r", regime="rehead", encoder_lr=1e-4, block="t3"), root=tmp_path / "res", device="cpu")


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


def _fake_block(tmp_path, seeds=3, ds="mhist", variants=("soft_bce", "b2r")):
    """Fake mhist-like table (N = 300, K = 1, n = 7) and frozen soft BCE / B²-r runs at n = 7 (full) and n = 3,
    and at 10% of the training clusters."""
    from dataclasses import asdict
    rng = np.random.default_rng(0)
    N = 300
    theta = rng.beta(1, 1.5, (N, 1))
    k = rng.binomial(7, theta).astype(float)
    table = LabelTable(np.array([f"{i:04d}" for i in range(N)]), k, np.full_like(k, 7.0), np.arange(N),
                       np.array(["train"] * 100 + ["val"] * 100 + ["test"] * 100), np.array([""] * N))
    val = table.take("val")
    z = np.log(theta[200 - 100:200] / (1 - theta[200 - 100:200]))
    for n_sub, frac in ((None, 1.0), (3, 1.0), (None, 0.1)):
        for v in variants:
            for seed in range(seeds):
                spec = R.RunSpec(ds, v, seed=seed, n_sub=n_sub, train_frac=frac, block="fake")
                d = spec.out_dir(tmp_path)
                d.mkdir(parents=True)
                (d / "spec.json").write_text(json.dumps(asdict(spec)))
                (d / "summary.json").write_text("{}")
                zz = z + rng.normal(0, 0.5, z.shape)
                bce = v in ("soft_bce", "hard_bce", "hard_bce_w")
                out = zz[..., None] if bce else np.stack([-zz / 2 + 1, zz / 2 + 1], -1)
                for c in "AB":
                    np.savez(d / f"val_{c}.npz", ids=val.ids, out=out, scale=np.array([4.0 if bce else 1.0]))
    return table


def test_fig2_and_fig5_on_fake_block(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from b2j import analysis as A
    table = _fake_block(tmp_path)
    monkeypatch.setattr(A, "load_table", lambda ds: table)
    out = tmp_path / "fig"
    assert A.fig2_advantage_vs_n("fake", "mhist", "val", out, root=tmp_path, B=50).exists()
    f2 = json.loads((out / "F2_advantage_mhist.json").read_text())
    assert set(f2) == {"3", "7"} and all(r["lo"] <= r["estimate"] <= r["hi"] for r in f2.values())
    assert all(r["seeds_gs"] == r["seeds_b2r"] == 3 for r in f2.values())
    assert A.fig5_pr_and_risk("fake", "mhist", "val", out, root=tmp_path, B=50).exists()
    f5 = json.loads((out / "F5_pr_mhist.json").read_text())
    assert set(f5["precision"]) == {"B²-r P_maj", "B²-r m", "soft BCE m"}
    d = f5["diff_pmaj_minus_m"]
    assert len(d["estimate"]) == len(A.RECALL_GRID)
    assert all(lo <= hi for lo, hi in zip(d["lo"], d["hi"]))
    df = A.collect("fake", "mhist", "val", root=tmp_path)
    assert not df["row"].str.contains(r"S\*").any() and "soft_bce:plugin" in set(df["row"])
    assert {"ensemble", "ensemble_gs", "ensemble_b2r"} <= set(df["row"])                    # D3 rows
    d = sorted((tmp_path / "fake" / "mhist").glob("b2r__frozen__lr*seed0"))[0]
    ev = table.take("val")
    one = A.run_rows("b2r", d, "val", ev, ev)["b2r"]
    two = A.ensemble_row([d, d], "val", ev, ev, "b2r", spread=True)               # identical members = the member
    assert np.allclose(one["pmf"], two["pmf"]) and np.allclose(one["m"], two["m"])


def test_train_fraction_figure_on_fake_block(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from b2j import analysis as A
    table = _fake_block(tmp_path, ds="endoscapes")
    monkeypatch.setattr(A, "load_table", lambda ds: table)
    assert A.figS_train_fraction("fake", "val", tmp_path / "fig", root=tmp_path, B=50, datasets=("endoscapes",)).exists()
    r = json.loads((tmp_path / "fig" / "S_train_fraction.json").read_text())["endoscapes"]
    assert set(r) == {"0.1", "1.0"} and all(v["map"]["lo"] <= v["map"]["hi"] for v in r.values())


def test_verdicts_on_fake_block(tmp_path, monkeypatch):
    from b2j import analysis as A
    from b2j import verdicts as V
    table = _fake_block(tmp_path, variants=("soft_bce", "b2r", "b2r_w", "b2r_kl", "hard_bce", "hard_bce_w"))
    monkeypatch.setattr(V, "load_table", lambda ds: table)
    rows = V.read("fake", "val", root=tmp_path, B=30, datasets=("mhist",))
    got = {(r["id"], r["contrast"]): r for r in rows}
    assert {r["id"] for r in rows} == {"S0", "ST-w", "ST-KL", "S4a", "S4b", "S6b"}
    assert all(r["reading"] in ("success", "fail", "inconclusive", "better", "worse", "no evidence of a difference")
               for r in rows)
    assert all(r["lo"] <= r["estimate"] <= r["hi"] or r["lo"] <= r["hi"] for r in rows)
    assert (tmp_path / "fake" / "verdicts_val.md").exists()
    with pytest.raises(PermissionError):
        V.read("fake", "test", root=tmp_path, datasets=("mhist",))


def test_preflight_votes_give_finite_loss_for_every_variant():
    # the GPU preflight's fake votes on the CPU: every variant must build and give a finite loss (job 7575102)
    import torch
    from b2j.launch import PREFLIGHT_K
    from b2j.variants import VARIANTS, build
    k, n = PREFLIGHT_K, torch.full_like(PREFLIGHT_K, 3.0)
    for name in VARIANTS:
        head, loss = build(name, 8, k, n)
        value = loss(head(torch.randn(len(k), 8)), k, n)
        value.backward()
        assert torch.isfinite(value), name
