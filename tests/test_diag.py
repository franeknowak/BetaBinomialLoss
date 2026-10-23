"""Round D, D0 diagnostics (b2j/diag.py) on fake runs."""
import json
from dataclasses import asdict

import numpy as np

from b2j import data as D
from b2j import diag
from b2j.runs import RunSpec


def _fake(tmp_path, monkeypatch, ds="cifar10h", N=60, K=10):
    rng = np.random.default_rng(0)
    counts = rng.multinomial(50, np.ones(K) / K, size=N).astype(float)
    t = D.LabelTable(np.array([f"{i:05d}" for i in range(N)]), counts.copy(), np.repeat(counts.sum(1, keepdims=True), K, 1),
                     np.arange(N), np.array(["train", "val"] * (N // 2)), np.array([""] * N), counts=counts)
    monkeypatch.setattr(diag, "load_table", lambda name: t)
    import b2j.analysis as A
    monkeypatch.setattr(A, "load_table", lambda name: t)
    for v in ("soft_bce", "b2r", "b2ms_shrink0.3"):
        for n_sub in (None, 3):
            for s in range(2):
                spec = RunSpec(ds, v, seed=s, n_sub=n_sub, block="blk")
                d = spec.out_dir(tmp_path)
                d.mkdir(parents=True)
                (d / "spec.json").write_text(json.dumps(asdict(spec)))
                (d / "summary.json").write_text("{}")
                out = rng.normal(size=(N // 2, K, 1 if v == "soft_bce" else 2))
                np.savez(d / "val_B.npz", ids=t.take("val").ids, out=out, scale=np.full(K, 2.0))
                np.savez(d / "val_A.npz", ids=t.take("val").ids, out=out, scale=np.full(K, 2.0))
    return t


def test_report_runs_and_bins_cover_everything(tmp_path, monkeypatch):
    t = _fake(tmp_path, monkeypatch)
    runs = diag.main_runs("blk", "cifar10h", tmp_path)
    rows = diag.gain_by_ambiguity(runs, t.take("val"))
    for s in (0, 1):
        r = [x for x in rows if x["seed"] == s]
        assert abs(sum(x["share"] for x in r) - 1) < 1e-9                 # the bins partition the (image, head) pairs
        assert abs(sum(x["gain_sum"] for x in r) - r[0]["total"]) < 1e-6
    ref, res = diag.cifar_ceiling("blk", tmp_path, t)
    assert np.isfinite(ref) and set(res) >= {("soft_bce", 3), ("b2r", "all"), ("soft_bce plug-in", 3)}
    text = diag.report("blk", tmp_path, datasets=("cifar10h",)).read_text()
    assert "Ceiling" in text and "unanimous" in text and "### b2ms_shrink0.3" in text
    assert "b2ms_shrink0.3 closes" in text and "Concentration ablation" in text


def test_concentration_ablation_is_zero_for_a_constant_concentration(tmp_path, monkeypatch):
    t = _fake(tmp_path, monkeypatch)
    runs = diag.main_runs("blk", "cifar10h", tmp_path)
    ev = t.take("val")
    r = diag.concentration_ablation(runs, ev, "b2r")
    assert len(r) == 2 and all(np.isfinite(list(x.values())).all() for x in r)
    monkeypatch.setattr(diag, "_panel", lambda v, d: (np.full(ev.k.shape, 0.1), np.full(ev.k.shape, 5.0)))
    r = diag.concentration_ablation(runs, ev, "b2r")
    for x in r:                                   # a per-head constant S: the fitted one can only be as good or better
        assert x["nll_const"] <= x["nll_own"] + 1e-9 and abs(x["auroc_own"] - x["auroc_const"]) < 1e-12


def test_mean_recovery_scores_against_the_full_panel(tmp_path, monkeypatch):
    t = _fake(tmp_path, monkeypatch)
    monkeypatch.setattr(diag, "boot_strata", lambda ds, tab: None)
    res = diag.mean_recovery("blk", "cifar10h", tmp_path, B=50)
    assert set(res) == {(ck, n) for ck in "AB" for n in (3, "all")}
    ev = t.take("val")
    share = ev.k / ev.n
    for r in res.values():
        assert r["seeds"] == (2, 2) and r["lo"] <= r["diff"] <= r["hi"]
        assert abs(r["diff"] - (r["mse_soft_bce"] - r["mse_b2r"])) < 1e-12
        assert abs(r["floor"] - (share * (1 - share) / (ev.n - 1)).mean()) < 1e-12
    text = diag.recovery_report("blk", tmp_path, datasets=("cifar10h",)).read_text()
    assert "| A | 3 |" in text and "Noise floor" in text
