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
                     np.arange(N), np.array(["val"] * N), np.array([""] * N), counts=counts)
    monkeypatch.setattr(diag, "load_table", lambda name: t)
    import b2j.analysis as A
    monkeypatch.setattr(A, "load_table", lambda name: t)
    for v in ("soft_bce", "b2r"):
        for n_sub in (None, 3):
            for s in range(2):
                spec = RunSpec(ds, v, seed=s, n_sub=n_sub, block="blk")
                d = spec.out_dir(tmp_path)
                d.mkdir(parents=True)
                (d / "spec.json").write_text(json.dumps(asdict(spec)))
                (d / "summary.json").write_text("{}")
                out = rng.normal(size=(N, K, 2 if v == "b2r" else 1))
                np.savez(d / "val_B.npz", ids=t.ids, out=out, scale=np.full(K, 2.0))
    return t


def test_report_runs_and_bins_cover_everything(tmp_path, monkeypatch):
    t = _fake(tmp_path, monkeypatch)
    runs = diag.main_runs("blk", "cifar10h", tmp_path)
    rows = diag.gain_by_ambiguity(runs, t)
    for s in (0, 1):
        r = [x for x in rows if x["seed"] == s]
        assert abs(sum(x["share"] for x in r) - 1) < 1e-9                 # the bins partition the (image, head) pairs
        assert abs(sum(x["gain_sum"] for x in r) - r[0]["total"]) < 1e-6
    ref, res = diag.cifar_ceiling("blk", tmp_path, t)
    assert np.isfinite(ref) and set(res) >= {("soft_bce", 3), ("b2r", "all"), ("soft_bce plug-in", 3)}
    text = diag.report("blk", tmp_path, datasets=("cifar10h",)).read_text()
    assert "Ceiling" in text and "unanimous" in text
