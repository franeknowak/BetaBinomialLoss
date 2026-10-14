"""Unit tests for module g (choice of head lr / wd, margins). Fake run outputs; no training.

Run: python -m pytest tests/test_module_g.py -q
"""
import json

import numpy as np
import pytest

from b2j import pilot as G
from b2j.data import DATA_ROOT, load_table
from b2j.runs import RunSpec


def _summary(map_, nll, a_ep, b_ep, epochs=15):
    return {"config": {"epochs": epochs}, "A": {"val_map": map_, "epoch": a_ep}, "B": {"val_nll_per_vote": nll, "epoch": b_ep}}


def test_choose_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "DATASETS", ("endoscapes",))
    monkeypatch.setattr(G, "FAMILIES", ("b2r",))
    monkeypatch.setattr(G, "GRID", [(1e-4, 1e-4), (3e-4, 1e-2), (1e-3, 1e-2)])
    vals = {(1e-4, 1e-4): (0.5400, 0.30, 3, 4), (3e-4, 1e-2): (0.5395, 0.20, 2, 14), (1e-3, 1e-2): (0.5300, 0.10, 2, 3)}
    for (lr, wd), v in vals.items():
        d = RunSpec("endoscapes", "b2r", lr=lr, wd=wd, block="tune").out_dir(tmp_path)
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps(_summary(*v)))
    chosen, late = G.choose(tmp_path)
    c = chosen["endoscapes"]["b2r"]
    assert (c["lr"], c["wd"]) == (3e-4, 1e-2)          # within 1e-3 of the best mAP, lower NLL wins; 0.53 is out
    assert late == ["endoscapes/b2r"]                   # B at epoch 14 > 80% of 15
    assert G.hyper("endoscapes", "b2r_kl", tmp_path) == {"lr": 3e-4, "wd": 1e-2}   # inherited by the family
    assert (tmp_path / "tune" / "tune_report.md").exists()


@pytest.mark.skipif(not (DATA_ROOT / "cifar10h" / "cifar10h-counts.npy").exists()
                    or not (DATA_ROOT / "endoscapes" / "all_metadata.csv").exists(), reason="labels not present")
def test_margins_on_fake_pilot(tmp_path, monkeypatch):
    chosen = {ds: {f: {"lr": 3e-4, "wd": 1e-2} for f in G.FAMILIES} for ds in G.DATASETS}
    (tmp_path / "tune").mkdir()
    G.chosen_path(tmp_path).write_text(json.dumps(chosen))
    rng = np.random.default_rng(0)
    for ds in G.PILOT_DATASETS:
        val = load_table(ds).take("val")
        f = val.k / val.n
        for spec in G.pilot_specs(tmp_path):
            if spec.dataset != ds:
                continue
            d = spec.out_dir(tmp_path)
            d.mkdir(parents=True)
            z = np.log((f * 0.8 + 0.1) / (1 - (f * 0.8 + 0.1))) + rng.normal(0, 1, f.shape)   # informative logits
            out = z[..., None] if spec.variant == "soft_bce" else np.stack([-z / 2 + 1, z / 2 + 1], -1)
            for c in "AB":
                np.savez(d / f"val_{c}.npz", ids=val.ids, out=out,
                         scale=np.full(f.shape[1], 3.0 if spec.variant == "soft_bce" else 1.0))
    res = G.margins(tmp_path, B=50)
    for ds in G.PILOT_DATASETS:
        assert set(res[ds]) == {"bacc", "map", "nll", "fcr", "ap"}
        for r in res[ds].values():
            assert r["lo"] <= r["hi"] and r["h"] >= 0
            assert r["delta"] is None or r["delta"] == pytest.approx(2 * r["h"])
    assert (tmp_path / "pilot" / "margins.md").exists()
