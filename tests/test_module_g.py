"""Unit tests for module g (choice of head lr / wd; the margins step was dropped 2026-10-16). Fake run outputs.

Run: python -m pytest tests/test_module_g.py -q
"""
import json

from b2j import pilot as G
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



def test_no_margins_command():
    assert not hasattr(G, "margins")
    assert len(G.pilot_specs.__code__.co_varnames) >= 1        # the pilot block itself stays (validation preview)
