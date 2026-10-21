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
    monkeypatch.setattr(G, "DATASETS", ("endoscapes", "amibr"))        # amibr not tuned yet -> skipped
    chosen = G.choose(tmp_path)
    assert set(chosen) == {"endoscapes"}
    c = chosen["endoscapes"]["b2r"]
    assert (c["lr"], c["wd"]) == (3e-4, 1e-2)          # within 1e-3 of the best mAP, lower NLL wins; 0.53 is out
    assert "late" not in c                              # the 80%-budget flag was dropped (2026-10-16)
    assert G.hyper("endoscapes", "b2r_kl", tmp_path) == {"lr": 3e-4, "wd": 1e-2}   # inherited by the family
    assert (tmp_path / "tune" / "tune_report.md").exists()
    log = [{"epoch": e, "val_map": 0.5, "val_nll_per_vote": 0.3, "train_map": 0.6, "train_nll_per_vote": 0.2}
           for e in (0.25, 0.5, 0.75)]
    (RunSpec("endoscapes", "b2r", lr=3e-4, wd=1e-2, block="tune").out_dir(tmp_path) / "log.jsonl").write_text(
        "\n".join(map(json.dumps, log)))
    assert all(f.exists() for f in G.curves(tmp_path))


def test_r1_block(tmp_path):
    from b2j.blocks import r1
    (tmp_path / "tune").mkdir()
    G.chosen_path(tmp_path).write_text(json.dumps({ds: {f: {"lr": 1e-3, "wd": 1e-4} for f in G.FAMILIES} for ds in G.DATASETS}))
    specs = r1(tmp_path)
    assert len(specs) == 135 + 10 + 55 + 40 + 40 == len({(s.dataset, s.name) for s in specs})
    assert all(s.regime == "frozen" and s.block == "r1" and s.lr == 1e-3 for s in specs)
    assert sum(s.train_frac < 1 for s in specs) == 40 and sum(s.val_full_panel for s in specs) == 10


def test_d_base_block(tmp_path):
    from b2j.blocks import d_base, r1
    (tmp_path / "tune").mkdir()
    G.chosen_path(tmp_path).write_text(json.dumps({ds: {f: {"lr": 1e-3, "wd": 1e-4} for f in G.FAMILIES} for ds in G.DATASETS}))
    specs = d_base(tmp_path)
    assert len(specs) == 90 == len({(s.dataset, s.name) for s in specs}) and {s.dataset for s in specs} == {"amibr", "ferplus"}
    assert "amibr" not in {s.dataset for s in r1(tmp_path)}              # R1 stays pinned


def test_d12_block(tmp_path):
    from b2j.blocks import d12
    (tmp_path / "tune").mkdir()
    G.chosen_path(tmp_path).write_text(json.dumps({ds: {"hard_bce": {"lr": 1e-3, "wd": 1e-4}, "soft_bce": {"lr": 1e-4, "wd": 1e-4},
                                                        "b2r": {"lr": 3e-4, "wd": 1e-2}} for ds in G.DATASETS}))
    specs = d12(tmp_path)
    assert len(specs) == 5 * 6 * 5 == len({(s.dataset, s.name) for s in specs})
    assert {s.variant for s in specs} == {"soft_bce", "b2r", "b2ms", "b2ms_var0.03", "b2ms_var0.3", "b2ms_var3"}
    assert all((s.lr, s.wd) == ((1e-4, 1e-4) if s.variant == "soft_bce" else (3e-4, 1e-2)) for s in specs)


def test_d5_block(tmp_path):
    from b2j.blocks import d5
    (tmp_path / "tune").mkdir()
    G.chosen_path(tmp_path).write_text(json.dumps({ds: {"hard_bce": {"lr": 1e-3, "wd": 1e-4}, "soft_bce": {"lr": 1e-4, "wd": 1e-4},
                                                        "b2r": {"lr": 3e-4, "wd": 1e-2}} for ds in G.DATASETS}))
    specs = d5(tmp_path)
    assert len(specs) == 2 * 3 * 5 == len({(s.dataset, s.name) for s in specs})
    assert {s.dataset for s in specs} == {"ferplus", "cifar10h"}
    assert all((s.lr, s.wd) == ((3e-4, 1e-2) if s.variant == "b2r" else (1e-4, 1e-4)) for s in specs)


def test_d6_block(tmp_path):
    from b2j.blocks import d6
    (tmp_path / "tune").mkdir()
    G.chosen_path(tmp_path).write_text(json.dumps({ds: {"hard_bce": {"lr": 1e-3, "wd": 1e-4}, "soft_bce": {"lr": 1e-4, "wd": 1e-4},
                                                        "b2r": {"lr": 3e-4, "wd": 1e-2}} for ds in G.DATASETS}))
    specs = d6(tmp_path)
    assert len(specs) == 2 * 5 * 5 == len({(s.dataset, s.name) for s in specs})
    assert all((s.lr, s.wd) == ((1e-4, 1e-4) if s.variant in ("soft_bce", "soft_ce") else (3e-4, 1e-2)) for s in specs)


def test_no_margins_command():
    assert not hasattr(G, "margins")
    assert len(G.pilot_specs.__code__.co_varnames) >= 1        # the pilot block itself stays (validation preview)
