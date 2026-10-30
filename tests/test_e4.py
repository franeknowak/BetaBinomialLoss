"""Phase 2, E4: seed averaging, right/wrong combinations, end to end on fake outputs."""
import json

import numpy as np
import pandas as pd
import pytest

from b2j import e1, e4
from tests.test_e1 import fake_outputs as _fake_outputs


def fake_outputs(tmp_path):
    """As in test_e1, but every seed and loss sees the same frames and votes (true of the real outputs)."""
    _fake_outputs(tmp_path)
    files = sorted(tmp_path.glob("final*/toy/*/seed*/outputs_*.json"))
    ref = json.loads(files[0].read_text())["images"]
    for f in files:
        d = json.loads(f.read_text())
        for im, r in zip(d["images"], ref):
            im["k"], im["n"] = r["k"], r["n"]
        f.write_text(json.dumps(d))


def test_combo_names_cover_all_eight_cases():
    names = set()
    for r in np.ndindex(2, 2, 2):
        names.add(e4.combo_name(dict(zip(e4.MODELS, map(bool, r)))))
    assert names == set(e4.COMBOS) and len(names) == 8
    assert e4.combo_name({"hard_bce": False, "soft_bce": False, "b2_b1": True}) == "only b2_b1 right"
    assert e4.combo_name({"hard_bce": True, "soft_bce": False, "b2_b1": True}) == "only soft_bce wrong"


def test_averaged_score_is_the_mean_over_seeds(tmp_path):
    fake_outputs(tmp_path)
    avg, labels = e4.averaged_scores("toy", "b2_b1", "bacc", root=tmp_path)
    per_seed = []
    for s in e1.SEEDS:
        data, _ = e1.load_outputs("toy", "b2", s, "bacc", tmp_path)
        out, k, n = data["test"]
        per_seed.append(np.stack([e1.score("b2_b1", out[:, j], n[:, j]) for j in range(len(labels))], 1))
    np.testing.assert_allclose(avg["test"][1], np.mean(per_seed, 0), rtol=1e-12)


def test_end_to_end_on_fake_outputs(tmp_path):
    fake_outputs(tmp_path)
    e4.main("toy", root=tmp_path)
    res = json.loads((tmp_path / "phase2" / "toy" / "e4.json").read_text())["results"]
    fr = pd.read_csv(tmp_path / "phase2" / "toy" / "e4_frames.csv")
    assert len(fr) == 2 * 2 * 80                                                 # exps x labels x test frames
    for key, r in res.items():
        exp, lab = key.split("|")
        sub = fr[(fr.exp == exp) & (fr.label == lab)]
        assert r["combos"]["all"]["n"] == 80 == sum(r["combos"]["all"][c] for c in e4.COMBOS)
        assert sum(v["n"] for p, v in r["combos"].items() if p != "all") == 80
        for m, v in r["models"].items():
            t = v["test"]["all"]
            right = sub[f"{m}_right"].to_numpy()
            pos = sub.majority.to_numpy() == 1
            assert t["sensitivity"] == pytest.approx(right[pos].mean())
            assert ((sub[f"{m}_score"] >= v["cutoff"]).astype(int) == sub[f"{m}_call"]).all()
        assert r["same_call"]["hard_bce|soft_bce"] == pytest.approx((sub.hard_bce_call == sub.soft_bce_call).mean())
    assert {v["checkpoint"] for k, r in res.items() if k.startswith("E4b") for v in r["models"].values()} == \
        {"hardbce", "softbce", "nll"}
