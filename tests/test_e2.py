"""Phase 2, E2: disagreement scores, end to end on fake outputs."""
import json

import numpy as np
import pytest
from scipy import stats

from b2j import e2
from tests.test_e1 import fake_outputs


def test_p_disagree_matches_closed_forms():
    z = np.array([[-3.0], [0.0], [1.2], [40.0]])
    p, q = 1 / (1 + np.exp(-z[:, 0])), 1 / (1 + np.exp(z[:, 0]))                    # q = 1 - p without rounding
    np.testing.assert_allclose(e2.p_disagree("soft_bce", z, np.full(4, 3.0)), 3 * p * q, rtol=1e-12)
    assert e2.p_disagree("hard_bce", z, np.full(4, 3.0))[3] > 0                     # confident frame keeps a score
    a0, a1 = np.array([3.6, 36.0, 0.06, 5.0]), np.array([0.4, 4.0, 0.04, 5.0])
    S, m = a0 + a1, a1 / (a0 + a1)
    out = np.stack([a0, a1], 1)
    np.testing.assert_allclose(e2.p_disagree("b2", out, np.full(4, 3.0)), 3 * m * (1 - m) * S / (S + 1), rtol=1e-10)
    np.testing.assert_allclose(e2.p_disagree("b2", out[:2], np.array([7.0, 7.0])),
                               [1 - stats.betabinom.pmf(0, 7, b, a) - stats.betabinom.pmf(7, 7, b, a)
                                for a, b in zip(a0[:2], a1[:2])], rtol=1e-10)


def test_end_to_end_on_fake_outputs(tmp_path):
    fake_outputs(tmp_path)
    e2.main("toy", root=tmp_path)
    res = json.loads((tmp_path / "phase2" / "toy" / "e2.json").read_text())
    assert len(res["rows"]) == 2 * 3 * 5 * 2                                         # exps x methods x seeds x labels
    r = res["rows"][0]
    assert set(r["by_pattern"]) <= {"0/3", "1/3", "2/3", "3/3"}
    assert sum(v["n"] for v in r["by_pattern"].values()) == r["test"]["n"] == 80
    disputed = sum(v["n"] for p, v in r["by_pattern"].items() if p in ("1/3", "2/3"))
    assert r["test"]["n_pos"] == disputed
    flagged = sum(v["n"] * v["share_flagged"] for p, v in r["by_pattern"].items() if p in ("1/3", "2/3"))
    assert r["test"]["sensitivity"] == pytest.approx(flagged / disputed)
    assert {x["checkpoint"] for x in res["rows"] if x["exp"] == "E2b"} == {"hardbce", "softbce", "nll"}
    assert "by_pattern" in res["summary"]["E2a|b2|C1"] and "by_pattern" not in res["summary"]["E2a|b2|macro"]


def test_summary_with_label_specific_vote_patterns(tmp_path):
    fake_outputs(tmp_path)
    for f in tmp_path.glob("final*/toy/*/seed*/outputs_*.json"):            # label C2 gets 4 raters, C1 keeps 3
        d = json.loads(f.read_text())
        for im in d["images"]:
            im["n"] = [3, 4]
        f.write_text(json.dumps(d))
    e2.main("toy", root=tmp_path)
    res = json.loads((tmp_path / "phase2" / "toy" / "e2.json").read_text())
    assert all(p.endswith("/4") for p in res["summary"]["E2a|b2|C2"]["by_pattern"])
    assert all(p.endswith("/3") for p in res["summary"]["E2a|b2|C1"]["by_pattern"])
