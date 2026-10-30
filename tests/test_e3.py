"""Phase 2, E3: risks, coverage bookkeeping, end to end on fake outputs."""
import json

import numpy as np
import pytest
from scipy import stats

from b2j import e1, e3
from tests.test_e1 import fake_outputs


def test_scaled_distance_example_from_the_plan_discussion():
    p = np.array([0.05, 0.55, 0.08, 0.12])
    z = np.log(p / (1 - p))[:, None]
    r, yes = e3.risk("hard_bce", z, np.full(4, 3.0), 0.10)
    np.testing.assert_allclose(1 - r, [0.5, 0.5, 0.2, 0.02 / 0.9], rtol=1e-9)
    assert list(yes) == [False, True, False, True]


def test_majority_maps_and_decisions_match_e1():
    z = np.array([[-3.0], [-1.0], [0.3], [2.0], [40.0]])
    n = np.full(5, 3.0)
    p = 1 / (1 + np.exp(-z[:, 0]))
    q, qc, cq = e3.q_and_cut("soft_bce", z, n, 0.2)
    np.testing.assert_allclose(q, 3 * p ** 2 - 2 * p ** 3, rtol=1e-10)
    np.testing.assert_allclose(q + qc, 1, rtol=1e-12)
    assert cq[0] == pytest.approx(3 * 0.04 - 2 * 0.008)
    assert qc[4] > 0                                                         # confident frame keeps a distinct risk
    a0, a1 = np.array([6.0, 0.6, 60.0, 0.3]), np.array([4.0, 0.4, 40.0, 3.0])
    out = np.stack([a0, a1], 1)
    m = a1 / (a0 + a1)
    q, _, _ = e3.q_and_cut("b2_b0", out, np.full(4, 3.0), 0.3)
    np.testing.assert_allclose(q, 3 * m ** 2 - 2 * m ** 3, rtol=1e-10)
    for method in ("hard_bce", "soft_bce"):
        _, yes = e3.risk(method, z, n, 0.2)
        assert (yes == (e1.score(method, z, n) >= 0.2)).all()
    for method in ("b2_b0", "b2_b1", "b2_b2", "b2_b3"):
        _, yes = e3.risk(method, out, np.full(4, 3.0), 0.3)
        assert (yes == (e1.score(e3.E1_SCORE.get(method, method), out, np.full(4, 3.0)) >= 0.3)).all()


def test_b3_is_beta_mass_on_the_other_side():
    out = np.array([[85.0, 15.0], [17.0, 3.0], [4.25, 0.75], [9.0, 0.5]])  # means .15, .15, .15, .053
    r, yes = e3.risk("b2_b3", out, np.full(4, 3.0), 0.10)
    np.testing.assert_allclose(r[:3], [0.068, 0.295, 0.488], atol=1e-3)
    assert list(yes) == [True, True, True, False]
    assert r[3] == pytest.approx(stats.beta.sf(0.10, 0.5, 9.0))


def test_coverage_keeps_lowest_risk_and_counts_ties():
    r = np.array([0.9, 0.1, 0.5, 0.5, 0.2])
    yes = np.array([True, False, True, False, True])
    y = np.array([0, 0, 1, 0, 1])
    disputed = np.array([True, False, False, True, False])
    v = e3.at_coverage(r, yes, y, disputed, 0.6)                             # keeps 3: frames 1, 4, 2
    assert v["n_kept"] == 3 and v["n_pos_kept"] == 2 and v["n_tied_at_boundary"] == 2
    assert v["sensitivity"] == 1 and v["specificity"] == 1
    assert v["disputed_share_aside"] == 1.0                                  # aside: frames 3 and 0


def test_end_to_end_full_coverage_equals_e1(tmp_path):
    fake_outputs(tmp_path)
    e1.main("toy", root=tmp_path)
    e3.main("toy", root=tmp_path)
    r1 = json.loads((tmp_path / "phase2" / "toy" / "e1.json").read_text())["rows"]
    r3 = json.loads((tmp_path / "phase2" / "toy" / "e3.json").read_text())["rows"]
    assert len(r3) == 2 * 6 * 5 * 2                                          # exps x methods x seeds x labels
    key = lambda x, m: (x["exp"][-1], m, x["seed"], x["label"])
    e1_bacc = {key(x, x["method"]): x["test"]["all"]["bacc"] for x in r1}
    for x in r3:
        full = x["coverage"]["1.00"]
        assert full["bacc"] == pytest.approx(e1_bacc[key(x, e3.E1_SCORE.get(x["method"], x["method"]))])
        assert full["n_pos_kept"] == x["n_pos"] and full["n_kept"] == 80
        assert x["coverage"]["0.50"]["n_kept"] == 40
