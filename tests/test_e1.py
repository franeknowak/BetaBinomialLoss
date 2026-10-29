"""Phase 2, E1: cut-off search, scores, end to end on fake outputs."""
import json

import numpy as np
import pytest
from scipy import integrate, stats

from b2j import e1


def _brute_best(s, y):
    """Every cut-off on a fine grid; the set of best BACC values must match the exact search."""
    grid = np.linspace(s.min() - 1, s.max() + 1, 20001)
    return max(e1.bacc_at(s, y, t) for t in grid)


def test_fit_cutoff_toy_example_from_the_plan_discussion():
    s = np.array([0.05, 0.12, 0.30, 0.41, 0.66, 0.80])
    y = np.array([0, 0, 1, 0, 1, 1])
    t, b, n_tied = e1.fit_cutoff(s, y)
    assert b == pytest.approx(5 / 6) and n_tied == 2
    assert t == pytest.approx(0.21)                                     # two tied: the lower of the two middle ones


def test_fit_cutoff_is_exact_and_middle_of_ties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = np.round(rng.random(60), 2)                                 # rounding makes repeated scores
        y = (rng.random(60) < s).astype(int)
        t, b, n_tied = e1.fit_cutoff(s, y)
        assert b == pytest.approx(_brute_best(s, y)) == pytest.approx(e1.bacc_at(s, y, t))
    s, y = np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 0, 1, 1])
    assert e1.fit_cutoff(s, y)[0] == pytest.approx(0.25)
    t, b, n = e1.fit_cutoff(np.array([0.2, 0.2, 0.7]), np.array([0, 1, 1]))   # repeated score, >= is positive
    assert e1.bacc_at(np.array([0.2, 0.2, 0.7]), np.array([0, 1, 1]), t) == b


def test_scores_match_the_definitions():
    a0, a1, n = np.array([6.0, 0.6, 60.0]), np.array([4.0, 0.4, 40.0]), np.array([3.0, 3.0, 7.0])
    out = np.stack([a0, a1], 1)
    np.testing.assert_allclose(e1.score("b2_b0", out, n), 0.4)
    g = lambda t: 3 * t ** 2 - 2 * t ** 3                                # P(at least 2 of 3 | theta)
    b1 = e1.score("b2_b1", out, n)
    for i in range(2):
        f = lambda t: g(t) * stats.beta.pdf(t, a1[i], a0[i])
        assert b1[i] == pytest.approx(integrate.quad(f, 0, 1, points=[0.5], limit=200)[0], rel=1e-6)
    assert b1[0] == pytest.approx(0.3636, abs=1e-4)                      # worked example: Beta(4, 6)
    assert b1[2] == pytest.approx(stats.betabinom.pmf([4, 5, 6, 7], 7, 40, 60).sum())   # majority of 7 = k >= 4
    np.testing.assert_allclose(e1.score("b2_b2", out, n), [stats.beta.sf(0.5, a, b) for a, b in zip(a1, a0)])
    assert e1.score("b2_b2", out, n)[0] == pytest.approx(0.254, abs=1e-3)
    np.testing.assert_allclose(e1.score("soft_bce", np.array([[0.0], [2.0]]), n[:2]), [0.5, 1 / (1 + np.exp(-2))])


def fake_outputs(tmp_path):
    """Fake outputs_<ckpt>.json for every loss, seed and checkpoint: 2 labels, 80 val + 80 test frames, n = 3."""
    rng = np.random.default_rng(1)
    labels = ["C1", "C2"]
    for loss, ckpts in {"hard_bce": ("bacc", "hardbce"), "soft_bce": ("bacc", "softbce"), "b2": ("bacc", "nll")}.items():
        for seed in e1.SEEDS:
            d = tmp_path / e1.RUN_DIR[loss] / "toy" / loss / f"seed{seed}"
            d.mkdir(parents=True)
            for c in ckpts:
                images = []
                for split in ("val", "test"):
                    for i in range(80):
                        k = [int(x) for x in rng.integers(0, 4, 2)]
                        out = [[float(rng.normal() + kk - 1.5)] if loss != "b2" else
                               [float(rng.uniform(0.1, 5)), float(rng.uniform(0.1, 5) + kk)] for kk in k]
                        images.append({"id": f"{split}{i}", "split": split, "out": out, "k": k, "n": [3, 3]})
                (d / f"outputs_{c}.json").write_text(json.dumps({"labels": labels, "images": images}))


def test_end_to_end_on_fake_outputs(tmp_path):
    fake_outputs(tmp_path)
    e1.main("toy", root=tmp_path)
    res = json.loads((tmp_path / "phase2" / "toy" / "e1.json").read_text())
    assert len(res["rows"]) == 2 * 5 * 5 * 2                            # exps x methods x seeds x labels
    r = res["rows"][0]
    assert r["test"]["all"]["n"] == 80 == r["test"]["unanimous"]["n"] + r["test"]["disputed"]["n"]
    assert {x["checkpoint"] for x in res["rows"] if x["exp"] == "E1b"} == {"hardbce", "softbce", "nll"}
    m = res["summary"]["E1a|b2_b2|all|macro"]["bacc"][0]
    per = [np.mean([x["test"]["all"]["bacc"] for x in res["rows"] if x["exp"] == "E1a" and x["method"] == "b2_b2"
                    and x["seed"] == s]) for s in e1.SEEDS]
    assert m == pytest.approx(np.mean(per))
