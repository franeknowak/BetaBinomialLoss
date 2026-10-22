"""Round D, D7 synthetic study (b2j/synth.py) on a tiny world."""
import numpy as np
import pytest

from b2j import posthoc as P
from b2j import synth as Y


def test_world_is_reproducible_and_beta_marginal_matches():
    s = Y.Setting("t", n=10, gamma=1.0)
    a, b = Y.draw_split(s, 0), Y.draw_split(s, 0)
    assert all(np.array_equal(a[2][k], b[2][k]) for k in ("z", "k", "theta"))
    assert not np.array_equal(Y.draw_split(s, 1)[2]["k"], a[2]["k"])
    _, _, ev = a
    nu, panels, oracle = Y.truths(s, ev, 0)
    assert np.allclose(nu, ev["mu"]) and set(panels) == set(Y.N_EVAL_PANELS)
    k = panels[10]                                   # beta world: the oracle is the exact Beta-Binomial
    assert np.allclose(oracle[10], P.vote_nll(ev["mu"], ev["S"], k, np.full_like(k, 10)))
    assert np.std(np.log(ev["S"])) == pytest.approx(1.0, abs=0.05)          # γ = 1: log S sd 1
    assert np.std(np.log(Y.sample(Y.Setting("t", gamma=0.0), 100, np.random.default_rng(0))["S"])) < 1e-12


def test_raters_world_poisson_binomial_and_panels():
    p = np.array([[0.2, 0.5, 0.9]])
    f = Y._pb(p)[0]
    assert f.sum() == pytest.approx(1) and f[0] == pytest.approx(0.8 * 0.5 * 0.1) and f[3] == pytest.approx(0.2 * 0.5 * 0.9)
    pan = Y._panels(np.random.default_rng(0), (500,), 3)          # distinct raters per panel, panels differ by image
    assert all(len(set(r)) == 3 for r in pan) and len({tuple(sorted(r)) for r in pan}) > 100
    assert len(Y._rater_bias()) == Y.RATER_POOL and abs(Y._rater_bias().mean()) < 1e-12
    s = Y.Setting("r", world="raters")
    x = Y.sample(s, 50, np.random.default_rng(0))
    assert (x["n"] == 3).all() and set(np.unique(x["k"])) <= {0, 1, 2, 3}
    nu, panels, oracle = Y.truths(s, x, 0)
    assert set(panels) == set(Y.RATER_PANELS) and all(np.isfinite(o).all() for o in oracle.values())
    assert panels[10].max() <= 10
    sf = Y.Setting("f", world="raters_fixed")
    x = Y.sample(sf, 50, np.random.default_rng(0))
    assert (x["n"] == 3).all() and set(Y.truths(sf, x, 0)[1]) == {3}


def test_variable_n_and_specs():
    s = Y.Setting("v", n_range=(4, 10))
    x = Y.sample(s, 500, np.random.default_rng(0))
    assert x["n"].min() == 4 and x["n"].max() == 10 and (x["k"] <= x["n"]).all()
    sp = Y.specs()
    assert len(sp) == len({(s.name, m, seed) for s, m, seed in sp})
    assert {m for s, m, _ in sp if s.weighted_baseline} >= {"soft_bce_n"}


def test_train_score_report_tiny(tmp_path, monkeypatch):
    monkeypatch.setattr(Y, "N_EVAL", 150)
    monkeypatch.setattr(Y, "N_DRAWS", 10)
    monkeypatch.setattr(Y, "SEEDS", range(1))
    tiny = [Y.Setting("S1_tiny", n=3, n_train=90), Y.Setting("S4_raters_tiny", world="raters", n_train=90)]
    monkeypatch.setattr(Y, "settings", lambda: tiny)
    for s in tiny:
        for m in Y.models(s):
            assert Y.train_one(s, m, 0, root=tmp_path) == "done"
    assert Y.train_one(tiny[0], "b2r", 0, root=tmp_path) == "skipped"
    rows = Y.score(tiny[0], 0, root=tmp_path)
    assert {r["model"] for r in rows} == {"soft_bce", "b2r", "ref200"} and len(rows) == 6
    assert all(np.isfinite(r["mse_nu"]) and np.isfinite(r["excess_nll_n50"]) for r in rows)
    text = Y.report(root=tmp_path).read_text()
    assert "S1_tiny" in text and "S4_raters_tiny" in text
