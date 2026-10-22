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
    nu, panels, oracle, _ = Y.truths(s, ev, 0)
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
    nu, panels, oracle, ordered = Y.truths(s, x, 0)
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
    monkeypatch.setattr(Y, "N_DRAWS_POOL", 50)
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
    assert all(np.isfinite(r["d_nll_spread_n3"]) for r in rows if r["model"] == "b2r")
    rows = Y.score(tiny[1], 0, root=tmp_path)
    assert all(np.isfinite(r["d_nll_spread_n3"]) for r in rows if r["model"] in ("b2r", "b2rba"))
    text = Y.report(root=tmp_path).read_text()
    assert "S1_tiny" in text and "S4_raters_tiny" in text and "paired B²-rba − B²-r" in text


def test_logitnormal_matches_beta_mean_and_variance():
    """The fixed logit-normal world (2026-10-22): E θ = μ and Var θ = μ(1 − μ)/(S + 1) at every image, incl. extreme
    μ; so ν = μ and near-certain images stay near-certain (the first version put θ > 0.5 on a third of μ < 0.05)."""
    rng = np.random.default_rng(0)
    mu = np.concatenate([[1e-6, 1e-4, 0.01, 0.5, 0.99, 1 - 1e-6], rng.uniform(0, 1, 200)])
    S = np.concatenate([[0.2, 5, 50, 1, 5, 300], np.exp(rng.normal(np.log(5), 1, 200))])
    th = Y._theta_draws(Y.Setting("ln", world="logitnormal"), mu, S, rng, 40000)
    sel = (mu > 1e-3) & (mu < 1 - 1e-3)
    np.testing.assert_allclose(th.mean(1)[sel], mu[sel], atol=4 * np.sqrt(mu * (1 - mu) / (S + 1) / 40000)[sel].max())
    rho = th.var(1) / (mu * (1 - mu))
    mid = (mu > 0.05) & (mu < 0.95)                                  # (sample variance is noisy for skewed θ)
    np.testing.assert_allclose(rho[mid], 1 / (S[mid] + 1), rtol=0.05, atol=0.01)
    c, sig = Y.ln_params(mu, S)
    a, b = Y._ln_moments(c, np.log(sig))
    assert np.abs(a - Y.logit(mu)).max() < 1e-4 and np.abs(b + np.log(S)).max() < 1e-4
    x = Y.sample(Y.Setting("ln", world="logitnormal"), 20000, np.random.default_rng(1))
    lo = x["mu"] < 0.05
    assert (x["theta"][lo] > 0.5).mean() < 0.02


def test_oracles_match_monte_carlo():
    """Quadrature oracles (logitnormal counts; fixed-panel counts and ordered votes) agree with a large Monte Carlo."""
    from scipy.special import logsumexp
    from scipy.stats import binom
    rng = np.random.default_rng(3)
    s = Y.Setting("ln", world="logitnormal")
    ev = Y.sample(s, 40, rng)
    _, panels, oracle, _ = Y.truths(s, ev, 0)
    th = Y._theta_draws(s, ev["mu"], ev["S"], rng, 200000)
    for ne in (3, 50):
        mc = -(logsumexp(binom.logpmf(panels[ne][:, None], ne, th), 1) - np.log(th.shape[1]))
        np.testing.assert_allclose(oracle[ne], mc, atol=0.03 if ne == 50 else 0.01)
    sf = Y.Setting("f", world="raters_fixed")
    ev = Y.sample(sf, 40, rng)
    _, panels, oracle, ordered = Y.truths(sf, ev, 0)
    th = Y._theta_draws(sf, ev["mu"], ev["S"], rng, 200000)
    p = Y._rater_p(th, fixed=True)                                   # [N, draws, 3]
    y = ordered[3][0][..., 0]
    lp = (y[:, None] * np.log(p) + (1 - y[:, None]) * np.log1p(-p)).sum(-1)
    np.testing.assert_allclose(ordered[3][1], -(logsumexp(lp, 1) - np.log(lp.shape[1])), atol=0.01)
    pmf = Y._pb(p).mean(1)
    np.testing.assert_allclose(oracle[3], -np.log(np.take_along_axis(pmf, panels[3].astype(int)[:, None], 1)[:, 0]),
                               atol=0.01)


def test_spread_ablation_sign():
    """Per-image S equal to the truth beats one S on Beta-Binomial panels (> 0); a constant S cannot (≤ 0)."""
    rng = np.random.default_rng(0)
    m = rng.uniform(0.2, 0.8, 20000)
    S = np.where(rng.random(20000) < 0.5, 1.0, 100.0)
    k = rng.binomial(3, rng.beta(m * S, (1 - m) * S)).astype(float)
    assert Y.spread_ablation(m, S, k=k) > 0.02
    assert Y.spread_ablation(m, np.full_like(m, 5.0), k=k) <= 1e-9     # the fitted single S is at least as good
