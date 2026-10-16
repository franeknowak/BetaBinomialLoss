"""Unit tests for module d (post-hoc: panel probabilities, ensemble, thresholds; S*(m) dropped 2026-10-16).

Run: python -m pytest tests/test_module_d.py -q
"""
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import betabinom

from b2j import posthoc as P


# ---------------------------------------------------------------- pmf and panel probabilities
def test_vote_pmf_matches_scipy_and_varies_with_n():
    m = np.array([[0.2], [0.7], [0.5]])
    S = np.array([[1.5], [4.0], [np.inf]])
    n = np.array([[3], [5], [4]])
    pmf = P.vote_pmf(m, S, n)
    assert pmf.shape == (3, 1, 6)
    np.testing.assert_allclose(pmf.sum(-1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(pmf[0, 0, :4], betabinom.pmf(np.arange(4), 3, 0.3, 1.2), rtol=1e-10)
    np.testing.assert_allclose(pmf[1, 0], betabinom.pmf(np.arange(6), 5, 2.8, 1.2), rtol=1e-10)
    assert (pmf[0, 0, 4:] == 0).all() and pmf[2, 0, 5] == 0


@pytest.mark.parametrize("t, row", [(0.15, [0.156, 0.165, 0.182, 0.207, 0.244]),
                                    (0.20, [0.206, 0.216, 0.232, 0.255, 0.287]),
                                    (0.30, [0.306, 0.313, 0.326, 0.342, 0.363])])
def test_plan_boundary_table(t, row):
    """Plan §3 point 3: the m at which P_maj(n = 3) = t, for S = 0.5 / 1 / 2 / 5 / inf."""
    for S, expected in zip([0.5, 1, 2, 5, np.inf], row):
        f = lambda m: P.p_majority(P.vote_pmf(np.array([[m]]), np.array([[S]]), np.array([[3]])), 3)[0, 0] - t
        assert brentq(f, 1e-6, 0.5) == pytest.approx(expected, abs=1.5e-3)


def test_plan_pmaj_ranges():
    """Plan §3 point 4: at m = 0.3, P_maj over S in [1, inf) is 0.216-0.286 (n = 3) and 0.001-0.273 (n = 51)."""
    def pm(S, n):
        return P.p_majority(P.vote_pmf(np.array([[0.3]]), np.array([[S]]), np.array([[n]])), n)[0, 0]
    assert sorted([pm(1, 3), pm(np.inf, 3)]) == pytest.approx([0.216, 0.286], abs=1e-3)
    assert sorted([pm(1, 51), pm(np.inf, 51)]) == pytest.approx([0.001, 0.273], abs=1e-3)


def test_pmaj_symmetry_and_ties():
    """Plan §3 point 1: P_maj > 0.5 iff m > 0.5 for every S (odd n). Even n: a tie is not a yes-majority."""
    m = np.linspace(0.05, 0.95, 19)[:, None]
    for S in (0.3, 2.0, 50.0, np.inf):
        pm = P.p_majority(P.vote_pmf(m, np.full_like(m, S), np.full_like(m, 3)), 3)[:, 0]
        assert ((pm > 0.5) == (m[:, 0] > 0.5)).all()
    pmf = P.vote_pmf(np.array([[0.5]]), np.array([[np.inf]]), np.array([[2]]))
    assert P.p_majority(pmf, 2)[0, 0] == pytest.approx(0.25)


def test_split_and_unanimous():
    m = np.random.default_rng(0).uniform(0.01, 0.99, (50, 2))
    S = np.random.default_rng(1).uniform(0.1, 30, (50, 2))
    S[:5] = np.inf
    pmf3 = P.vote_pmf(m, S, np.full_like(m, 3))
    np.testing.assert_allclose(P.p_split3(m, S), 1 - P.p_unanimous(pmf3, np.full(m.shape, 3)), rtol=1e-10)
    np.testing.assert_allclose(P.agree_with_decision(np.array([0.8, 0.8]), [1, 0]), [0.8, 0.2])


def test_mixture():
    a, b = (np.full((4, 1), 0.2), np.full((4, 1), 2.0)), (np.full((4, 1), 0.6), np.full((4, 1), np.inf))
    n = np.full((4, 1), 3)
    np.testing.assert_allclose(P.mixture_pmf([a], n), P.vote_pmf(*a, n))
    np.testing.assert_allclose(P.mixture_pmf([a, b], n), (P.vote_pmf(*a, n) + P.vote_pmf(*b, n)) / 2)
    np.testing.assert_allclose(P.mixture_p_split3([a, b]), (P.p_split3(*a) + P.p_split3(*b)) / 2)


# ---------------------------------------------------------------- thresholds
def test_threshold_max_bacc_brute_force():
    rng = np.random.default_rng(0)
    y = rng.random((300, 3)) < [0.1, 0.3, 0.5]
    score = np.round(rng.normal(y * 1.0, 1.0), 1)                     # ties on purpose
    t = P.threshold_max_bacc(score, y)

    def bacc(s, yy, th):
        p = s >= th
        return 0.5 * (p[yy].mean() + (~p[~yy]).mean())
    for j in range(3):
        best = max(bacc(score[:, j], y[:, j], c) for c in np.unique(score[:, j]))
        assert bacc(score[:, j], y[:, j], t[j]) == pytest.approx(best)
    assert P.threshold_max_bacc(np.zeros((5, 1)), np.zeros((5, 1)))[0] == 0.5


def test_threshold_at_recall():
    rng = np.random.default_rng(1)
    y = rng.random((500, 2)) < 0.3
    score = rng.normal(y * 1.5, 1.0)
    t = P.threshold_at_recall(score, y, 0.9)
    for j in range(2):
        pos = score[y[:, j], j]
        assert (pos >= t[j]).mean() >= 0.9
        assert (pos > t[j]).mean() < 0.9                               # the largest such threshold

