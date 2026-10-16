"""Unit tests for module e (metrics, two-level bootstrap, readings; no margins since 2026-10-16).

Run: python -m pytest tests/test_module_e.py -q
"""
import numpy as np
import pytest
import torch
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             precision_recall_curve, roc_auc_score)

from b2j import losses
from b2j import metrics as M
from b2j import stats as St


def _data(N=400, K=2, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.random((N, K)) < 0.3
    score = 1 / (1 + np.exp(-rng.normal(1.5 * y - 0.5, 1.0)))
    return score, y


# ---------------------------------------------------------------- metrics
def test_discrimination_matches_sklearn():
    score, y = _data()
    t = np.array([0.4, 0.55])
    d = M.discrimination(score, y, t)
    pred = score >= t
    assert d["bacc"] == pytest.approx(np.mean([balanced_accuracy_score(y[:, j], pred[:, j]) for j in range(2)]))
    assert d["f1"] == pytest.approx(np.mean([f1_score(y[:, j], pred[:, j]) for j in range(2)]))
    assert d["map"] == pytest.approx(np.mean([average_precision_score(y[:, j], score[:, j]) for j in range(2)]))
    assert d["auroc"] == pytest.approx(np.mean([roc_auc_score(y[:, j], score[:, j]) for j in range(2)]))
    assert d["acc"] == pytest.approx((pred == y).mean())


def test_soft_targets_weighted():
    score, y = _data()
    assert M.auroc(score[:, 0], y[:, 0].astype(float)) == pytest.approx(roc_auc_score(y[:, 0], score[:, 0]))
    assert M.ap(score[:, 0], y[:, 0].astype(float)) == pytest.approx(average_precision_score(y[:, 0], score[:, 0]))
    # a soft target of 0.5 everywhere carries no ranking information
    assert M.auroc(score[:, 0], np.full(len(score), 0.5)) == pytest.approx(0.5)


def test_split_target_matches_losses():
    k = np.array([[0.0], [1], [2], [3], [10], [25]])
    n = np.array([[3.0], [3], [3], [3], [51], [51]])
    np.testing.assert_allclose(M.split_target(k, n), losses.split_target(torch.tensor(k), torch.tensor(n)).numpy())
    np.testing.assert_array_equal(M.split_target(k[:4], n[:4])[:, 0], [0, 1, 1, 0])


def test_fcr_at_recall_matches_pr_curve():
    score, y = _data(N=1000)
    for j in range(2):
        prec, rec, _ = precision_recall_curve(y[:, j], score[:, j])
        i = np.flatnonzero(rec >= 0.9)[-1]                   # sklearn lists recall in decreasing order
        assert M.fcr_at_recall(score[:, j:j + 1], y[:, j:j + 1]) == pytest.approx(1 - prec[i])
    # hand case with ties: positives at scores 0.9, 0.8, 0.8; negatives 0.8, 0.1 -> recall 1 first at t = 0.8
    s = np.array([[0.9], [0.8], [0.8], [0.8], [0.1]])
    t = np.array([[1], [1], [1], [0], [0]], bool)
    assert M.fcr_at_recall(s, t) == pytest.approx(0.25)
    r = M.fcr_at_threshold(s, t, [0.85])
    assert r["fcr"] == 0 and r["recall"] == pytest.approx(1 / 3)


def test_aurc_and_coverage():
    p = np.array([[0.99], [0.9], [0.6], [0.45]])
    y = np.array([[1], [1], [0], [1]], bool)                   # errors: 0.6 (pred 1, y 0), 0.45 (pred 0, y 1)
    # order by confidence: 0.99 ok, 0.9 ok, 0.6 err, 0.45 err -> selective risks 0, 0, 1/3, 2/4
    assert M.aurc(p, y) == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)
    ps = np.array([[0.1], [0.2], [0.3], [0.9], [0.8]])
    tg = np.array([[0.0], [0.0], [1.0], [1.0], [1.0]])
    assert M.split_rate_at_coverage(ps, tg, 0.6) == pytest.approx(1 / 3)
    tr = M.triage(ps, tg, top=0.4)
    assert tr["catch"] == pytest.approx(2 / 3) and tr["auroc"] == pytest.approx(1.0)


def test_calibration_metrics():
    rng = np.random.default_rng(0)
    p = rng.random((200000, 1))
    y = rng.random(p.shape) < p
    assert M.ece(p, y) < 0.01
    assert M.ece(np.full_like(p, 0.9), y) == pytest.approx(0.4, abs=0.01)
    assert M.brier(p, y) == pytest.approx(1 / 6, abs=0.005)
    rel = M.split_reliability(p, y.astype(float))
    assert rel.shape == (1, 10, 3)
    np.testing.assert_allclose(rel[0, :, 0], rel[0, :, 1], atol=0.01)
    tb = y[np.argsort(p[:, 0], kind="stable")[:20000], 0].astype(float)
    assert rel[0, 0, 2] == pytest.approx(tb.std(ddof=1) / np.sqrt(len(tb)), rel=1e-6)   # jackknife SE of a mean


def test_vote_nll_and_theta():
    m, S = np.full((3, 1), 0.5), np.full((3, 1), np.inf)
    k, n = np.array([[0.0], [1], [3]]), np.full((3, 1), 3.0)
    v = M.vote_nll(m, S, k, n)
    expected = -np.log([1 / 8, 3 / 8, 1 / 8]).mean()
    assert v["nll"] == pytest.approx(expected) and v["nll_per_vote"] == pytest.approx(expected / 3)
    assert M.theta_mse(np.full((2, 1), 0.5), np.array([[1.0], [2]]), np.full((2, 1), 4.0)) == pytest.approx(1 / 32)   # errors 0.25² and 0


# ---------------------------------------------------------------- bootstrap
def _mean_metric(p, rows):
    return p[rows].mean()


def test_bootstrap_identical_models_and_clusters():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(20), 10)
    x = rng.normal(size=200)
    r = St.two_level_bootstrap(_mean_metric, {"a": [x], "b": [x]}, lambda v: v["a"] - v["b"], clusters, B=200)
    assert r.estimate == 0 and (r.samples == 0).all()
    with pytest.raises(ValueError):
        St.cluster_indices(clusters, strata=np.arange(200) % 2)        # cluster spans two strata


def test_bootstrap_coverage_and_strata():
    """95% CI of a mean difference covers the truth in ~95% of simulated studies (shared clusters, 5 seeds)."""
    covered = 0
    for rep in range(60):
        rng = np.random.default_rng(rep)
        clusters = np.repeat(np.arange(40), 5)
        u = np.repeat(rng.normal(size=40), 5)                          # cluster effect shared by both models
        a = [u + 0.2 + rng.normal(0, 0.3, 200) for _ in range(5)]
        b = [u + rng.normal(0, 0.3, 200) for _ in range(5)]
        r = St.two_level_bootstrap(_mean_metric, {"a": a, "b": b}, lambda v: v["a"] - v["b"], clusters, B=300, seed=rep)
        covered += r.lo <= 0.2 <= r.hi
    assert covered >= 51                                                # >= 85% of 60 (nominal 95%)
    strata = np.repeat(np.arange(2), 100)
    rows, groups = St.cluster_indices(clusters, strata)
    assert [len(g) for g in groups] == [20, 20]


def test_paired_groups_narrow_within_model_contrasts():
    rng = np.random.default_rng(0)
    clusters = np.arange(300)
    runs = [rng.normal(s, 1.0, 300) for s in np.linspace(-2, 2, 5)]     # large seed-to-seed shifts
    models = {"pmaj": runs, "m": [r - 0.1 for r in runs]}              # two scores of the same runs
    comb = lambda v: v["pmaj"] - v["m"]
    paired = St.two_level_bootstrap(_mean_metric, models, comb, clusters, paired=[("pmaj", "m")], B=300)
    unpaired = St.two_level_bootstrap(_mean_metric, models, comb, clusters, B=300)
    assert paired.half_width < 1e-9 < unpaired.half_width
    assert paired.estimate == pytest.approx(0.1)


def test_vector_metric_matches_scalar_runs():
    """A metric returning an array gives the same pointwise CI as running each component as a scalar metric."""
    rng = np.random.default_rng(3)
    clusters = rng.integers(0, 40, 400)
    models = {"a": [rng.normal(0, 1, (400, 2)) for _ in range(3)], "b": [rng.normal(0.2, 1, (400, 2)) for _ in range(3)]}
    comb = lambda v: v["a"] - v["b"]
    vec = St.two_level_bootstrap(lambda p, r: p[r].mean(0), models, comb, clusters, B=200)
    assert vec.estimate.shape == vec.lo.shape == vec.hi.shape == (2,)
    for j in range(2):
        sc = St.two_level_bootstrap(lambda p, r: p[r, j].mean(), models, comb, clusters, B=200)
        assert isinstance(sc.lo, float)
        assert (sc.estimate, sc.lo, sc.hi) == pytest.approx((vec.estimate[j], vec.lo[j], vec.hi[j]))


def test_precision_at_recalls_matches_fcr():
    score, y = _data(K=3)
    grid = np.array([0.5, 0.7, 0.9, 1.0])
    prec = M.precision_at_recalls(score, y, grid)
    assert prec.shape == (4,)
    assert prec[2] == pytest.approx(1 - M.fcr_at_recall(score, y, 0.9))
    assert prec[3] == pytest.approx(np.mean(y.mean(0) / np.mean(score >= np.min(np.where(y, score, 1), 0), 0)))


# ---------------------------------------------------------------- readings (no margins)
def _r(lo, hi):
    return St.Result((lo + hi) / 2, lo, hi, np.array([]))


@pytest.mark.parametrize("kind, lo, hi, direction, expected", [
    ("superiority", -0.3, -0.1, -1, "success"),
    ("superiority", 0.1, 0.3, -1, "fail"),
    ("superiority", -0.1, 0.1, -1, "inconclusive"),
    ("superiority", 0.1, 0.3, +1, "success"),
    ("superiority", -0.3, -0.1, +1, "fail"),
    ("difference", 0.01, 0.03, +1, "better"),                       # e.g. Δ BACC > 0
    ("difference", -0.03, -0.01, +1, "worse"),
    ("difference", -0.03, -0.01, -1, "better"),                     # e.g. Δ NLL or Δ FCR < 0
    ("difference", 0.01, 0.03, -1, "worse"),
    ("difference", -0.001, 0.001, +1, "no evidence of a difference"),   # a narrow CI is still never "equivalent"
])
def test_verdicts(kind, lo, hi, direction, expected):
    assert St.verdict(_r(lo, hi), kind, direction) == expected


def test_no_margins_left():
    assert not hasattr(St, "margin") and not hasattr(St, "CAPS")
    with pytest.raises(ValueError):
        St.verdict(_r(-0.01, 0.01), "equivalence")
    with pytest.raises(ValueError):
        St.verdict(_r(-0.01, 0.01), "difference", 0)
