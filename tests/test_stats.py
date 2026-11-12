"""Tests for b2j/stats.py (phase-1 statistics). Run: python -m pytest tests/test_stats.py -q"""
import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from b2j import losses as L
from b2j import metrics as M
from b2j import stats as T


def _data(rng, N=300, K=3, n=3):
    y_share = rng.random((N, K))
    k = rng.binomial(n, y_share)
    return k, np.full((N, K), n)


@pytest.mark.parametrize("ties", [False, True])
def test_weighted_auroc_ap_match_sklearn(ties):
    rng = np.random.default_rng(0)
    score = rng.random(200)
    if ties:
        score = np.round(score, 1)
    y = rng.random(200) < 0.3
    W = rng.integers(0, 4, size=(5, 200)).astype(float)
    a, p = T.weighted_auroc(score, y, W), T.weighted_ap(score, y, W)
    for b in range(5):
        assert a[b] == pytest.approx(roc_auc_score(y, score, sample_weight=W[b]))
        assert p[b] == pytest.approx(average_precision_score(y, score, sample_weight=W[b]))


@pytest.mark.parametrize("loss", ["b2", "soft_bce"])
def test_unit_weights_reproduce_metrics_compute(loss):
    """All weights 1 = the plain test set: the same 7 numbers as b2j.metrics.compute."""
    rng = np.random.default_rng(1)
    k, n = _data(rng)
    raw = torch.randn(300, 3, 2 if loss == "b2" else 1)
    saved = torch.stack(L.alphas(raw.double()), -1).numpy() if loss == "b2" else raw.double().numpy()
    ref = M.compute(loss, raw, torch.tensor(k).float(), torch.tensor(n).float(), ["a", "b", "c"])
    got = T.weighted_metrics(T.entries(loss, saved, k, n), np.ones((1, 300)))
    for x in ("ap", "auroc", "mse", "vote_nll", "rps"):
        assert got[x][0] == pytest.approx(ref["macro"][x], rel=1e-6)
    assert got["spread"][0] == pytest.approx(ref["pooled"]["spread_ratio"], rel=1e-6)
    assert got["cov_gap"][0] == pytest.approx(ref["pooled"]["coverage"]["gap"], abs=1e-9)


def test_cluster_weights_keep_clusters_together():
    rng = np.random.default_rng(2)
    cluster = np.repeat(np.arange(10), 5)                  # 10 clusters of 5 images
    W = T.cluster_weights(cluster, 100, rng)
    assert np.all(W.reshape(100, 10, 5).std(2) == 0)       # all images of a cluster share one weight
    assert np.allclose(W.sum(1), 50)                       # 10 clusters drawn, 5 images each


def test_improvement_sign_and_welch_direction():
    assert T.improvement("auroc", 0.8, 0.7) > 0 and T.improvement("vote_nll", 0.8, 0.9) > 0
    assert T.improvement("spread", 1.05, 2.0) > 0 and T.improvement("spread", 0.5, 1.2) < 0
    b2, other = np.array([0.80, 0.81, 0.79, 0.80, 0.82]), np.array([0.90, 0.91, 0.92, 0.90, 0.89])
    assert T.welch_one_tailed("vote_nll", b2, other) < 0.01        # lower NLL for B² -> significant
    assert T.welch_one_tailed("auroc", b2, other) > 0.99           # lower AUROC for B² -> not
    assert np.isnan(T.welch_one_tailed("auroc", np.ones(5), np.ones(5)))
