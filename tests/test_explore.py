"""explore/: rater subsampling and the weighted metrics of the bootstrap."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from explore.bootstrap import weighted_auroc, weighted_bacc
from explore.subsample import subsample


def test_subsample_single_choice_and_per_label():
    rng = np.random.default_rng(0)
    k = np.array([[30, 20, 1], [0, 51, 0]], float)
    ks, ns = subsample(k, np.full((2, 3), 51.0), 3, rng)
    assert (ks.sum(1) == 3).all() and (ns == 3).all() and (ks <= k).all()
    k = np.array([[5], [7], [0]], float)
    ks, ns = subsample(k, np.full((3, 1), 7.0), 3, rng)
    assert ks[1, 0] == 3 and ks[2, 0] == 0 and (ns == 3).all()
    ks, ns = subsample(np.array([[2, 1]], float), np.array([[3, 3]], float), 5, rng)   # fewer raters than m: kept
    assert ks.tolist() == [[2, 1]] and ns.tolist() == [[3, 3]]


def test_weighted_auroc_matches_sklearn_and_repetition():
    rng = np.random.default_rng(1)
    s = np.round(rng.random(200), 2)                                        # ties
    y = (rng.random(200) < s).astype(int)
    assert weighted_auroc(s, y, np.ones((1, 200)))[0] == pytest.approx(roc_auc_score(y, s))
    w = rng.integers(0, 3, 200)
    rep = np.repeat(np.arange(200), w)
    assert weighted_auroc(s, y, w[None].astype(float))[0] == pytest.approx(roc_auc_score(y[rep], s[rep]))


def test_weighted_bacc():
    pred = np.array([True, False, True, False])
    y = np.array([1, 1, 0, 0])
    assert weighted_bacc(pred, y, np.ones((1, 4)))[0] == pytest.approx(0.5)
    assert weighted_bacc(pred, y, np.array([[2.0, 0, 0, 1]]))[0] == pytest.approx(1.0)
