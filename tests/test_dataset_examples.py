import numpy as np

from b2j.dataset_examples import vote_bins


def test_fixed_panel_uses_exact_counts():
    k, n = np.array([[0.0], [2.0], [3.0]]), np.full((3, 1), 3.0)
    assert vote_bins(k, n)[:, 0].tolist() == ["0/3", "2/3", "3/3"]


def test_varying_panel_uses_share_and_skips_zero():
    k, n = np.array([[0.0], [5.0], [50.0], [11.0]]), np.array([[50.0], [50.0], [50.0], [55.0]])
    assert vote_bins(k, n)[:, 0].tolist() == ["", "0%-20%", "80%-100%", "0%-20%"]
