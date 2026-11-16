import numpy as np
import torch
from scipy.stats import betabinom, binom

from b2j import metrics as M
from b2j import rater_bias as RB


def _rand_out(N=50, K=3, w=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(N, K, w, generator=g, dtype=torch.float64) * 2


def test_count_pmf_b2_matches_beta_binomial_at_zero_bias():
    out = _rand_out()
    a0, a1 = RB.alphas(out)
    pmf = RB.count_pmf("b2", out, torch.zeros(3, 3)).numpy()
    ref = betabinom.pmf(np.arange(4), 3, a1.numpy()[..., None], a0.numpy()[..., None])
    assert np.abs(pmf - ref).max() < 1e-4
    assert np.allclose(pmf.sum(-1), 1)


def test_count_pmf_soft_bce_matches_binomial_at_zero_bias():
    out = _rand_out(w=1)
    pmf = RB.count_pmf("soft_bce", out, torch.zeros(3, 3)).numpy()
    ref = binom.pmf(np.arange(4), 3, torch.sigmoid(out[..., 0]).numpy()[..., None])
    assert np.abs(pmf - ref).max() < 1e-12


def test_ordered_nll_sums_to_count_pmf_with_bias():
    """Σ over the vote patterns with k yes votes of P(pattern) = P(k), for nonzero rater biases."""
    out = _rand_out(N=20, K=1)
    beta = torch.tensor([[-1.5, 0.3, 1.2]], dtype=torch.float64)
    a0, a1 = RB.alphas(out)
    pmf = RB.count_pmf("b2", out, beta)
    agg = torch.zeros_like(pmf)
    for pat in range(8):
        y = torch.tensor([(pat >> r) & 1 for r in range(3)], dtype=torch.float64).expand(20, 1, 3)
        p = torch.exp(-RB.b2_ordered_nll(a0, a1, beta, y))
        agg[..., int(y[0, 0].sum())] += p
    assert torch.allclose(agg, pmf, atol=1e-6)


def test_ordered_nll_equals_beta_binomial_plus_log_comb_at_zero_bias():
    out = _rand_out(N=30, K=1)
    a0, a1 = RB.alphas(out)
    y = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64).expand(30, 1, 3)
    nll = RB.b2_ordered_nll(a0, a1, torch.zeros(1, 3, dtype=torch.float64), y)
    ref = -betabinom.logpmf(2, 3, a1.numpy(), a0.numpy()) + np.log(3)
    assert np.abs(nll.numpy() - ref).max() < 1e-4


def test_soft_bce_votes_equals_soft_bce_at_zero_bias():
    z = torch.randn(10, 2)
    y = torch.randint(0, 2, (10, 2, 3)).float()
    a = RB.soft_bce_votes(z, torch.zeros(2, 3), y)
    b = torch.nn.functional.binary_cross_entropy_with_logits(z, y.mean(-1), reduction="none")
    assert torch.allclose(a, b, atol=1e-6)


def test_compute_pmf_matches_compute_for_b2():
    out = _rand_out(N=200, K=2).float()
    g = torch.Generator().manual_seed(1)
    k = torch.randint(0, 4, (200, 2), generator=g).float()
    n = torch.full_like(k, 3.0)
    ref = M.compute("b2", out, k, n, ["a", "b"])
    a0, a1 = RB.alphas(out.double())
    pmf = betabinom.pmf(np.arange(4), 3, a1.numpy()[..., None], a0.numpy()[..., None])
    got = M.compute_pmf(pmf, k.numpy(), n.numpy(), ["a", "b"])
    for x in M.MACRO:
        assert abs(got["macro"][x] - ref["macro"][x]) < 1e-6, x
    assert abs(got["pooled"]["spread_ratio"] - ref["pooled"]["spread_ratio"]) < 1e-6
    assert abs(got["pooled"]["coverage"]["gap"] - ref["pooled"]["coverage"]["gap"]) < 1e-6


def test_training_runs_and_learns_rater_bias():
    """Synthetic panel: rater 0 strict, rater 2 lenient; the learned β must order the raters correctly."""
    torch.manual_seed(0)
    N, D = 600, 16
    X = torch.randn(N, D)
    logit = X[:, :1] * 1.5
    true_beta = torch.tensor([-1.5, 0.0, 1.5])
    y = (torch.rand(N, 1, 3) < torch.sigmoid(logit[..., None] + true_beta)).float()
    k = y.sum(-1)
    n = torch.full_like(k, 3.0)
    ids = np.arange(N).astype(str)
    sp = {"train": slice(0, 400), "val": slice(400, 500), "test": slice(500, 600)}
    data = {s: (X[i], k[i], n[i], ids[i], y[i]) for s, i in sp.items()}
    for loss in ("b2", "soft_bce"):
        run, state = RB.train_run("synthetic", data, ["c"], loss, 1e-3, 0.25, 1, epochs=3)
        assert run["status"] == "ok"
        beta = np.array(run["curve"][run["best_val_loss_eval"] - 1]["beta"])[0]
        assert beta[0] < beta[1] < beta[2], (loss, beta)
