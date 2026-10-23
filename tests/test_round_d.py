"""Round D candidates (journal_plan/iterations.md): mean-concentration head (D1) and concentration shrinkage (D2)."""
import numpy as np
import pytest
import torch
from scipy import stats

from b2j import losses as L
from b2j import posthoc as P
from b2j.blocks import smoke
from b2j.variants import MAIN, build


def test_ms_alphas_match_mean_and_concentration_and_posthoc():
    out = torch.tensor([[[0.3, 1.2], [-4.0, 6.0], [50.0, -20.0]]], dtype=torch.float64)    # last: clamped
    a0, a1, log_s = L.ms_alphas(out)
    m, S = P.mean_concentration("b2ms", out.numpy())
    np.testing.assert_allclose((a1 / (a0 + a1)).numpy(), m, rtol=1e-10)
    np.testing.assert_allclose((a0 + a1).numpy(), S, rtol=1e-10)
    assert torch.isfinite(torch.log(a1)).all() and torch.isfinite(torch.log(a0)).all()
    assert log_s[0, 2].item() == L.LOG_S_BOUNDS[0]


def test_ms_loss_is_beta_binomial_and_has_finite_grads():
    head, loss = build("b2ms", 4, torch.zeros(5, 1), torch.full((5, 1), 3.0))
    out = torch.tensor([[[0.5, 0.0]], [[-2.0, 3.0]], [[8.0, -9.0]]], requires_grad=True)
    k, n = torch.tensor([[1.0], [0.0], [3.0]]), torch.full((3, 1), 3.0)
    ps = loss.per_sample(out, k, n)
    m = torch.sigmoid(out[..., 0].double()).detach().numpy()
    S = torch.exp(out[..., 1].double()).detach().numpy()
    ref = -stats.betabinom.logpmf(k.numpy(), n.numpy(), m * S, (1 - m) * S)
    np.testing.assert_allclose(ps.detach().numpy(), ref, rtol=1e-5)
    ps.sum().backward()
    assert torch.isfinite(out.grad).all()


def test_shrink_pulls_log_s_to_the_prior():
    k = torch.tensor([[0.0], [3.0], [1.0], [2.0]] * 50)
    n = torch.full_like(k, 3.0)
    _, loss = build("b2ms_shrink0.3", 4, k, n)
    log_s0 = torch.log(loss.prior[:, 0] + loss.prior[:, 1]).item()
    base = torch.zeros(1, 1, 2)
    at = lambda ls: loss.per_sample(base + torch.tensor([0.0, ls]), torch.tensor([[1.0]]), torch.tensor([[3.0]])).item()
    _, plain = build("b2ms", 4, k, n)
    pen = lambda ls: at(ls) - plain.per_sample(base + torch.tensor([0.0, ls]), torch.tensor([[1.0]]), torch.tensor([[3.0]])).item()
    assert pen(log_s0) == pytest.approx(0.0, abs=1e-6)
    assert pen(log_s0 + 2) == pytest.approx(0.3 * 4, rel=1e-4)


def test_ms_head_learns_a_concentration_that_depends_on_x():
    """x0 sets the mean, x1 the concentration: S = 1 where x1 > 0 (θ widely spread: panels lean hard one way), 100 elsewhere
    (votes follow m)."""
    g = torch.Generator().manual_seed(0)
    N = 4000
    x = torch.randn(N, 4, generator=g)
    m = torch.sigmoid(x[:, 0])
    S = torch.where(x[:, 1] > 0, 1.0, 100.0)
    theta = torch.distributions.Beta(m * S, (1 - m) * S).sample()
    k = torch.distributions.Binomial(torch.full((N,), 5.0), theta).sample()[:, None]
    n = torch.full_like(k, 5.0)
    head, loss = build("b2ms", 4, k, n, dropout=0.0)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    for _ in range(600):
        opt.zero_grad()
        loss(head(x), k, n).backward()
        opt.step()
    _, S_hat = P.mean_concentration("b2ms", head(x).detach().numpy())
    lo, hi = np.median(S_hat[x[:, 1].numpy() > 0]), np.median(S_hat[x[:, 1].numpy() <= 0])
    assert lo < 3 and hi > 20


def test_round_d_variants_stay_out_of_the_pre_registered_blocks():
    import inspect
    import b2j.blocks as B
    assert {s.variant for s in smoke()} <= set(MAIN)
    assert "for v in VARIANTS" not in inspect.getsource(B)          # r1 itself needs tune/chosen.json


def test_batch_shrink_penalises_only_the_variation_of_log_s():
    k = torch.tensor([[1.0], [2.0], [0.0], [3.0]])
    n = torch.full_like(k, 3.0)
    _, loss = build("b2ms_var3", 4, k, n)
    _, plain = build("b2ms", 4, k, n)
    assert loss.prior is None                                   # no prior needed
    same = torch.tensor([[[0.3, 1.7]], [[-0.2, 1.7]], [[1.0, 1.7]], [[0.0, 1.7]]])   # equal log S: no penalty
    assert torch.allclose(loss.per_sample(same, k, n), plain.per_sample(same, k, n))
    out = same.clone()
    out[:, 0, 1] = torch.tensor([0.0, 1.0, 2.0, 3.0])          # log S varies: penalty = 3 * (log S - 1.5)^2
    pen = loss.per_sample(out, k, n) - plain.per_sample(out, k, n)
    assert torch.allclose(pen[:, 0], 3 * (out[:, 0, 1] - 1.5) ** 2, atol=1e-5)
    out.requires_grad_(True)
    g, = torch.autograd.grad((loss(out, k, n) - plain(out, k, n)), out)
    assert abs(g[:, 0, 1].sum().item()) < 1e-6                 # the penalty does not move the mean level of log S


def test_vote_count_weighted_soft_bce_is_the_rescaled_binomial_nll():
    from b2j.losses import LossConfig, PanelLoss
    import torch
    torch.manual_seed(0)
    out = torch.randn(6, 2, 1)
    n = torch.tensor([4., 10., 6., 8., 10., 5.])[:, None].repeat(1, 2)
    k = torch.minimum(torch.randint(0, 11, (6, 2)).float(), n)
    plain = PanelLoss(LossConfig("soft_bce")).per_sample(out, k, n)
    nw = PanelLoss(LossConfig("soft_bce", n_weight=True)).per_sample(out, k, n)
    assert torch.allclose(nw, plain * n / n.mean(0))
    p = torch.sigmoid(out[..., 0])                 # Binomial NLL without the binomial coefficient = n × soft BCE
    binom = -(k * torch.log(p) + (n - k) * torch.log(1 - p))
    assert torch.allclose(nw, binom / n.mean(0), atol=1e-5)
    same_n = torch.full_like(n, 7.0)
    assert torch.allclose(PanelLoss(LossConfig("soft_bce", n_weight=True)).per_sample(out, k.clamp(max=7), same_n),
                          PanelLoss(LossConfig("soft_bce")).per_sample(out, k.clamp(max=7), same_n))


def test_n_weight_is_soft_bce_only():
    import pytest
    from b2j.losses import LossConfig
    with pytest.raises(ValueError):
        LossConfig("b2", n_weight=True)


def test_dirichlet_multinomial_nll_matches_scipy_and_reduces_to_beta_binomial():
    import torch
    from scipy.stats import dirichlet_multinomial, betabinom
    from b2j.losses import beta_binomial_nll, dm_nll
    alpha = torch.tensor([[0.3, 2.0, 5.0], [40.0, 1e-3, 7.0], [1e4, 2e4, 3e4]]).double()
    k = torch.tensor([[0., 3, 7], [5, 0, 5], [10, 20, 33]]).double()
    want = [-dirichlet_multinomial.logpmf(k[i].numpy(), alpha[i].numpy(), int(k[i].sum())) for i in range(3)]
    assert torch.allclose(dm_nll(alpha.log(), k), torch.tensor(want).double(), rtol=1e-8, atol=1e-8)
    a2 = torch.tensor([[0.7, 1.3], [5.0, 0.2]]).double()        # K = 2: the Beta-Binomial (votes for class 1 of n)
    k2 = torch.tensor([[2., 5], [6, 1]]).double()
    bb = beta_binomial_nll(a2[:, 0], a2[:, 1], k2[:, 1], k2.sum(1))
    assert torch.allclose(dm_nll(a2.log(), k2), bb)
    assert np.allclose(bb.numpy(), [-betabinom.logpmf(5, 7, 1.3, 0.7), -betabinom.logpmf(1, 7, 0.2, 5.0)])


def test_dm_head_shares_one_concentration_and_marginals_are_beta_binomial():
    import torch
    from b2j import posthoc as P
    from b2j.variants import build
    k = torch.distributions.Multinomial(10, torch.ones(4)).sample((16,))
    n = k.sum(1, keepdim=True).expand(-1, 4).contiguous()
    head, loss = build("dm", 8, k, n)
    out = head(torch.randn(16, 8)).detach()
    assert out.shape == (16, 4, 2) and torch.equal(out[..., 1], out[:, :1, 1].expand(-1, 4))
    m, S = P.mean_concentration("dm", out.numpy())
    assert np.allclose(m.sum(1), 1) and np.allclose(S, S[:, :1])
    value = loss(out.requires_grad_(), k, n)                     # = batch mean of the per-image DM NLL
    from b2j.losses import dm_nll
    lm = torch.log_softmax(out[..., 0].double(), -1)
    assert torch.allclose(value.double(), dm_nll(lm + out[:, 0, 1].double()[:, None], k.double()).mean(), atol=1e-5)
    with pytest.raises(ValueError):                              # multi-label votes are refused
        build("dm", 8, torch.ones(4, 3), torch.full((4, 3), 2.0))


def test_soft_ce_and_joint_b2r_shapes():
    import torch
    from b2j.variants import build
    k = torch.distributions.Multinomial(5, torch.ones(3)).sample((8,))
    n = k.sum(1, keepdim=True).expand(-1, 3).contiguous()
    h, l = build("soft_ce", 8, k, n)
    assert h(torch.randn(8, 8)).shape == (8, 3, 1) and torch.isfinite(l(h(torch.randn(8, 8)), k, n))
    h, l = build("b2r_joint", 8, k, n)
    assert h(torch.randn(8, 8)).shape == (8, 3, 2) and sum(p.numel() for p in h.parameters()) == 8 * 256 + 256 + 256 * 6 + 6
