"""Unit tests for module a (heads, losses, arm registry). Run: python -m pytest tests/test_module_a.py -q"""
import math

import numpy as np
import pytest
import torch
from scipy import stats

from b2j import losses as L
from b2j.arms import ARMS, build
from b2j.heads import MultiHead

torch.manual_seed(0)


def _votes(N=4000, K=3, n=3, a=(0.6, 0.15)):
    a1, a0 = a
    theta = torch.distributions.Beta(torch.tensor(a1), torch.tensor(a0)).sample((N, K))
    nn_ = torch.full((N, K), float(n))
    return torch.distributions.Binomial(nn_, theta).sample(), nn_


def test_beta_binomial_matches_scipy():
    a0 = torch.tensor([0.3, 2.0, 50.0, 1e-3]).double()
    a1 = torch.tensor([0.7, 0.5, 10.0, 2.0]).double()
    k = torch.tensor([0.0, 2.0, 31.0, 1.0]).double()
    n = torch.tensor([3.0, 3.0, 51.0, 7.0]).double()
    ours = L.beta_binomial_nll(a0, a1, k, n).numpy()
    ref = -stats.betabinom.logpmf(k.numpy(), n.numpy(), a1.numpy(), a0.numpy())
    np.testing.assert_allclose(ours, ref, rtol=1e-8)


def test_beta_kl_matches_torch():
    a0, a1 = torch.tensor([0.5, 3.0]), torch.tensor([2.0, 0.2])
    p0, p1 = torch.tensor([1.7, 1.0]), torch.tensor([0.3, 1.0])
    ref = torch.distributions.kl_divergence(torch.distributions.Beta(a1, a0), torch.distributions.Beta(p1, p0))
    torch.testing.assert_close(L.beta_kl(a0, a1, p0, p1), ref)


def test_split_target_is_hypergeometric():
    for n in (3, 5, 7, 51):
        for k in range(n + 1):
            ref = 1 - stats.hypergeom.pmf(0, n, k, 3) - stats.hypergeom.pmf(3, n, k, 3)
            got = L.split_target(torch.tensor(float(k)), torch.tensor(float(n))).item()
            assert math.isclose(got, ref, abs_tol=1e-6)
    k = torch.arange(4.0)
    assert L.split_target(k, torch.full_like(k, 3.0)).tolist() == [0, 1, 1, 0]
    with pytest.raises(ValueError):
        L.split_target(torch.tensor(1.0), torch.tensor(2.0))


def test_majority_is_strict():
    k = torch.tensor([1.0, 2.0, 5.0, 6.0])
    n = torch.tensor([3.0, 3.0, 10.0, 10.0])
    assert L.majority(k, n).tolist() == [0, 1, 0, 1]


def test_class_weights_formula():
    k = torch.tensor([[3.0], [2.0], [0.0], [1.0]])
    w = L.class_weights(k, torch.full_like(k, 3.0))
    torch.testing.assert_close(w, torch.tensor([[4 / 4, 4 / 4]]))
    k = torch.tensor([[3.0], [0.0], [0.0], [1.0]])
    w = L.class_weights(k, torch.full_like(k, 3.0))
    torch.testing.assert_close(w, torch.tensor([[4 / 6, 4 / 2]]))


def test_moment_prior_recovers_population_beta():
    k, n = _votes(N=200_000, K=1, n=51, a=(1.2, 4.8))       # pi = 0.2, S = 6
    a0, a1 = L.moment_matched_prior(k, n)[0]
    assert abs(a1 / (a0 + a1) - 0.2) < 0.005
    assert abs((a0 + a1) - 6.0) < 0.3


def test_soft_bce_target_and_no_pos_weight():
    loss = L.PanelLoss(L.LossConfig("soft_bce"))
    z = torch.zeros(2, 1, 1)
    k, n = torch.tensor([[1.0], [3.0]]), torch.tensor([[3.0], [3.0]])
    assert torch.allclose(loss.per_sample(z, k, n), torch.full((2, 1), math.log(2)))


def test_b2r_minimum_is_at_vote_fraction():
    """At fixed large S, the B²-r loss over a batch is minimised at m = mean k/n (the decomposition)."""
    k, n = torch.tensor([[1.0], [2.0], [0.0]]).double(), torch.full((3, 1), 3.0).double()
    S = 1e4
    ms = torch.linspace(0.05, 0.95, 181).double()
    nll = torch.stack([L.beta_binomial_nll((1 - m) * S, m * S, k, n).mean() for m in ms])
    assert abs(ms[nll.argmin()].item() - 1 / 3) < 0.006


def test_legacy_b2m_matches_miccai_code():
    from scripts.bbkl_loss import total_bb_loss
    torch.manual_seed(1)
    k, n = _votes(N=64, K=3, n=3)
    logits = torch.randn(64, 3, 2) * 2
    weights = L.class_weights(k, n)
    prior = L.miccai_prior(k, n)
    ours = L.PanelLoss(ARMS["b2m"].loss, weights=weights, prior=prior)(logits, k, n)
    a0, a1 = L.alphas(logits, 1e-6, plus_one=True)
    alpha = [torch.stack([a0[:, i], a1[:, i]], -1) for i in range(3)]
    keys = ["C1", "C2", "C3"]
    ref = total_bb_loss(alpha, k / 3,
                        weights={c: (weights[i, 1].item(), weights[i, 0].item()) for i, c in enumerate(keys)},
                        lambda_reg=0.05, use_kl=True,
                        prior_alpha={c: tuple(prior[i].tolist()) for i, c in enumerate(keys)})
    torch.testing.assert_close(ours, ref.to(ours.dtype), rtol=1e-5, atol=1e-5)


def test_alphas_floor():
    a0, a1 = L.alphas(torch.full((1, 2), -50.0), eps=1e-3)
    assert torch.allclose(a1, torch.tensor(1e-3))
    a0, _ = L.alphas(torch.full((1, 2), -50.0), eps=1e-6, plus_one=True)
    assert a0.item() >= 1.0


def test_gradients_finite_at_extremes():
    k, n = torch.tensor([[0.0, 3.0, 25.0]]), torch.tensor([[3.0, 3.0, 51.0]])
    for scale in (-30.0, 0.0, 30.0):
        z = torch.full((1, 3, 2), scale, requires_grad=True)
        L.PanelLoss(L.LossConfig("b2"))(z, k, n).backward()
        assert torch.isfinite(z.grad).all()


def test_stuck_share():
    logits = torch.tensor([[[0.0, -8.0]], [[0.0, -8.0]], [[0.0, 1.0]], [[0.0, -9.0]]])
    k = torch.tensor([[1.0], [2.0], [1.0], [0.0]])
    assert L.stuck_share(logits, k).item() == pytest.approx(2 / 3)


def test_multihead_shapes():
    z = torch.randn(5, 768)
    assert MultiHead(768, 10, 2)(z).shape == (5, 10, 2)
    assert MultiHead(768, 3, 1, "linear")(z).shape == (5, 3, 1)
    mlp = MultiHead(768, 1, 2).heads[0]
    assert [m.__class__.__name__ for m in mlp] == ["Linear", "GELU", "Dropout", "Linear"]
    assert mlp[0].out_features == 256 and mlp[2].p == 0.1


@pytest.mark.parametrize("arm", list(ARMS))
def test_every_arm_trains_one_step(arm):
    k, n = _votes(N=32, K=3, n=3)
    head, loss = build(arm, 768, k, n)
    opt = torch.optim.SGD(head.parameters(), lr=1e-2)
    value = loss(head(torch.randn(32, 768)), k, n)
    value.backward()
    opt.step()
    assert torch.isfinite(value)


def test_dup_refuses_small_panels():
    k, n = _votes(N=8, K=1, n=1)
    with pytest.raises(ValueError):
        build("dup", 16, k, n)


def test_eps_override_spares_b2m():
    k, n = _votes(N=8, K=1, n=3)
    assert build("b2r", 16, k, n, eps=0.1)[1].cfg.eps == 0.1
    assert build("b2m", 16, k, n, eps=0.1)[1].cfg.eps == 1e-6


def test_config_validation():
    with pytest.raises(ValueError):
        L.LossConfig("soft_bce", kl_lambda=0.05)
    with pytest.raises(ValueError):
        L.PanelLoss(L.LossConfig("b2", weighted=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_loss_stays_on_gpu():
    k, n = (t.cuda() for t in _votes(N=16, K=3, n=3))
    head, loss = build("b2r_kl", 32, k, n)
    out = head.cuda()(torch.randn(16, 32, device="cuda"))
    assert loss.cuda().per_sample(out, k, n).is_cuda
