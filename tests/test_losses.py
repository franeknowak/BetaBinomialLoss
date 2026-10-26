"""Unit tests for the B² loss. Run: python -m pytest tests/test_losses.py -q"""
import numpy as np
import pytest
import torch
from scipy import stats

from b2j import losses as L

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


def test_b2_minimum_is_at_vote_fraction():
    """At fixed large S, the B² loss over a batch is minimised at m = mean k/n (the decomposition)."""
    k, n = torch.tensor([[1.0], [2.0], [0.0]]).double(), torch.full((3, 1), 3.0).double()
    S = 1e4
    ms = torch.linspace(0.05, 0.95, 181).double()
    nll = torch.stack([L.beta_binomial_nll((1 - m) * S, m * S, k, n).mean() for m in ms])
    assert abs(ms[nll.argmin()].item() - 1 / 3) < 0.006


def test_alphas_floor():
    a0, a1 = L.alphas(torch.full((1, 2), -50.0), eps=1e-3)
    assert torch.allclose(a1, torch.tensor(1e-3)) and torch.allclose(a0, torch.tensor(1e-3))


def test_gradients_finite_at_extremes():
    k, n = torch.tensor([[0.0, 3.0, 25.0]]), torch.tensor([[3.0, 3.0, 51.0]])
    for scale in (-30.0, 0.0, 30.0):
        z = torch.full((1, 3, 2), scale, requires_grad=True)
        L.B2Loss()(z, k, n).backward()
        assert torch.isfinite(z.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_loss_stays_on_gpu():
    k, n = (t.cuda() for t in _votes(N=16, K=3, n=3))
    out = torch.randn(16, 3, 2, device="cuda")
    assert L.B2Loss().per_sample(out, k, n).is_cuda


def test_float32_inputs_are_exact_at_large_concentration():
    """The loss is computed in float64 inside B2Loss, so float32 logits with S ~ 1e4 still match scipy."""
    logits = torch.tensor([[[7000.0, 3000.0]], [[6999.0, 3001.0]]])      # softplus(x) = x here
    k, n = torch.tensor([[1.0], [1.0]]), torch.tensor([[3.0], [3.0]])
    ours = L.B2Loss().per_sample(logits, k, n)
    assert ours.dtype == torch.float32
    a0, a1 = logits[..., 0].double() + 1e-3, logits[..., 1].double() + 1e-3
    ref = -stats.betabinom.logpmf(1, 3, a1.numpy(), a0.numpy())
    np.testing.assert_allclose(ours.numpy(), ref, rtol=1e-6)
    assert ours[1, 0] != ours[0, 0]


def test_bb_nll_matches_lgamma_form_and_keeps_lgamma_off_gpu(monkeypatch):
    """The lgamma-free B² NLL (issue #22 workaround) equals the closed form, incl. large S and n up to 63,
    and every lgamma/digamma call it makes is on a CPU tensor."""
    from b2j import losses as L
    g = torch.Generator().manual_seed(0)
    n = torch.randint(1, 64, (200, 3), generator=g).double()
    k = torch.floor(torch.rand(200, 3, generator=g, dtype=torch.float64) * (n + 1)).clamp(max=n)
    a0 = torch.exp(torch.randn(200, 3, generator=g, dtype=torch.float64) * 3)
    a1 = torch.exp(torch.randn(200, 3, generator=g, dtype=torch.float64) * 3)
    closed = -(torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
               + L._log_beta(k + a1, n - k + a0) - L._log_beta(a1, a0))
    torch.testing.assert_close(L.beta_binomial_nll(a0, a1, k, n), closed, rtol=1e-9, atol=1e-7)
    seen = []
    real = torch.lgamma, torch.digamma
    monkeypatch.setattr(torch, "lgamma", lambda x: (seen.append(x.device.type), real[0](x))[1])
    monkeypatch.setattr(torch, "digamma", lambda x: (seen.append(x.device.type), real[1](x))[1])
    L.beta_binomial_nll(a0, a1, k, n)
    L.beta_kl(a0, a1, torch.tensor([1.0, 2.0, 3.0]).double(), torch.tensor([2.0, 1.0, 0.5]).double())
    assert seen and set(seen) == {"cpu"}
