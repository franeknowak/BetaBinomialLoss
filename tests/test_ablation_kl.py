"""Tests for the KL-prior ablation of B² (b2j/ablation_kl.py).
Run: python -m pytest tests/test_ablation_kl.py -q
"""
import json

import numpy as np
import pytest
import torch
from scipy import integrate, stats

from b2j import ablation_kl as A
from b2j import losses as L
from tests.test_pipeline import _fake_cache


def test_moment_prior_worked_example_endoscapes_c1():
    """Counts of Endoscapes C1 train (k = 0..3 of 3): 4594, 951, 963, 193 -> pi .1719, rho .3313, nu 2.018."""
    k = torch.tensor(sum(([j] * c for j, c in enumerate((4594, 951, 963, 193))), []), dtype=torch.float64)[:, None]
    p = A.moment_prior(k, torch.full_like(k, 3.0))
    nu = float(p.sum())
    assert float(p[0, 1]) / nu == pytest.approx(0.171915, abs=1e-5) and nu == pytest.approx(2.01815, abs=1e-4)


def test_moment_prior_recovers_simulated_beta_binomial_with_unequal_n():
    rng = np.random.default_rng(0)
    pi, nu, N = 0.2, 3.0, 200_000
    theta = rng.beta(pi * nu, (1 - pi) * nu, N)
    n = rng.integers(2, 12, N).astype(float)
    k = rng.binomial(n.astype(int), theta).astype(float)
    p = A.moment_prior(torch.tensor(k)[:, None], torch.tensor(n)[:, None])
    assert float(p.sum()) == pytest.approx(nu, rel=0.03) and float(p[0, 1] / p.sum()) == pytest.approx(pi, abs=0.003)


def test_uniform_prior():
    k = torch.zeros(5, 3)
    assert torch.equal(A.make_prior("uniform", k, k + 3), torch.ones(3, 2, dtype=torch.float64))


def test_beta_kl_matches_numerical_integral_and_is_zero_at_prior():
    a1, a0, p1, p0 = 0.7, 3.0, 0.35, 1.67
    q, r = stats.beta(a1, a0), stats.beta(p1, p0)
    ref = integrate.quad(lambda x: q.pdf(x) * (q.logpdf(x) - r.logpdf(x)), 0, 1, limit=200)[0]
    t = lambda x: torch.tensor([x], dtype=torch.float64)
    assert float(L.beta_kl(t(a0), t(a1), t(p0), t(p1))) == pytest.approx(ref, rel=1e-5)
    assert float(L.beta_kl(t(p0), t(p1), t(p0), t(p1))) == pytest.approx(0.0, abs=1e-12)


def test_loss_is_b2_nll_plus_lambda_kl_mean_over_images_sum_over_labels():
    torch.manual_seed(0)
    out = torch.randn(6, 3, 2, dtype=torch.float64)
    k = torch.randint(0, 4, (6, 3)).double()
    n = torch.full_like(k, 3.0)
    prior = torch.tensor([[1.67, 0.35], [1.0, 1.0], [2.0, 0.5]], dtype=torch.float64)
    a0, a1 = L.alphas(out)
    kl = L.beta_kl(a0, a1, prior[:, 0].expand_as(a0), prior[:, 1].expand_as(a1))
    ref = (L.beta_binomial_nll(a0, a1, k, n) + 0.05 * kl).mean(0).sum()
    torch.testing.assert_close(A.B2KLLoss(prior)(out, k, n), ref)
    assert A.LAMBDA == 0.05
    zero = A.B2KLLoss(prior, lam=0.0)(out, k, n)
    torch.testing.assert_close(zero, L.B2Loss()(out, k, n))          # λ = 0 is plain B²


def test_run_uses_b2_settings_trains_on_full_loss_and_saves_b2_outputs(tmp_path, monkeypatch):
    from b2j.train import load_embeddings
    data = load_embeddings("toy", cache=_fake_cache(tmp_path))
    sel = tmp_path / "set" / "tuning" / "toy"
    sel.mkdir(parents=True)
    (sel / "selection.json").write_text(json.dumps({"b2": {"lr": 1e-3, "wd_level": 0.25}}))
    A.run("toy", variants=("moment",), seeds=(1,), root=tmp_path / "out", settings_root=tmp_path / "set", epochs=1,
          data=data)
    d = tmp_path / "out" / "ablation_kl" / "final" / "toy" / "b2_kl_moment" / "seed1"
    r = json.loads((d / "run.json").read_text())
    assert r["lr"] == 1e-3 and r["wd_level"] == 0.25 and r["kl_lambda"] == 0.05 and r["variant"] == "b2_kl_moment"
    assert r["checkpoints"]["b2kl"]["eval"] == r["best_val_loss_eval"]
    o = json.loads((d / "outputs_b2kl.json").read_text())
    assert o["outputs"] == ["alpha0", "alpha1"] and len(o["images"]) == 120
    # the curve's val loss is the full loss (NLL + λ KL), not the plain NLL
    from b2j.heads import MultiHead
    m = MultiHead(768, 3, 2)
    m.load_state_dict(torch.load(d / "head_b2kl.pt"))
    m.eval()
    X, k, n, _ = data[0]["val"]
    prior = torch.tensor(r["prior"])
    with torch.no_grad():
        full = A.B2KLLoss(prior)(m(X), k, n).item()
        plain = L.B2Loss()(m(X), k, n).item()
    assert full == pytest.approx(r["curve"][r["best_val_loss_eval"] - 1]["val_loss"], rel=1e-5) and full > plain
