"""Round D, D8: B²-rba (B² with one bias per rater) and soft BCE-ba. CPU, synthetic votes."""
import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.special import betaln, expit, gammaln
from scipy.stats import betabinom

from b2j import posthoc as P
from b2j import train as T
from b2j.heads import RaterOffsets, MultiHead
from b2j.losses import LossConfig, PanelLoss, rater_loglik
from b2j.variants import build


def _ll(m, S, beta, y):
    t = lambda x: torch.tensor(x, dtype=torch.float64)
    return rater_loglik(t([[(1 - m) * S]]), t([[m * S]]), t([[beta]]), t([[y]]), torch.ones(1, 1, len(y), dtype=torch.float64)).item()


@pytest.mark.parametrize("m", [0.001, 0.05, 0.5, 0.97])
@pytest.mark.parametrize("S", [0.05, 1.0, 30.0, 1e5])
def test_quadrature_matches_beta_binomial_at_zero_bias(m, S):
    for y in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)):
        k = sum(y)
        exact = betabinom.logpmf(k, 3, m * S, (1 - m) * S) - (gammaln(4) - gammaln(k + 1) - gammaln(4 - k))
        assert _ll(m, S, (0, 0, 0), y) == pytest.approx(exact, abs=1e-4)


def test_quadrature_with_biases_matches_adaptive_integration():
    for m, S, beta, y in ((0.2, 2.0, (-1.5, 0.3, 1.2), (0, 0, 1)), (0.6, 40.0, (2.0, -2.0, 0.0), (1, 0, 1)),
                          (0.05, 0.4, (-0.7, 0.0, 0.7), (0, 1, 1))):
        a1, a0 = m * S, (1 - m) * S
        f = lambda th: np.exp((a1 - 1) * np.log(th) + (a0 - 1) * np.log1p(-th) - betaln(a1, a0)) * np.prod(
            [expit(np.log(th / (1 - th)) + b) if v else expit(-np.log(th / (1 - th)) - b) for v, b in zip(y, beta)])
        exact = np.log(quad(f, 0, 1, limit=500, points=[1e-6, 1e-3, 0.5, 0.999])[0])
        assert _ll(m, S, beta, y) == pytest.approx(exact, abs=2e-4)


def test_numpy_and_torch_agree_and_exchangeable_case_is_exact():
    rng = np.random.default_rng(0)
    m, S = rng.uniform(0.05, 0.95, (40, 2)), rng.uniform(0.5, 60, (40, 2))
    v = (rng.random((40, 3, 2)) < 0.4).astype(float)
    beta = np.array([[-1.0, 0.2, 0.8], [0.5, -0.5, 0.0]])
    a = P.ordered_nll(m, S, beta, v)
    t = lambda x: torch.tensor(x, dtype=torch.float64)
    b = -rater_loglik(t((1 - m) * S), t(m * S), t(np.broadcast_to(beta, (40, 2, 3)).copy()), t(np.moveaxis(v, 1, 2)),
                      torch.ones(40, 2, 3, dtype=torch.float64)).numpy()
    assert np.abs(a - b).max() < 1e-9
    assert np.abs(P.ordered_nll(m, S, None, v) - P.ordered_nll(m, S, np.full((2, 3), 1e-12), v)).max() < 1e-4


def test_count_pmf_sums_to_one_handles_missing_raters_and_reduces_to_bb():
    rng = np.random.default_rng(1)
    m, S = rng.uniform(0.05, 0.95, (30, 1)), rng.uniform(0.5, 60, (30, 1))
    v = (rng.random((30, 3, 1)) < 0.5).astype(float)
    v[0, 2, 0] = np.nan
    pmf = P.rater_count_pmf(m, S, np.array([[-1.0, 0.0, 1.0]]), v)
    assert np.abs(pmf.sum(-1) - 1).max() < 1e-12 and pmf[0, 0, 3] == 0
    pmf0 = P.rater_count_pmf(m[1:], S[1:], np.full((1, 3), 1e-12), v[1:])
    assert np.abs(pmf0 - P.vote_pmf(m[1:], S[1:], np.full((29, 1), 3.0))).max() < 1e-4


def test_offsets_sum_to_zero_and_soft_bce_ba_reduces_to_soft_bce():
    head = RaterOffsets(MultiHead(8, 2, 1), 2, 3)
    with torch.no_grad():
        head.bias.copy_(torch.tensor([[1.0, 2.0, 6.0], [0.0, 0.0, 3.0]]))
    out = head(torch.randn(5, 8))
    assert out.shape == (5, 2, 4) and torch.allclose(out[..., 1:].sum(-1), torch.zeros(5, 2), atol=1e-6)
    v = (torch.rand(5, 3, 2) < 0.5).float()
    k, n = v.sum(1), torch.full((5, 2), 3.0)
    z = torch.randn(5, 2, 1)
    zero = torch.cat([z, torch.zeros(5, 2, 3)], -1)
    ba = PanelLoss(LossConfig("soft_bce", rater=True)).per_sample(zero, k, n, v)
    soft = PanelLoss(LossConfig("soft_bce")).per_sample(z, k, n)
    assert torch.allclose(ba, soft, atol=1e-6)


def test_b2rba_equals_b2r_up_to_the_ordering_constant_at_zero_bias():
    k = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    n = torch.full_like(k, 3.0)
    v = (torch.arange(3.0)[None, :, None] < k[:, None, :]).float()
    o = torch.randn(4, 1, 2)
    rba = PanelLoss(LossConfig("b2", rater=True)).per_sample(torch.cat([o, torch.zeros(4, 1, 3)], -1), k, n, v)
    b2r = PanelLoss(LossConfig("b2")).per_sample(o, k, n)
    logc = torch.tensor(gammaln(4) - gammaln(k.numpy() + 1) - gammaln(4 - k.numpy()), dtype=torch.float32)
    assert torch.allclose(rba, b2r + logc, atol=1e-3)      # ordered NLL = count NLL + log C(n, k)


def test_learned_biases_recover_known_rater_habits():
    """Fixed panel, no features: the biases a rater-aware model learns match the raters' true logit offsets."""
    rng = np.random.default_rng(0)
    N, true = 4000, np.array([-1.0, 0.0, 1.0])
    theta = rng.beta(2.0, 3.0, N)
    v = (rng.random((N, 3)) < expit(np.log(theta / (1 - theta))[:, None] + true)).astype(np.float32)[..., None]
    x = np.zeros((N, 4), np.float32)
    k = v.sum(1)
    sp = lambda idx, lab=True: T.Split(x[idx], np.arange(len(idx)).astype(str), k[idx] if lab else None,
                                       np.full_like(k[idx], 3.0) if lab else None, v[idx] if lab else None)
    cfg = T.TrainConfig("b2rba", lr=3e-2, epochs=12, evals_per_epoch=1, smooth=1, num_workers=0)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        T.run(cfg, sp(np.arange(3000)), sp(np.arange(3000, 3500)), sp(np.arange(3500, N), False), d, device="cpu",
              verbose=False)
        z = np.load(f"{d}/val_B.npz")
        beta = P.rater_offsets("b2rba", z["out"])[0]
        assert np.abs(beta - true).max() < 0.25
        m, S = P.panel("b2rba", z["out"], z["scale"])
        assert np.isfinite(S).all() and np.abs(m.mean() - 0.4).max() < 0.1


def test_rater_variants_need_votes():
    k, n = torch.ones(4, 1), torch.full((4, 1), 3.0)
    with pytest.raises(ValueError):
        build("b2rba", 8, k, n)
    _, loss = build("soft_bce_ba", 8, k, n, n_raters=3)
    with pytest.raises(ValueError):
        loss(torch.zeros(4, 1, 4), k, n)


def test_d8_block_and_report_on_fake_runs(tmp_path, monkeypatch):
    import json
    from b2j import d8 as D8
    from b2j import pilot as G
    from b2j.blocks import BLOCKS
    from b2j.data import LabelTable
    from b2j.runs import RunSpec
    # blocks: tune_ba = 2 datasets x 2 families x 6; d8 = 2 x 4 x 5 with the tune_ba choice for the new families
    assert len(BLOCKS["tune_ba"]()) == 24
    for blk, fams in (("tune", G.FAMILIES), ("tune_ba", G.BA_FAMILIES)):
        G.chosen_path(tmp_path, blk).parent.mkdir(parents=True, exist_ok=True)
        G.chosen_path(tmp_path, blk).write_text(json.dumps({ds: {f: {"lr": 1e-3 if blk == "tune_ba" else 1e-4, "wd": 1e-4}
                                                                  for f in fams} for ds in G.DATASETS}))
    specs = BLOCKS["d8"](root=tmp_path)
    assert len(specs) == 40 and {s.lr for s in specs if s.variant == "b2rba"} == {1e-3}
    # fake 3-rater table with a strong rater habit; tiny runs written where main_runs looks
    rng = np.random.default_rng(0)
    N, K = 240, 2
    X = rng.normal(size=(N, 8)).astype(np.float32)
    th = 1 / (1 + np.exp(-(X[:, :K] * 2)))
    v = (rng.random((N, 3, K)) < 1 / (1 + np.exp(-(np.log(th / (1 - th))[:, None, :] + np.array([-1, 0, 1.0])[None, :, None])))).astype(float)
    k = v.sum(1)
    split = np.array(["train"] * 160 + ["val"] * 40 + ["test"] * 40)
    table = LabelTable(np.array([f"i{i}" for i in range(N)]), k, np.full_like(k, 3.0), np.arange(N) // 4, split,
                       np.array([""] * N), votes=v)
    monkeypatch.setattr(D8, "load_table", lambda ds: table)
    for var in D8.D8:
        for seed in range(2):
            spec = RunSpec("endoscapes", var, seed=seed, block="fake", lr=1e-2)
            out = spec.out_dir(tmp_path)
            out.mkdir(parents=True)
            out.joinpath("spec.json").write_text(json.dumps({**spec.__dict__}))
            tr, va, te = (table.take(p) for p in ("train", "val", "test"))
            sp = lambda t, x, lab=True: T.split_from_table(t, x, lab)
            T.run(T.TrainConfig(var, seed=seed, lr=1e-2, epochs=2, num_workers=0), sp(tr, X[:160]), sp(va, X[160:200]),
                  sp(te, X[200:], False), out, device="cpu", verbose=False)
    text = D8.report("fake", root=tmp_path, datasets=("endoscapes",), B=30).read_text()
    assert "B²-rba" in text and "E1 (P1) ordered NLL" in text and "E3 (P3) interaction" in text
    assert "E0 Δ BACC" in text and "nan" not in text.split("Rater")[0]
    rows = text.split("Rater biases")[1]
    assert rows.count("| 0 | ") == 3                     # one line per rater of label 0
    # R3: the same report on test needs --open-test; threshold and P2's single S stay fitted on validation
    with pytest.raises(PermissionError):
        D8.report("fake", "test", root=tmp_path, datasets=("endoscapes",), B=30)
    tt = D8.report("fake", "test", root=tmp_path, open_test=True, datasets=("endoscapes",), B=30)
    assert tt.parent.name == "tables_test" and "E1 (P1)" in tt.read_text() and "nan" not in tt.read_text().split("Rater")[0]
    val = table.take("val")
    d = next(RunSpec("endoscapes", "b2r", seed=0, block="fake", lr=1e-2).out_dir(tmp_path).parent.glob("b2r__*seed0"))
    rv, rt = D8.run_arrays("b2r", d, val), D8.run_arrays("b2r", d, val, "test", table.take("test"))
    assert np.array_equal(rv["thr"], rt["thr"]) and rt["onll"].shape == (40, K)
