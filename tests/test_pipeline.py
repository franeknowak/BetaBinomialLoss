"""Tests for the phase-1 pipeline: BCE losses, vote NLL, schedule / weight decay, metrics, trainer, experiment.
Run: python -m pytest tests/test_pipeline.py -q
"""
import json
import math

import numpy as np
import pytest
import torch
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from b2j import losses as L
from b2j import metrics as M
from b2j import schedule as S


# ---------------------------------------------------------------- losses
def test_bce_losses_worked_example():
    """The worked example agreed with the user: k = 2 of n = 3."""
    k, n = torch.tensor([[2.0]]), torch.tensor([[3.0]])
    hard = L.HardBCELoss()(torch.tensor([[[1.2]]]), k, n).item()          # target 1, p = sigmoid(1.2) = 0.769
    soft = L.SoftBCELoss()(torch.tensor([[[0.7]]]), k, n).item()          # target 2/3, p = 0.668
    assert hard == pytest.approx(-math.log(1 / (1 + math.exp(-1.2))), rel=1e-6) and hard == pytest.approx(0.263, abs=1e-3)
    assert soft == pytest.approx(0.636, abs=1e-3)
    a0, a1 = torch.tensor([2.0]).double(), torch.tensor([4.0]).double()
    assert L.beta_binomial_nll(a0, a1, torch.tensor([2.0]).double(), torch.tensor([3.0]).double()).item() == \
        pytest.approx(-math.log(120 / 336), rel=1e-9)


def test_bce_reduction_mean_over_images_sum_over_labels():
    torch.manual_seed(0)
    out, k = torch.randn(5, 3, 1), torch.randint(0, 4, (5, 3)).float()
    n = torch.full_like(k, 3.0)
    for loss, target in ((L.HardBCELoss(), (k > 1.5).float()), (L.SoftBCELoss(), k / 3)):
        ref = torch.nn.functional.binary_cross_entropy_with_logits(out[..., 0], target, reduction="none").mean(0).sum()
        torch.testing.assert_close(loss(out, k, n), ref)


def test_hard_target_is_strict_majority():
    k, n = torch.tensor([[1.0, 2.0, 5.0, 4.0]]), torch.tensor([[3.0, 3.0, 10.0, 7.0]])
    z = torch.zeros(1, 4, 1)
    per = L.HardBCELoss().per_sample(torch.full_like(z, 30.0), k, n)       # p ~ 1: loss ~ 0 iff target 1
    assert (per[0] < 1e-6).tolist() == [False, True, False, True]          # 5 of 10 is not a majority


def test_binomial_vote_nll_matches_scipy_and_is_finite():
    z = torch.tensor([[0.3, -2.0, 40.0]]).double()
    k, n = torch.tensor([[2.0, 0.0, 1.0]]).double(), torch.tensor([[3.0, 7.0, 3.0]]).double()
    ours = L.vote_nll("soft_bce", z[..., None], k, n).numpy()[0]
    p = 1 / (1 + np.exp(-z.numpy()[0]))
    np.testing.assert_allclose(ours[:2], -stats.binom.logpmf([2, 0], [3, 7], p[:2]), rtol=1e-9)
    assert np.isfinite(ours[2]) and ours[2] > 30                             # p ~ 1 but 1 of 3: large, finite


def test_vote_nll_b2_is_its_loss():
    out = torch.randn(4, 2, 2)
    k, n = torch.tensor([[0.0, 3], [1, 2], [2, 1], [3, 0]]), torch.full((4, 2), 3.0)
    torch.testing.assert_close(L.vote_nll("b2", out, k, n), L.B2Loss().per_sample(out.double(), k, n).double())


def test_predicted_fraction():
    out = torch.tensor([[[0.0, 1.0]]])
    a0, a1 = L.alphas(out.double())
    assert L.predicted_fraction("b2", out).item() == pytest.approx((a1 / (a0 + a1)).item())
    assert L.predicted_fraction("hard_bce", torch.tensor([[[0.0]]])).item() == 0.5


# ---------------------------------------------------------------- schedule and weight decay
def test_weight_decay_levels_match_agreed_values():
    """Values shown to the user (CIFAR-10H, 6,000 images, 94 steps/epoch)."""
    assert S.wd_for_level(1.0, 1e-3, 6000) == 0.0
    assert S.wd_for_level(0.75, 1e-3, 6000) == pytest.approx(0.245, abs=1e-3)
    assert S.wd_for_level(0.25, 1e-3, 6000) == pytest.approx(1.18, abs=5e-3)
    assert S.wd_for_level(0.25, 5e-5, 6000) == pytest.approx(23.6, abs=0.1)
    for lr in S.LRS:
        for level in (0.75, 0.25):
            for n in (6701, 6000, 1740, 2151, 28383):
                assert S.decay_left(S.wd_for_level(level, lr, n), lr, n) == pytest.approx(level, abs=1e-9)


def test_cosine_and_eval_points():
    f = S.lr_factors(100)
    assert f[0] == 1.0 and f[50] == pytest.approx(0.5) and f[-1] > 0 and np.all(np.diff(f) < 0)
    ev = S.eval_steps(6000)                                                  # 94 steps per epoch
    assert len(ev) == 100 and ev[-1] == 25 * 94 and ev[3] == 94 and len(set(ev)) == 100
    for n in (6701, 6000, 1740, 2151, 28383):
        assert len(S.eval_steps(n)) == 100


# ---------------------------------------------------------------- metrics
def test_label_metrics_against_sklearn_and_by_hand():
    rng = np.random.default_rng(0)
    p, y = rng.random(200), (rng.random(200) < 0.3).astype(int)
    frac, nll, rps = rng.random(200), rng.random(200), rng.random(200)
    m = M.label_metrics(p, y, frac, nll, rps)
    assert m["ap"] == pytest.approx(average_precision_score(y, p)) and m["auroc"] == pytest.approx(roc_auc_score(y, p))
    assert m["mse"] == pytest.approx(np.mean((p - frac) ** 2)) and m["vote_nll"] == pytest.approx(nll.mean())
    assert m["rps"] == pytest.approx(rps.mean()) and m["n_ref_pos"] == y.sum()
    m = M.label_metrics(p, np.zeros(200, int), frac, nll, rps)                 # one class only -> NaN ranking metrics
    assert np.isnan(m["ap"]) and np.isnan(m["auroc"])


def test_predictive_matches_scipy_and_variance_formula():
    """B²: Beta-Binomial(n; a1, a0) with the head's alphas; BCE: Binomial(n, sigmoid(logit)); Var(k/n) = Var(k)/n²."""
    torch.manual_seed(0)
    n = np.array([[3, 3], [7, 3]])
    out = torch.randn(2, 2, 2)
    m, var, F = M.predictive("b2", out, n)
    a0, a1 = (a.numpy() for a in L.alphas(out.double()))
    for i, j in np.ndindex(n.shape):
        d = stats.betabinom(n[i, j], a1[i, j], a0[i, j])
        assert np.allclose(F[i, j, :n[i, j] + 1], d.cdf(np.arange(n[i, j] + 1))) and np.allclose(F[i, j, n[i, j]:], 1)
        assert m[i, j] == pytest.approx(d.mean() / n[i, j]) and var[i, j] == pytest.approx(d.var() / n[i, j] ** 2)
    m, var, F = M.predictive("soft_bce", out[..., :1], n)
    d = stats.binom(7, m[1, 0])
    assert np.allclose(F[1, 0], d.cdf(np.arange(8))) and var[1, 0] == pytest.approx(d.var() / 49)


def test_rps_by_hand():
    """Worked example: Binomial(3, 0.3), F = (.343, .784, .973); k = 2 -> (.343² + .784² + (.973 − 1)²) / 3."""
    F = stats.binom(3, 0.3).cdf(np.arange(4))[None]
    want = (0.343 ** 2 + 0.784 ** 2 + (0.973 - 1) ** 2) / 3
    assert M.rps_over_n(F, np.array([2]), np.array([3]))[0] == pytest.approx(want)


def test_central_interval_worked_example():
    """The worked example agreed with the user (n = 3): B² (a1 = 0.6, a0 = 1.4) and soft BCE (p = 0.3) at X = 50%, 80%."""
    j = np.arange(4)
    Fb, Fs = stats.betabinom(3, 0.6, 1.4).cdf(j), stats.binom(3, 0.3).cdf(j)
    L_, U_, c = M.central_interval(Fb, 0.5)
    assert (L_, U_) == (0, 2) and c == pytest.approx(0.896, abs=1e-3)
    L_, U_, c = M.central_interval(Fs, 0.5)
    assert (L_, U_) == (0, 1) and c == pytest.approx(0.784, abs=1e-3)
    L_, U_, c = M.central_interval(Fb, 0.8)
    assert (L_, U_) == (0, 3) and c == pytest.approx(1.0)
    L_, U_, c = M.central_interval(Fs, 0.8)
    assert (L_, U_) == (0, 2) and c == pytest.approx(0.973, abs=1e-3)
    for X in M.LEVELS:                                                          # claimed is never below the level
        assert M.central_interval(Fb, X)[2] >= X and M.central_interval(Fs, X)[2] >= X


def test_calibrated_model_scores_one_and_zero():
    """k drawn from the predicted Beta-Binomial: spread ratio ≈ 1, hit rate ≈ claimed (gap ≈ 0)."""
    rng = np.random.default_rng(0)
    N, n = 200_000, 3
    a0, a1 = rng.uniform(0.3, 5, N), rng.uniform(0.3, 5, N)
    k = rng.binomial(n, rng.beta(a1, a0))
    nn_ = np.full(N, n)
    phi, m = a0 + a1, a1 / (a0 + a1)
    var = m * (1 - m) / n * (1 + (n - 1) / (phi + 1))
    F = stats.betabinom.cdf(np.arange(n + 1), n, a1[:, None], a0[:, None])
    assert M.spread_ratio(m, var, k, nn_) == pytest.approx(1, abs=0.02)
    assert M.coverage(F, k)["gap"] < 0.005
    # the same votes scored as a Binomial with the same mean: spread too narrow
    assert M.spread_ratio(m, m * (1 - m) / n, k, nn_) > 1.2


def test_compute_structure_and_macro():
    torch.manual_seed(0)
    k = torch.randint(0, 4, (50, 3)).float()
    n = torch.full_like(k, 3.0)
    for loss, w in (("soft_bce", 1), ("b2", 2)):
        out = torch.randn(50, 3, w)
        r = M.compute(loss, out, k, n, ["a", "b", "c"])
        assert set(r["per_label"]) == {"a", "b", "c"} and r["n_images"] == 50
        assert set(r["macro"]) == set(M.MACRO) and set(r["pooled"]) == {"spread_ratio", "coverage"}
        for x in M.MACRO:
            assert r["macro"][x] == pytest.approx(np.mean([r["per_label"][lab][x] for lab in "abc"]))
        assert r["per_label"]["a"]["vote_nll"] == pytest.approx(L.vote_nll(loss, out, k, n)[:, 0].mean().item())
        assert len(r["pooled"]["coverage"]["hit"]) == len(M.LEVELS)


# ---------------------------------------------------------------- trainer and experiment (CPU, random embeddings)
def _fake_cache(tmp_path, n_train=300, K=3, raters=3):
    rng = np.random.default_rng(0)
    N = n_train + 120
    split = np.array(["train"] * n_train + ["val"] * 60 + ["test"] * 60)
    emb = rng.normal(size=(N, 768)).astype(np.float32)
    theta = 1 / (1 + np.exp(-emb[:, :K] * 2))
    k = rng.binomial(raters, theta).astype(float)
    np.savez(tmp_path / "toy.npz", emb=emb, ids=np.array([f"v{i // 10}_{i}" for i in range(N)]), split=split,
             k=k, n=np.full_like(k, raters), labels=np.array(["C1", "C2", "C3"][:K]))
    return tmp_path


def test_train_run_checkpoints_and_curve(tmp_path):
    from b2j.train import load_embeddings, train_run
    data, labels = load_embeddings("toy", cache=_fake_cache(tmp_path))
    run, states = train_run("toy", data, labels, "b2", 1e-3, 0.25, seed=0, epochs=2)
    assert len(run["curve"]) == 8 and set(states) == {"nll"}
    c = run["curve"]
    assert run["checkpoints"]["nll"]["eval"] == run["best_val_loss_eval"] == 1 + int(np.argmin([x["val_loss"] for x in c]))
    assert c[0]["lr"] == pytest.approx(1e-3 * S.lr_factors(2 * 5)[c[0]["step"] - 1])
    assert run["wd"] == pytest.approx(S.wd_for_level(0.25, 1e-3, 300, epochs=2))
    run2, _ = train_run("toy", data, labels, "b2", 1e-3, 0.25, seed=0, epochs=2)
    assert [x["val_loss"] for x in run2["curve"]] == [x["val_loss"] for x in c]   # same seed, same run
    run3, _ = train_run("toy", data, labels, "hard_bce", 1e-3, 1.0, seed=0, epochs=1)
    assert set(run3["checkpoints"]) == {"hardbce"} and run3["wd"] == 0.0
    assert run3["checkpoints"]["hardbce"]["eval"] == run3["best_val_loss_eval"]


def test_train_and_val_loss_at_eval_are_eval_mode_full_passes(tmp_path):
    from b2j.heads import MultiHead
    from b2j.train import load_embeddings, train_run
    data, labels = load_embeddings("toy", cache=_fake_cache(tmp_path))
    run, states = train_run("toy", data, labels, "soft_bce", 1e-3, 1.0, seed=0, epochs=1)
    e = run["checkpoints"]["softbce"]["eval"]
    m = MultiHead(768, 3, 1)
    m.load_state_dict(states["softbce"])
    m.eval()
    with torch.no_grad():
        tr = L.SoftBCELoss()(m(data["train"][0]), data["train"][1], data["train"][2]).item()   # all 300 train images
        va = L.SoftBCELoss()(m(data["val"][0]), data["val"][1], data["val"][2]).item()
    assert tr == pytest.approx(run["curve"][e - 1]["train_loss"], rel=1e-6)
    assert va == pytest.approx(run["curve"][e - 1]["val_loss"], rel=1e-6) == pytest.approx(run["best_val_loss"], rel=1e-6)


def test_vote_nll_hard_bce_is_binomial():
    z = torch.tensor([[[0.4], [-1.0]]])
    k, n = torch.tensor([[1.0, 5.0]]), torch.tensor([[3.0, 9.0]])
    p = torch.sigmoid(z[..., 0].double()).numpy()[0]
    np.testing.assert_allclose(L.vote_nll("hard_bce", z, k, n).numpy()[0], -stats.binom.logpmf([1, 5], [3, 9], p),
                               rtol=1e-9)


def test_experiment_tune_select_final(tmp_path, monkeypatch):
    from b2j import experiment as E
    from b2j import train as T
    cache = _fake_cache(tmp_path)
    monkeypatch.setattr(T, "CACHE", cache)
    monkeypatch.setattr(E, "load_embeddings", lambda ds, device="cpu": T.load_embeddings(ds, cache=cache, device=device))
    root = tmp_path / "results"
    E.tune("toy", lrs=(1e-4, 1e-3), levels=(1.0, 0.25), root=root, epochs=1)
    assert len(list((root / "tuning" / "toy").glob("*/*/run.json"))) == 3 * 2 * 2
    sel = E.select("toy", root=root)
    for loss, s in sel.items():
        runs = [json.loads(p.read_text()) for p in (root / "tuning" / "toy" / loss).glob("*/run.json")]
        assert s["smoothed_val_loss"] == min(E.smoothed_min(r["curve"])[0] for r in runs)
        assert s["lr_at_edge"] in ("low", "high")                                # only two lrs tried
    with pytest.raises(SystemExit):                                                      # lr at the edge: refused
        E.final("toy", seeds=(1,), root=root, epochs=1)
    E.final("toy", seeds=(1,), root=root, epochs=1, allow_edge=True)
    d = root / "final" / "toy" / "b2" / "seed1"
    run = json.loads((d / "run.json").read_text())
    assert set(run["checkpoints"]) == {"nll"} and (d / "head_nll.pt").exists()
    out = json.loads((d / "outputs_nll.json").read_text())
    assert out["outputs"] == ["alpha0", "alpha1"] and {i["split"] for i in out["images"]} == {"val", "test"}
    assert len(out["images"]) == 120 and len(out["images"][0]["out"]) == 3 and out["images"][0]["n"] == [3, 3, 3]
    assert min(min(a) for i in out["images"] for a in i["out"]) >= 1e-3                 # alphas, not logits
    assert set(run["checkpoints"]["nll"]["metrics"]) == {"train", "val", "test"}
    assert set((root / "final" / "toy").iterdir()) == {root / "final" / "toy" / x for x in ("b2", "hard_bce", "soft_bce")}
    assert set(json.loads((root / "final" / "toy" / "soft_bce" / "seed1" / "run.json").read_text())["checkpoints"]) \
        == {"softbce"}
    h = json.loads((root / "final" / "toy" / "hard_bce" / "seed1" / "outputs_hardbce.json").read_text())
    assert h["outputs"] == ["logit"] and len(h["images"][0]["out"]) == 3 and len(h["images"][0]["out"][0]) == 1
    assert sorted(p.name for p in (root / "final" / "toy" / "hard_bce" / "seed1").iterdir()) == \
        ["head_hardbce.pt", "outputs_hardbce.json", "run.json"]
    E.final("toy", seeds=(2,), root=root, epochs=1, allow_edge=True, losses=("hard_bce",), out="rerun")
    assert [p.relative_to(root / "rerun").as_posix() for p in (root / "rerun").glob("*/*/*")] == ["toy/hard_bce/seed2"]


def test_evaluate_recomputes_saved_metrics(tmp_path, monkeypatch):
    """b2j.evaluate on finished final runs (hard BCE read from final_hardbce/) reproduces the metrics in run.json."""
    from b2j import evaluate as V
    from b2j import experiment as E
    from b2j import train as T
    cache = _fake_cache(tmp_path)
    monkeypatch.setattr(E, "load_embeddings", lambda ds, device="cpu": T.load_embeddings(ds, cache=cache, device=device))
    monkeypatch.setattr(V, "load_embeddings", lambda ds: T.load_embeddings(ds, cache=cache))
    root = tmp_path / "results"
    E.tune("toy", lrs=(1e-4,), levels=(1.0,), root=root, epochs=1)
    E.select("toy", root=root)
    E.final("toy", seeds=(1,), root=root, epochs=1, allow_edge=True, losses=["b2", "soft_bce"])
    E.final("toy", seeds=(1,), root=root, epochs=1, allow_edge=True, losses=["hard_bce"], out="final_hardbce")
    V.main("toy", seeds=(1,), root=root)
    for loss, sub in V.RUN_DIR.items():
        d = root / sub / "toy" / loss / "seed1"
        c = T.CKPT[loss]
        res = json.loads((d / f"metrics_{c}.json").read_text())
        saved = json.loads((d / "run.json").read_text())["checkpoints"][c]["metrics"]
        assert res["max_abs_diff_vs_saved_outputs"] < 1e-9
        for s in ("train", "val", "test"):
            assert res["metrics"][s]["macro"] == pytest.approx(saved[s]["macro"], nan_ok=True)
            assert res["metrics"][s]["pooled"]["spread_ratio"] == pytest.approx(saved[s]["pooled"]["spread_ratio"])


def test_resubmitted_tuning_skips_finished_runs(tmp_path, monkeypatch, capsys):
    from b2j import experiment as E
    from b2j import train as T
    cache = _fake_cache(tmp_path)
    monkeypatch.setattr(E, "load_embeddings", lambda ds, device="cpu": T.load_embeddings(ds, cache=cache, device=device))
    root = tmp_path / "results"
    E.tune("toy", lrs=(1e-3,), levels=(1.0,), losses=("hard_bce",), root=root, epochs=1)
    f = E.tuning_dir("toy", "hard_bce", 1e-3, 1.0, root) / "run.json"
    before = f.read_text()
    E.tune("toy", lrs=(1e-3,), levels=(1.0,), losses=("hard_bce",), root=root, epochs=1)
    assert "skip" in capsys.readouterr().out and f.read_text() == before


def test_grid_extension_only_for_the_given_loss(tmp_path, monkeypatch):
    from b2j import experiment as E
    from b2j import train as T
    cache = _fake_cache(tmp_path)
    monkeypatch.setattr(E, "load_embeddings", lambda ds, device="cpu": T.load_embeddings(ds, cache=cache, device=device))
    root = tmp_path / "results"
    E.tune("toy", lrs=(1e-4, 1e-3), levels=(1.0,), root=root, epochs=1)
    E.tune("toy", lrs=(5e-3,), levels=(1.0, 0.75, 0.25), losses=("b2",), root=root, epochs=1)
    tried = {loss: sorted(p.parent.name for p in (root / "tuning" / "toy" / loss).glob("*/run.json"))
             for loss in ("hard_bce", "soft_bce", "b2")}
    assert tried["hard_bce"] == tried["soft_bce"] == ["lr0.0001_wd1", "lr0.001_wd1"]
    assert tried["b2"] == ["lr0.0001_wd1", "lr0.001_wd1", "lr0.005_wd0.25", "lr0.005_wd0.75", "lr0.005_wd1"]
    sel = E.select("toy", root=root)
    assert sel["b2"]["lrs_tried"] == [1e-4, 1e-3, 5e-3] and sel["hard_bce"]["lrs_tried"] == [1e-4, 1e-3]


def test_smoothed_selection_ignores_a_single_low_evaluation(tmp_path):
    from b2j import experiment as E
    root = tmp_path / "results"

    def write(lr, vals):
        d = E.tuning_dir("toy", "b2", lr, 1.0, root)
        d.mkdir(parents=True)
        curve = [{"eval": i + 1, "val_loss": v} for i, v in enumerate(vals)]
        (d / "run.json").write_text(json.dumps({"lr": lr, "wd_level": 1.0, "wd": 0.0, "status": "ok", "epochs": 25,
                                                "evals_per_epoch": 4, "best_val_loss": min(vals),
                                                "best_val_loss_eval": vals.index(min(vals)) + 1, "curve": curve}))
    write(1e-3, [1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0])          # one lucky dip: raw minimum 0.5
    write(1e-4, [1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])          # steady lower curve
    assert E.smoothed_min([{"eval": i + 1, "val_loss": v} for i, v in enumerate([3, 1, 2])], window=5) == (2.0, 1)
    s = E.select("toy", root=root)["b2"]
    assert s["lr"] == 1e-4 and s["smoothed_val_loss"] == pytest.approx(0.8) and s["best_val_loss"] == 0.8


def test_diverged_run_is_saved_and_excluded(tmp_path, monkeypatch):
    from b2j import experiment as E
    from b2j import train as T
    cache = _fake_cache(tmp_path)
    monkeypatch.setattr(E, "load_embeddings", lambda ds, device="cpu": T.load_embeddings(ds, cache=cache, device=device))
    real = T.make_loss

    def exploding(name):                                     # the lr-1e-3 run's loss turns NaN after 3 calls
        fn, w = real(name)
        calls = {"n": 0}

        def f(out, k, n):
            calls["n"] += 1
            v = fn(out, k, n)
            return v * float("nan") if exploding.on and calls["n"] > 3 else v
        return f, w
    monkeypatch.setattr(T, "make_loss", exploding)
    root = tmp_path / "results"
    exploding.on = False
    E.tune("toy", lrs=(1e-4,), levels=(1.0,), losses=("soft_bce",), root=root, epochs=1)
    exploding.on = True
    E.tune("toy", lrs=(1e-3,), levels=(1.0,), losses=("soft_bce",), root=root, epochs=1)    # job carries on
    bad = json.loads((E.tuning_dir("toy", "soft_bce", 1e-3, 1.0, root) / "run.json").read_text())
    assert bad["status"] == "diverged" and "non-finite training loss" in bad["error"] and bad["checkpoints"] == {}
    assert bad["best_val_loss"] is None and len(bad["curve"]) >= 1                      # partial curve kept
    sel = E.select("toy", root=root)["soft_bce"]
    assert sel["lr"] == 1e-4 and sel["n_diverged"] == 1


def test_adamw_leaves_the_planned_fraction_of_a_weight():
    """Real torch AdamW with zero gradients under our cosine schedule: only decay acts."""
    n, lr, level = 1740, 1e-3, 0.25
    w = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.AdamW([w], lr=lr, weight_decay=S.wd_for_level(level, lr, n))
    for f in S.lr_factors(S.EPOCHS * S.steps_per_epoch(n)):
        opt.param_groups[0]["lr"] = lr * float(f)
        w.grad = torch.zeros(3)
        opt.step()
    assert w.detach().numpy() == pytest.approx(level, abs=1e-4)


# ---------------------------------------------------------------- embedding
def test_cls_embedding_is_token0_after_final_norm():
    import timm
    from b2j.embed import MODEL, cls_embedding
    m = timm.create_model(MODEL, pretrained=False, num_classes=0).eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        f = m.forward_features(x)
        e = cls_embedding(m, x)
    assert e.shape == (2, 768) and torch.equal(e, f[:, 0])
    assert m.pretrained_cfg["input_size"] == (3, 256, 256)
