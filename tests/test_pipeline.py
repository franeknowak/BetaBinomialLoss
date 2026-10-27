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
def test_binary_metrics_against_sklearn_and_by_hand():
    rng = np.random.default_rng(0)
    p, y = rng.random(200), (rng.random(200) < 0.3).astype(int)
    frac = rng.random(200)
    m = M.binary_metrics(p, y, frac, np.zeros(200))
    assert m["ap"] == pytest.approx(average_precision_score(y, p)) and m["auroc"] == pytest.approx(roc_auc_score(y, p))
    pred = p > 0.5
    tp, fp, fn, tn = (pred & (y == 1)).sum(), (pred & (y == 0)).sum(), (~pred & (y == 1)).sum(), (~pred & (y == 0)).sum()
    assert m["sensitivity"] == pytest.approx(tp / (tp + fn)) and m["specificity"] == pytest.approx(tn / (tn + fp))
    assert m["bacc"] == pytest.approx((tp / (tp + fn) + tn / (tn + fp)) / 2) and m["acc"] == pytest.approx((tp + tn) / 200)
    assert m["f1"] == pytest.approx(2 * tp / (2 * tp + fp + fn)) and m["mse"] == pytest.approx(np.mean((p - frac) ** 2))


def test_no_positive_predictions_gives_nan_f1():
    """The FER+ disgust example: 11 positives, nothing predicted positive."""
    y = np.r_[np.ones(11), np.zeros(3534)].astype(int)
    m = M.binary_metrics(np.full(len(y), 0.1), y, y.astype(float), np.zeros(len(y)))
    assert np.isnan(m["f1"]) and m["sensitivity"] == 0 and m["specificity"] == 1 and m["bacc"] == 0.5
    assert m["n_pred_pos"] == 0 and m["n_ref_pos"] == 11


def test_top1_ties_count_any():
    k = np.array([[4, 4, 2], [10, 0, 0], [1, 5, 5]])
    p = np.array([[0.2, 0.9, 0.1], [0.1, 0.8, 0.2], [0.9, 0.1, 0.2]])
    assert M.top1_accuracy(p, k) == pytest.approx(1 / 3)                    # tie hit, miss, miss


def test_compute_structure_and_macro():
    torch.manual_seed(0)
    out = torch.randn(50, 3, 1)
    k = torch.randint(0, 4, (50, 3)).float()
    n = torch.full_like(k, 3.0)
    r = M.compute("soft_bce", out, k, n, ["a", "b", "c"], "endoscapes")
    assert set(r["per_label"]) == {"a", "b", "c"} and "top1" not in r and r["n_images"] == 50
    assert r["macro"]["bacc"] == pytest.approx(np.mean([r["per_label"][x]["bacc"] for x in "abc"]))
    assert r["macro"]["bacc"] == pytest.approx(M.macro_bacc("soft_bce", out, k, n))
    assert "top1" in M.compute("b2", torch.randn(50, 3, 2), k, n, ["a", "b", "c"], "cifar10h")


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
    assert len(run["curve"]) == 8 and set(states) == {"bacc", "nll"}
    c = run["curve"]
    best_bacc = max(range(len(c)), key=lambda i: (c[i]["val_bacc"], -i))
    assert run["checkpoints"]["bacc"]["eval"] == best_bacc + 1                 # earliest of the best
    assert run["checkpoints"]["nll"]["eval"] == run["best_val_loss_eval"] == 1 + int(np.argmin([x["val_loss"] for x in c]))
    assert c[0]["lr"] == pytest.approx(1e-3 * S.lr_factors(2 * 5)[c[0]["step"] - 1])
    assert run["wd"] == pytest.approx(S.wd_for_level(0.25, 1e-3, 300, epochs=2))
    run2, _ = train_run("toy", data, labels, "b2", 1e-3, 0.25, seed=0, epochs=2)
    assert [x["val_loss"] for x in run2["curve"]] == [x["val_loss"] for x in c]   # same seed, same run
    run3, _ = train_run("toy", data, labels, "hard_bce", 1e-3, 1.0, seed=0, epochs=1)
    assert set(run3["checkpoints"]) == {"bacc"} and run3["wd"] == 0.0


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
        assert s["best_val_loss"] == min(r["best_val_loss"] for r in runs)
        assert s["lr_at_edge"] in ("low", "high")                                # only two lrs tried
    with pytest.raises(SystemExit):                                                      # lr at the edge: refused
        E.final("toy", seeds=(1,), root=root, epochs=1)
    E.final("toy", seeds=(1,), root=root, epochs=1, allow_edge=True)
    d = root / "final" / "toy" / "b2" / "seed1"
    run = json.loads((d / "run.json").read_text())
    assert set(run["checkpoints"]) == {"bacc", "nll"} and (d / "head_nll.pt").exists()
    out = json.loads((d / "outputs_nll.json").read_text())
    assert out["outputs"] == ["alpha0", "alpha1"] and {i["split"] for i in out["images"]} == {"val", "test"}
    assert len(out["images"]) == 120 and len(out["images"][0]["out"]) == 3 and out["images"][0]["n"] == [3, 3, 3]
    assert min(min(a) for i in out["images"] for a in i["out"]) >= 1e-3                 # alphas, not logits
    assert set(run["checkpoints"]["nll"]["metrics"]) == {"train", "val", "test"}
    assert set((root / "final" / "toy").iterdir()) == {root / "final" / "toy" / x for x in ("b2", "hard_bce", "soft_bce")}
    assert set(json.loads((root / "final" / "toy" / "soft_bce" / "seed1" / "run.json").read_text())["checkpoints"]) \
        == {"bacc", "softbce"}
    h = json.loads((root / "final" / "toy" / "hard_bce" / "seed1" / "outputs_bacc.json").read_text())
    assert h["outputs"] == ["logit"] and len(h["images"][0]["out"]) == 3 and len(h["images"][0]["out"][0]) == 1
    assert sorted(p.name for p in (root / "final" / "toy" / "hard_bce" / "seed1").iterdir()) == \
        ["head_bacc.pt", "outputs_bacc.json", "run.json"]


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
