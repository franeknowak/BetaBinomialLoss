"""Final runs with m raters per image on train and val (drawn without replacement from each image's panel,
one fixed draw per m, shared by every loss and seed); test keeps the full panel. Settings = the full-panel selection
(results/tuning/<ds>/selection.json). Writes results/explore/sub/n<m>/final{,_hardbce}/<ds>/<loss>/seed<s>/.

  python -m explore.subsample <dataset> <m> [--seeds 1 2 3 4 5]
"""
import argparse
import json

import numpy as np
import torch
from b2j.experiment import RESULTS, _device
from b2j.train import load_embeddings, save_final, train_run
from explore import sub_root


def subsample(k, n, m, rng):
    """k, n: [N, K]. Single-choice panels (k sums to n over labels) draw m raters jointly, else per label."""
    k, n = k.astype(np.int64), n.astype(np.int64)
    out_k, out_n = np.empty_like(k), np.minimum(n, m)
    joint = k.shape[1] > 1 and np.array_equal(k.sum(1), n[:, 0])
    for i in range(len(k)):
        if joint:
            out_k[i] = rng.multivariate_hypergeometric(k[i], int(out_n[i, 0]))
        else:
            for j in range(k.shape[1]):
                out_k[i, j] = rng.hypergeometric(k[i, j], n[i, j] - k[i, j], out_n[i, j])
    return out_k.astype(np.float32), out_n.astype(np.float32)


def main(ds, m, seeds):
    dev = _device()
    data, labels = load_embeddings(ds, device=dev)
    rng = np.random.default_rng(1000 + m)
    for split in ("train", "val"):
        X, k, n, ids = data[split]
        ks, ns = subsample(k.cpu().numpy(), n.cpu().numpy(), m, rng)
        data[split] = (X, torch.as_tensor(ks, device=dev), torch.as_tensor(ns, device=dev), ids)
        print(f"{split}: mean n {n.mean():.2f} -> {ns.mean():.2f}; unanimous share (label 0) "
              f"{((k[:, 0] == 0) | (k[:, 0] == n[:, 0])).float().mean():.3f} -> "
              f"{((ks[:, 0] == 0) | (ks[:, 0] == ns[:, 0])).mean():.3f}", flush=True)
    sel = json.loads((RESULTS / "tuning" / ds / "selection.json").read_text())
    root = sub_root(m)
    for loss, s in sel.items():
        for seed in seeds:
            d = root / ("final_hardbce" if loss == "hard_bce" else "final") / ds / loss / f"seed{seed}"
            if (d / "run.json").exists():
                print(f"skip {d}", flush=True)
                continue
            run, states = train_run(ds, data, labels, loss, s["lr"], s["wd_level"], seed, dev)
            run["subsample_m"] = m
            if run["status"] != "ok":
                d.mkdir(parents=True, exist_ok=True)
                (d / "run.json").write_text(json.dumps(run, indent=1))
                print(f"{ds} m={m} {loss} seed {seed}: DIVERGED", flush=True)
                continue
            save_final(d, ds, data, labels, run, states, dev)
            print(f"{ds} m={m} {loss} seed {seed}: " +
                  ", ".join(f"{c} @ epoch {v['epoch']:g}" for c, v in run["checkpoints"].items()), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("m", type=int)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    main(a.dataset, a.m, a.seeds)
