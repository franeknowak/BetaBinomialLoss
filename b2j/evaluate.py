"""Recompute the phase-1 metrics (PLAN §1.9) of finished final runs from their saved head weights.

  python -m b2j.evaluate <dataset> [--seeds 1 2 3 4 5]

For each loss and seed: loads head_<ckpt>.pt and the cached embeddings, computes the metrics of b2j.metrics on train,
val and test, and writes metrics_<ckpt>.json next to run.json (run.json is left as it is). Hard BCE's runs are read
from results/final_hardbce/, the other losses from results/final/ (PLAN §1.9c). Also checks that the recomputed
val/test outputs match outputs_<ckpt>.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from b2j import metrics as M
from b2j.heads import MultiHead
from b2j.losses import LOSSES, alphas, make_loss
from b2j.train import CKPT, _outputs, load_embeddings

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
RUN_DIR = {"b2": "final", "soft_bce": "final", "hard_bce": "final_hardbce"}


def evaluate_run(d, loss, data, labels):
    c = CKPT[loss]
    _, width = make_loss(loss)
    model = MultiHead(data["train"][0].shape[1], len(labels), width)
    model.load_state_dict(torch.load(d / f"head_{c}.pt", map_location="cpu"))
    saved = {(i["split"], i["id"]): i["out"] for i in json.loads((d / f"outputs_{c}.json").read_text())["images"]}
    mets, max_diff = {}, 0.0
    for s in ("train", "val", "test"):
        X, k, n, ids = data[s]
        out = _outputs(model, X)
        mets[s] = M.compute(loss, out, k, n, labels)
        if s != "train":
            raw = torch.stack(alphas(out.double()), -1) if loss == "b2" else out.double()
            ref = np.array([saved[s, str(i)] for i in ids])
            max_diff = max(max_diff, float(np.abs(raw.numpy() - ref).max()))
    return {"checkpoint": c, "metrics": mets, "max_abs_diff_vs_saved_outputs": max_diff}


def main(dataset, seeds, root=RESULTS):
    data, labels = load_embeddings(dataset)
    for loss in LOSSES:
        for seed in seeds:
            d = Path(root) / RUN_DIR[loss] / dataset / loss / f"seed{seed}"
            res = evaluate_run(d, loss, data, labels)
            (d / f"metrics_{CKPT[loss]}.json").write_text(json.dumps(res, indent=1))
            t = res["metrics"]["test"]
            print(f"{dataset} {loss} seed {seed}: test RPS/n {t['macro']['rps']:.4f}, spread ratio "
                  f"{t['pooled']['spread_ratio']:.3f}, coverage gap {t['pooled']['coverage']['gap']:.4f}; "
                  f"max |diff| vs saved outputs {res['max_abs_diff_vs_saved_outputs']:.2e}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    main(a.dataset, a.seeds)
