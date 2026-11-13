"""Endoscapes: train on frames the three raters agree on, validate and test on frames they split on.

One model per criterion (C1, C2, C3). For criterion c:
  train = train frames with k_c in {0, 3}  (unanimous on c)
  val   = val frames with k_c in {1, 2}    (split on c)
  test  = test frames with k_c in {1, 2}   (split on c)
Learning rate and weight-decay level are the phase-1 Endoscapes choices of each loss (selection.json); the weight
decay itself is solved for the smaller training set, as for every run. Checkpoint: lowest val loss of the model's
own loss, as in phase 1. Outputs, heads and metrics as for the final runs (train.save_final).

  python -m b2j.unanimous C1 [--seeds 1 2 3 4 5] [--losses b2 soft_bce hard_bce] [--out unanimous]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from b2j.experiment import FINAL_SEEDS, RESULTS, _device
from b2j.losses import LOSSES
from b2j.train import load_embeddings, save_final, train_run

DATASET = "endoscapes"
CRITERIA = ("C1", "C2", "C3")
KEEP = {"train": "unanimous", "val": "split", "test": "split"}


def criterion_data(data, labels, criterion):
    """Restrict every split to one criterion (K = 1) and to the frames named in KEEP."""
    j = labels.index(criterion)
    out = {}
    for s, (X, k, n, ids) in data.items():
        kj, nj = k[:, j], n[:, j]
        unanimous = (kj == 0) | (kj == nj)
        m = unanimous if KEEP[s] == "unanimous" else ~unanimous
        mi = np.asarray(m.cpu())
        out[s] = (X[m], k[m][:, j:j + 1], n[m][:, j:j + 1], ids[mi])
    return out, [criterion]


def run(criterion, seeds=FINAL_SEEDS, losses=tuple(LOSSES), root=RESULTS, out="unanimous", epochs=25):
    sel = json.loads((Path(root) / "tuning" / DATASET / "selection.json").read_text())
    dev = _device()
    full, labels = load_embeddings(DATASET, device=dev)
    data, lab = criterion_data(full, labels, criterion)
    print(criterion, {s: len(v[3]) for s, v in data.items()}, flush=True)
    for loss in losses:
        s = sel[loss]
        for seed in seeds:
            d = Path(root) / out / DATASET / criterion / loss / f"seed{seed}"
            if (d / "run.json").exists():
                print(f"skip {d} (done)", flush=True)
                continue
            r, states = train_run(DATASET, data, lab, loss, s["lr"], s["wd_level"], seed, dev, epochs)
            r["setup"] = {"criterion": criterion, "keep": KEEP}
            if r["status"] != "ok":
                d.mkdir(parents=True, exist_ok=True)
                (d / "run.json").write_text(json.dumps(r, indent=1))
                print(f"{criterion} {loss} seed {seed}: DIVERGED ({r['error']})", flush=True)
                continue
            save_final(d, DATASET, data, lab, r, states, dev)
            print(f"{criterion} {loss} seed {seed}: lr {s['lr']:g} level {s['wd_level']:g} wd {r['wd']:.4g}, "
                  + ", ".join(f"{c} @ epoch {v['epoch']:g}" for c, v in r["checkpoints"].items())
                  + f" ({r['seconds']} s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("criterion", choices=CRITERIA)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(FINAL_SEEDS))
    ap.add_argument("--losses", nargs="+", choices=list(LOSSES), default=list(LOSSES))
    ap.add_argument("--out", default="unanimous")
    a = ap.parse_args()
    run(a.criterion, seeds=tuple(a.seeds), losses=tuple(a.losses), out=a.out)
