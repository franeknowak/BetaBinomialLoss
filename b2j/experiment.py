"""Tuning, selection and final runs for one dataset (PLAN §1.7-1.8).

  python -m b2j.experiment tune <dataset> [--lrs 5e-3]      tuning grid (seed 0); results/tuning/<dataset>/...
  python -m b2j.experiment select <dataset>                 per loss: setting with the lowest val loss; edge flags
  python -m b2j.experiment final <dataset> [--seeds 1 2 3 4 5]

Tuning: every loss x learning rate x weight-decay level (3 x 4 x 3 = 36 runs), one seed (0); only run.json is kept.
A run whose run.json exists is skipped, so a job can be resubmitted. --lrs runs extra learning rates (grid
extension). Selection: lowest best_val_loss over all tuning runs of that loss (ties -> smaller lr, then the level
order); flags a chosen learning rate at the edge of the learning rates tried, and reports where the best evaluation
fell (the 25-epoch budget check). Final: the selected setting x seeds 1-5, saving outputs, heads and metrics.
"""
import argparse
import json
from pathlib import Path

import torch

from b2j.losses import LOSSES
from b2j.schedule import LRS, WD_LEVELS
from b2j.train import load_embeddings, save_final, train_run

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
TUNING_SEED, FINAL_SEEDS = 0, (1, 2, 3, 4, 5)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def tuning_dir(dataset, loss, lr, level, root=RESULTS):
    return Path(root) / "tuning" / dataset / loss / f"lr{lr:g}_wd{level:g}"


def tune(dataset, lrs=LRS, levels=WD_LEVELS, losses=tuple(LOSSES), root=RESULTS, epochs=25):
    dev = _device()
    data, labels = load_embeddings(dataset, device=dev)
    for loss in losses:
        for lr in lrs:
            for level in levels:
                d = tuning_dir(dataset, loss, lr, level, root)
                if (d / "run.json").exists():
                    print(f"skip {d} (done)", flush=True)
                    continue
                run, _ = train_run(dataset, data, labels, loss, lr, level, TUNING_SEED, dev, epochs)
                d.mkdir(parents=True, exist_ok=True)
                (d / "run.json").write_text(json.dumps(run, indent=1))
                print(f"{dataset} {loss} lr {lr:g} level {level:g} wd {run['wd']:.4g}: best val loss "
                      f"{run['best_val_loss']:.5f} at eval {run['best_val_loss_eval']} ({run['seconds']} s)", flush=True)


def select(dataset, root=RESULTS):
    sel = {}
    for loss in LOSSES:
        runs = [json.loads(p.read_text()) for p in (Path(root) / "tuning" / dataset / loss).glob("*/run.json")]
        if not runs:
            continue
        order = {lv: i for i, lv in enumerate(WD_LEVELS)}
        runs.sort(key=lambda r: (r["best_val_loss"], r["lr"], order.get(r["wd_level"], 99)))
        r = runs[0]
        lrs_tried = sorted({x["lr"] for x in runs})
        sel[loss] = {"lr": r["lr"], "wd_level": r["wd_level"], "wd": r["wd"], "best_val_loss": r["best_val_loss"],
                     "best_eval": r["best_val_loss_eval"], "best_epoch": r["best_val_loss_eval"] / r["evals_per_epoch"],
                     "lr_at_edge": "low" if r["lr"] == lrs_tried[0] else "high" if r["lr"] == lrs_tried[-1] else None,
                     "lrs_tried": lrs_tried, "n_runs": len(runs)}
        print(f"{dataset} {loss}: lr {r['lr']:g}, level {r['wd_level']:g} (wd {r['wd']:.4g}), best val loss "
              f"{r['best_val_loss']:.5f} at epoch {sel[loss]['best_epoch']:g}/{r['epochs']}"
              + (f"  ** lr at {sel[loss]['lr_at_edge']} edge **" if sel[loss]["lr_at_edge"] else ""), flush=True)
    out = Path(root) / "tuning" / dataset / "selection.json"
    out.write_text(json.dumps(sel, indent=1))
    return sel


def final(dataset, seeds=FINAL_SEEDS, root=RESULTS, epochs=25):
    sel = json.loads((Path(root) / "tuning" / dataset / "selection.json").read_text())
    dev = _device()
    data, labels = load_embeddings(dataset, device=dev)
    for loss, s in sel.items():
        for seed in seeds:
            d = Path(root) / "final" / dataset / loss / f"seed{seed}"
            if (d / "run.json").exists():
                print(f"skip {d} (done)", flush=True)
                continue
            run, states = train_run(dataset, data, labels, loss, s["lr"], s["wd_level"], seed, dev, epochs)
            save_final(d, dataset, data, labels, run, states, dev)
            print(f"{dataset} {loss} seed {seed}: checkpoints "
                  + ", ".join(f"{c} @ epoch {v['epoch']:g}" for c, v in run["checkpoints"].items())
                  + f" ({run['seconds']} s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tune", "select", "final"])
    ap.add_argument("dataset")
    ap.add_argument("--lrs", type=float, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(FINAL_SEEDS))
    a = ap.parse_args()
    if a.stage == "tune":
        tune(a.dataset, lrs=tuple(a.lrs) if a.lrs else LRS)
    elif a.stage == "select":
        select(a.dataset)
    else:
        final(a.dataset, seeds=tuple(a.seeds))
