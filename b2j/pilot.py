"""Module g: tuning grid, choice of head lr / wd per loss family, and the 5-seed pilot block (plan §4.4 g).
Validation only.

  python -m b2j.pilot choose    # after block `tune`  -> results/journal/tune/chosen.json + tune_report.md
                                #   + curves_<dataset>.png: train-subset and validation mAP / NLL per evaluation

The `pilot` block (finished, job 7522400) is kept as a validation preview only. It used to set equivalence
margins; margins were dropped by 05_plan amendment 2026-10-16 #1 (results/journal/pilot/margins.md on the
cluster is historical). See journal_plan/specs/g_pilot.md.
"""
import json
import sys
from pathlib import Path

from b2j.runs import RESULTS, RunSpec

DATASETS = ("endoscapes", "cifar10h", "mhist", "amibr", "ferplus")   # amibr, ferplus added in round D
PILOT_DATASETS = ("endoscapes", "cifar10h")          # the datasets where claims are decided
FAMILIES = ("hard_bce", "soft_bce", "b2r")
FAMILY = {"hard_bce": "hard_bce", "hard_bce_w": "hard_bce", "soft_bce": "soft_bce", "dup": "soft_bce", "soft_bce_n": "soft_bce",
          "b2r": "b2r", "b2r_w": "b2r", "b2r_kl": "b2r", "b2r_linear": "b2r", "b2m": "b2r",
          "b2ms": "b2r", **{f"b2ms_{t}{lam:g}": "b2r" for t in ("shrink", "var") for lam in (0.03, 0.3, 3.0)}}   # round D: B²-r's lr / wd
GRID = [(lr, wd) for lr in (1e-4, 3e-4, 1e-3) for wd in (1e-4, 1e-2)]
MAP_TIE = 1e-3                    # within this of the best val mAP, the lower val vote NLL wins (CIFAR-10H: mAP at ceiling)
SEEDS = range(5)


def chosen_path(root=RESULTS) -> Path:
    return Path(root) / "tune" / "chosen.json"


# ---------------------------------------------------------------- blocks
def tune_specs():
    return [RunSpec(ds, fam, lr=lr, wd=wd, block="tune") for ds in DATASETS for fam in FAMILIES for lr, wd in GRID]


def hyper(dataset: str, variant: str, root=RESULTS) -> dict:
    """Chosen {lr, wd} for a variant (inherited from its loss family)."""
    chosen = json.loads(chosen_path(root).read_text())
    c = chosen[dataset][FAMILY[variant]]
    return {"lr": c["lr"], "wd": c["wd"]}


def pilot_specs(root=RESULTS):
    return [RunSpec(ds, v, seed=s, block="pilot", **hyper(ds, v, root)) for ds in PILOT_DATASETS
            for v in ("soft_bce", "b2r") for s in SEEDS]


# ---------------------------------------------------------------- choice
def choose(root=RESULTS):
    chosen, lines = {}, ["# Tuning (validation only)", "",
                         "| dataset | family | lr | wd | val mAP (A) | val NLL/vote (B) | A epoch | B epoch | chosen |",
                         "|---|---|---|---|---|---|---|---|---|"]
    for ds in DATASETS:
        if not all((RunSpec(ds, f, lr=lr, wd=wd, block="tune").out_dir(root) / "summary.json").exists()
                   for f in FAMILIES for lr, wd in GRID):
            print(f"choose: tuning of {ds} incomplete, skipped")
            continue
        chosen[ds] = {}
        for fam in FAMILIES:
            cands = []
            for lr, wd in GRID:
                spec = RunSpec(ds, fam, lr=lr, wd=wd, block="tune")
                s = json.loads((spec.out_dir(root) / "summary.json").read_text())
                cands.append({"lr": lr, "wd": wd, "map": s["A"]["val_map"], "nll": s["B"]["val_nll_per_vote"],
                              "A_epoch": s["A"]["epoch"], "B_epoch": s["B"]["epoch"]})
            top = max(c["map"] for c in cands)
            best = min((c for c in cands if c["map"] >= top - MAP_TIE), key=lambda c: c["nll"])
            chosen[ds][fam] = best
            for c in cands:
                lines.append(f"| {ds} | {fam} | {c['lr']:g} | {c['wd']:g} | {c['map']:.4f} | {c['nll']:.4f} | "
                             f"{c['A_epoch']:.2f} | {c['B_epoch']:.2f} | {'**yes**' if c is best else ''} |")
    path = chosen_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chosen, indent=1))
    (path.parent / "tune_report.md").write_text("\n".join(lines) + "\n")
    return chosen


def curves(root=RESULTS):
    """Per dataset: train-subset vs validation mAP and NLL/vote against epoch for each family's chosen setting,
    with the A and B checkpoints marked. Runs trained before the probe existed are drawn without the train curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chosen = json.loads(chosen_path(root).read_text())
    made = []
    for ds in (d for d in DATASETS if d in chosen):
        fig, ax = plt.subplots(2, len(FAMILIES), figsize=(3.2 * len(FAMILIES), 5), squeeze=False)
        for j, fam in enumerate(FAMILIES):
            c = chosen[ds][fam]
            d = RunSpec(ds, fam, lr=c["lr"], wd=c["wd"], block="tune").out_dir(root)
            log = [json.loads(l) for l in (d / "log.jsonl").read_text().splitlines()]
            ep = [r["epoch"] for r in log]
            for i, key in enumerate(("map", "nll_per_vote")):
                a = ax[i][j]
                a.plot(ep, [r[f"val_{key}"] for r in log], color="#2a78d6", lw=1.4, label="validation")
                if f"train_{key}" in log[0]:
                    a.plot(ep, [r[f"train_{key}"] for r in log], color="#eb6834", lw=1.4, label="training subset")
                for cp, ls in (("A", "--"), ("B", ":")):
                    a.axvline(c[f"{cp}_epoch"], color="#8a8980", lw=0.9, ls=ls)
                a.set(xlabel="epoch", ylabel={"map": "mAP", "nll_per_vote": "vote NLL / vote"}[key])
                a.grid(True, color="#e6e5e0", lw=0.6)
            ax[0][j].set_title(f"{fam} (lr {c['lr']:g}; A --, B :)", fontsize=9)
        ax[0][0].legend(frameon=False, fontsize=7)
        fig.suptitle(f"{ds}: chosen settings, tuning runs (seed 0)", fontsize=10)
        fig.tight_layout()
        out = chosen_path(root).parent / f"curves_{ds}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        made.append(out)
    return made


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "choose":
        choose()
        print((chosen_path().parent / "tune_report.md").read_text())
        print("\n".join(map(str, curves())))
    else:
        raise SystemExit("usage: python -m b2j.pilot choose")
