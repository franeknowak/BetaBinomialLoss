"""Module g: tuning grid, choice of head lr / wd per loss family, and the 5-seed pilot block (plan §4.4 g).
Validation only.

  python -m b2j.pilot choose    # after block `tune`  -> results/journal/tune/chosen.json + tune_report.md

The `pilot` block (finished, job 7522400) is kept as a validation preview only. It used to set equivalence
margins; margins were dropped by 05_plan amendment 2026-10-16 #1 (results/journal/pilot/margins.md on the
cluster is historical). See journal_plan/specs/g_pilot.md.
"""
import json
import sys
from pathlib import Path

from b2j.runs import RESULTS, RunSpec

DATASETS = ("endoscapes", "cifar10h", "mhist")
PILOT_DATASETS = ("endoscapes", "cifar10h")          # the datasets where claims are decided
FAMILIES = ("hard_bce", "soft_bce", "b2r")
FAMILY = {"hard_bce": "hard_bce", "hard_bce_w": "hard_bce", "soft_bce": "soft_bce", "dup": "soft_bce",
          "b2r": "b2r", "b2r_w": "b2r", "b2r_kl": "b2r", "b2r_linear": "b2r", "b2m": "b2r"}
GRID = [(lr, wd) for lr in (1e-4, 3e-4, 1e-3) for wd in (1e-4, 1e-2)]
MAP_TIE = 1e-3                    # within this of the best val mAP, the lower val vote NLL wins (CIFAR-10H: mAP at ceiling)
LATE = 0.8                        # a chosen evaluation after 80% of the budget is flagged
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
                         "| dataset | family | lr | wd | val mAP (A) | val NLL/vote (B) | A epoch | B epoch | chosen | late |",
                         "|---|---|---|---|---|---|---|---|---|---|"]
    for ds in DATASETS:
        chosen[ds] = {}
        for fam in FAMILIES:
            cands = []
            for lr, wd in GRID:
                spec = RunSpec(ds, fam, lr=lr, wd=wd, block="tune")
                s = json.loads((spec.out_dir(root) / "summary.json").read_text())
                ep = s["config"]["epochs"]
                cands.append({"lr": lr, "wd": wd, "map": s["A"]["val_map"], "nll": s["B"]["val_nll_per_vote"],
                              "A_epoch": s["A"]["epoch"], "B_epoch": s["B"]["epoch"],
                              "late": max(s["A"]["epoch"], s["B"]["epoch"]) > LATE * ep})
            top = max(c["map"] for c in cands)
            best = min((c for c in cands if c["map"] >= top - MAP_TIE), key=lambda c: c["nll"])
            chosen[ds][fam] = best
            for c in cands:
                lines.append(f"| {ds} | {fam} | {c['lr']:g} | {c['wd']:g} | {c['map']:.4f} | {c['nll']:.4f} | "
                             f"{c['A_epoch']:.2f} | {c['B_epoch']:.2f} | {'**yes**' if c is best else ''} | "
                             f"{'late' if c['late'] else ''} |")
    late = [f"{ds}/{fam}" for ds in chosen for fam, c in chosen[ds].items() if c["late"]]
    lines += ["", "Chosen settings with a checkpoint after 80% of the budget: " + (", ".join(late) if late else "none")]
    path = chosen_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chosen, indent=1))
    (path.parent / "tune_report.md").write_text("\n".join(lines) + "\n")
    return chosen, late


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "choose":
        print(choose()[1])
        print((chosen_path().parent / "tune_report.md").read_text())
    else:
        raise SystemExit("usage: python -m b2j.pilot choose")
