"""Module g: tuning grid, choice of head lr / wd per loss family, and the 5-seed pilot block (plan §4.4 g).
Validation only.

  python -m b2j.pilot choose    # after block `tune`  -> results/journal/tune/chosen.json + tune_report.md
                                #   + curves_<dataset>.png: train-subset and validation mAP / NLL per evaluation
  python -m b2j.pilot choose_r2 # after block `r2_tune` -> results/journal/r2_tune/chosen.json + tune_report.md

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
FAMILY = {"hard_bce": "hard_bce", "hard_bce_w": "hard_bce", "soft_bce": "soft_bce", "dup": "soft_bce", "soft_bce_n": "soft_bce", "soft_ce": "soft_bce", "dm": "b2r", "b2r_joint": "b2r",
          "b2r": "b2r", "b2r_w": "b2r", "b2r_kl": "b2r", "b2r_linear": "b2r", "b2m": "b2r",
          "b2ms": "b2r", **{f"b2ms_{t}{lam:g}": "b2r" for t in ("shrink", "var") for lam in (0.03, 0.3, 3.0)}}   # round D: B²-r's lr / wd
FAMILY.update({"b2rba": "b2rba", "soft_bce_ba": "soft_bce_ba"})   # D8: tuned on their own (block tune_ba, same grid)
BA_FAMILIES = ("soft_bce_ba", "b2rba")
BA_DATASETS = ("endoscapes", "amibr")                              # the datasets with rater identities
GRID = [(lr, wd) for lr in (1e-4, 3e-4, 1e-3) for wd in (1e-4, 1e-2)]
MAP_TIE = 1e-3                    # within this of the best val mAP, the lower val vote NLL wins (CIFAR-10H: mAP at ceiling)
SEEDS = range(5)
R2_DATASETS = ("endoscapes", "cifar10h")   # plan §4.4 R2
R2_ELR = (1e-5, 3e-6)                      # the 2 encoder lrs of the R2 tuning block (head lr / wd from frozen tuning)


def chosen_path(root=RESULTS, block="tune") -> Path:
    return Path(root) / block / "chosen.json"


# ---------------------------------------------------------------- blocks
def tune_specs():
    return [RunSpec(ds, fam, lr=lr, wd=wd, block="tune") for ds in DATASETS for fam in FAMILIES for lr, wd in GRID]


def tune_ba_specs():
    """D8: the baselines' tuning grid for the two rater-aware variants (fairness: same grid size, same choice rule)."""
    return [RunSpec(ds, fam, lr=lr, wd=wd, block="tune_ba") for ds in BA_DATASETS for fam in BA_FAMILIES
            for lr, wd in GRID]


def r2_families(dataset):
    """R2 fine-tuning families: hard BCE, soft BCE, B²-r everywhere; B²-rba and its control soft BCE-ba where the
    rater of each vote is known (B²-final, §5 re-commit 2026-10-22)."""
    return FAMILIES + (BA_FAMILIES if dataset in BA_DATASETS else ())


def r2_tune_specs(root=RESULTS):
    """R2 tuning (plan §4.4): 1 seed, 5 epochs, each family × R2_ELR, head lr / wd as chosen frozen. Same grid for
    every family (fairness rule)."""
    return [RunSpec(ds, fam, regime="finetune", encoder_lr=elr, block="r2_tune", **hyper(ds, fam, root))
            for ds in R2_DATASETS for fam in r2_families(ds) for elr in R2_ELR]


def hyper_ft(dataset: str, variant: str, root=RESULTS) -> dict:
    """Chosen {lr, wd, encoder_lr} for a fine-tuned variant (block r2_tune; inherited by loss family)."""
    c = json.loads(chosen_path(root, "r2_tune").read_text())[dataset][FAMILY[variant]]
    return {"lr": c["lr"], "wd": c["wd"], "encoder_lr": c["encoder_lr"]}


def choose_r2(root=RESULTS):
    """Per dataset and family: the encoder lr with the best validation mAP (A), ties within MAP_TIE broken by the
    validation NLL (B) — the rule of `choose`."""
    chosen, lines = {}, ["# R2 tuning: encoder lr (validation only; block r2_tune)", "",
                         "| dataset | family | encoder lr | head lr | wd | val mAP (A) | val NLL/vote (B) | A epoch | "
                         "B epoch | chosen |", "|---|---|---|---|---|---|---|---|---|---|"]
    specs = r2_tune_specs(root)
    for ds in R2_DATASETS:
        mine = [x for x in specs if x.dataset == ds]
        if not all((x.out_dir(root) / "summary.json").exists() for x in mine):
            print(f"choose_r2: tuning of {ds} incomplete, skipped")
            continue
        chosen[ds] = {}
        for fam in r2_families(ds):
            cands = []
            for x in (y for y in mine if y.variant == fam):
                sm = json.loads((x.out_dir(root) / "summary.json").read_text())
                cands.append({"encoder_lr": x.encoder_lr, "lr": x.lr, "wd": x.wd, "map": sm["A"]["val_map"],
                              "nll": sm["B"]["val_nll_per_vote"], "A_epoch": sm["A"]["epoch"],
                              "B_epoch": sm["B"]["epoch"]})
            top = max(c["map"] for c in cands)
            best = min((c for c in cands if c["map"] >= top - MAP_TIE), key=lambda c: c["nll"])
            chosen[ds][fam] = best
            for c in cands:
                lines.append(f"| {ds} | {fam} | {c['encoder_lr']:g} | {c['lr']:g} | {c['wd']:g} | {c['map']:.4f} | "
                             f"{c['nll']:.4f} | {c['A_epoch']:.2f} | {c['B_epoch']:.2f} | "
                             f"{'**yes**' if c is best else ''} |")
    path = chosen_path(root, "r2_tune")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chosen, indent=1))
    (path.parent / "tune_report.md").write_text("\n".join(lines) + "\n")
    return chosen


def hyper(dataset: str, variant: str, root=RESULTS) -> dict:
    """Chosen {lr, wd} for a variant (inherited from its loss family)."""
    fam = FAMILY[variant]
    chosen = json.loads(chosen_path(root, "tune_ba" if fam in BA_FAMILIES else "tune").read_text())
    c = chosen[dataset][fam]
    return {"lr": c["lr"], "wd": c["wd"]}


def pilot_specs(root=RESULTS):
    return [RunSpec(ds, v, seed=s, block="pilot", **hyper(ds, v, root)) for ds in PILOT_DATASETS
            for v in ("soft_bce", "b2r") for s in SEEDS]


# ---------------------------------------------------------------- choice
def choose(root=RESULTS, block="tune", datasets=None, families=None):
    """Per dataset and family: the best validation mAP (A), ties within MAP_TIE broken by validation NLL (B).
    block 'tune_ba' (D8): the rater-aware families; their NLL is on the ordered votes."""
    chosen, lines = {}, [f"# Tuning (validation only; block {block})", "",
                         "| dataset | family | lr | wd | val mAP (A) | val NLL/vote (B) | A epoch | B epoch | chosen |",
                         "|---|---|---|---|---|---|---|---|---|"]
    datasets, families = datasets or DATASETS, families or FAMILIES
    for ds in datasets:
        if not all((RunSpec(ds, f, lr=lr, wd=wd, block=block).out_dir(root) / "summary.json").exists()
                   for f in families for lr, wd in GRID):
            print(f"choose: tuning of {ds} incomplete, skipped")
            continue
        chosen[ds] = {}
        for fam in families:
            cands = []
            for lr, wd in GRID:
                spec = RunSpec(ds, fam, lr=lr, wd=wd, block=block)
                s = json.loads((spec.out_dir(root) / "summary.json").read_text())
                cands.append({"lr": lr, "wd": wd, "map": s["A"]["val_map"], "nll": s["B"]["val_nll_per_vote"],
                              "A_epoch": s["A"]["epoch"], "B_epoch": s["B"]["epoch"]})
            top = max(c["map"] for c in cands)
            best = min((c for c in cands if c["map"] >= top - MAP_TIE), key=lambda c: c["nll"])
            chosen[ds][fam] = best
            for c in cands:
                lines.append(f"| {ds} | {fam} | {c['lr']:g} | {c['wd']:g} | {c['map']:.4f} | {c['nll']:.4f} | "
                             f"{c['A_epoch']:.2f} | {c['B_epoch']:.2f} | {'**yes**' if c is best else ''} |")
    path = chosen_path(root, block)
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
    elif cmd == "choose_ba":
        choose(block="tune_ba", datasets=BA_DATASETS, families=BA_FAMILIES)
        print((chosen_path(block="tune_ba").parent / "tune_report.md").read_text())
    elif cmd == "choose_r2":
        choose_r2()
        print((chosen_path(block="r2_tune").parent / "tune_report.md").read_text())
    else:
        raise SystemExit("usage: python -m b2j.pilot choose | choose_ba | choose_r2")
