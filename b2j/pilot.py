"""Module g: tuning grid, choice of head lr / wd per loss family, pilot seeds and equivalence margins (plan §4.4 g,
§4.3 "Margins"). Validation only.

  python -m b2j.pilot choose    # after block `tune`  -> results/journal/tune/chosen.json + tune_report.md
  python -m b2j.pilot margins   # after block `pilot` -> results/journal/pilot/margins.json + margins.md
See journal_plan/specs/g_pilot.md.
"""
import json
import sys
from pathlib import Path

import numpy as np

from b2j import metrics as M
from b2j import posthoc as P
from b2j import stats as St
from b2j.data import DATA_ROOT, load_table
from b2j.runs import RESULTS, RunSpec

DATASETS = ("endoscapes", "cifar10h", "mhist")
PILOT_DATASETS = ("endoscapes", "cifar10h")          # the margins are needed where claims are decided
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


# ---------------------------------------------------------------- margins
def _seed_preds(ds, variant, val, root):
    out = []
    for s in SEEDS:
        d = RunSpec(ds, variant, seed=s, block="pilot", **hyper(ds, variant, root)).out_dir(root)
        a, b = np.load(d / "val_A.npz"), np.load(d / "val_B.npz")
        assert (a["ids"] == val.ids).all() and (b["ids"] == val.ids).all()
        mA = P.mean_spread(variant, a["out"])[0]
        mB, SB = P.panel(variant, b["out"], b["scale"])
        S_plug = np.full_like(mB, np.inf)
        y = M.majority(val.k, val.n)
        rec = {"mA": mA, "thr": P.threshold_max_bacc(mA, y), "mB": mB, "SB": SB, "S_plug": S_plug,
               "pmaj": P.p_majority(P.vote_pmf(mB, SB, val.n), val.n), "psplit": P.p_split3(mB, SB)}
        if variant == "b2r":    # S*(m_B²): the per-sample comparator, fitted on validation (as in R1)
            rec["S_sofm"] = P.s_of_m(P.fit_s_of_m(mB, val.k, val.n), mB)
        out.append(rec)
    return out


def margins(root=RESULTS, B=St.B_DEFAULT):
    res, lines = {}, ["# Pilot margins (validation only; δ = 2h, h = bootstrap 95% CI half-width)", ""]
    for ds in PILOT_DATASETS:
        val = load_table(ds).take("val")
        y = M.majority(val.k, val.n)
        split_t = M.split_target(val.k, val.n)
        strata = None
        if ds == "cifar10h":
            strata = np.load(DATA_ROOT / "cifar10h" / "test_labels.npy")[val.cluster]
        per_vote = ds == "cifar10h"
        nll_key = "nll_per_vote" if per_vote else "nll"
        soft, b2 = _seed_preds(ds, "soft_bce", val, root), _seed_preds(ds, "b2r", val, root)
        diff = lambda a, b: (lambda v: v[a] - v[b])
        disc = lambda key: (lambda p, r: M.discrimination(p["mA"][r], y[r], p["thr"])[key])
        tests = {   # name: (metric(pred, rows), models, combine, paired groups)
            # S0 (and ST-w): Δ BACC, Δ mAP on m at checkpoint A, B²-r − soft BCE (independent seeds)
            "bacc": (disc("bacc"), {"b2r": b2, "soft": soft}, diff("b2r", "soft"), ()),
            "map": (disc("map"), {"b2r": b2, "soft": soft}, diff("b2r", "soft"), ()),
            # S4c (and ST-KL): Δ vote NLL, B²-r − S*(m_B²), the same runs scored two ways
            "nll": (lambda p, r: M.vote_nll(p["mB"][r], p["S"][r], val.k[r], val.n[r])[nll_key],
                    {"b2r": [dict(q, S=q["SB"]) for q in b2], "sofm": [dict(q, S=q["S_sofm"]) for q in b2]},
                    diff("b2r", "sofm"), [("b2r", "sofm")]),
            # S6a: Δ FCR at recall 0.90, P_maj − m within B²-r
            "fcr": (lambda p, r: M.fcr_at_recall(p["score"][r], y[r]),
                    {"pmaj": [dict(q, score=q["pmaj"]) for q in b2], "m": [dict(q, score=q["mB"]) for q in b2]},
                    diff("pmaj", "m"), [("pmaj", "m")]),
            # H6a / U2: Δ AP of P(split), B²-r − the m-only score 3m(1 − m) of soft BCE
            "ap": (lambda p, r: M.triage(p["psplit"][r], split_t[r])["ap"],
                   {"b2r": b2, "soft": [dict(q, psplit=3 * q["mB"] * (1 - q["mB"])) for q in soft]},
                   diff("b2r", "soft"), ()),
        }
        # NLL cap: 1% of the soft-BCE plug-in validation NLL (seed mean)
        plug = np.mean([M.vote_nll(q["mB"], q["S_plug"], val.k, val.n)[nll_key] for q in soft])
        caps = {**St.CAPS, "nll": 0.01 * plug}
        res[ds] = {}
        lines += [f"## {ds}", "", "| metric | Δ estimate | 95% CI | h | δ = 2h | cap | margin |", "|---|---|---|---|---|---|---|"]
        for key, (fn, models, comb, paired) in tests.items():
            r = St.two_level_bootstrap(fn, models, comb, val.cluster, strata=strata, paired=paired, B=B)
            d = St.margin(r, caps[key])
            res[ds][key] = {"estimate": r.estimate, "lo": r.lo, "hi": r.hi, "h": r.half_width,
                            "delta": d, "cap": caps[key]}
            lines.append(f"| {key} | {r.estimate:+.4f} | [{r.lo:+.4f}, {r.hi:+.4f}] | {r.half_width:.4f} | "
                         f"{2 * r.half_width:.4f} | {caps[key]:.4f} | {'%.4f' % d if d is not None else 'above cap → no claim'} |")
        lines.append("")
    out = Path(root) / "pilot"
    (out / "margins.json").write_text(json.dumps(res, indent=1))
    (out / "margins.md").write_text("\n".join(lines))
    return res


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "choose":
        print(choose()[1])
        print((chosen_path().parent / "tune_report.md").read_text())
    elif cmd == "margins":
        margins()
        print((RESULTS / "pilot" / "margins.md").read_text())
    else:
        raise SystemExit("usage: python -m b2j.pilot choose|margins")
