"""Module f: smoke-test report (validation only). python -m b2j.report_smoke [block]

Reads spec.json / summary.json / log.jsonl and val_*.npz of every run in results/journal/<block>; never opens a
test file. Writes <block>/report.md with:
  1. one row per run: chosen evaluations, validation scores, non-finite steps, stuck share, S ratio, time;
  2. the ε fallback rule (plan §2): B²-r family, any non-finite loss or > 5% of k >= 1 training samples stuck;
  3. module e on real validation predictions (Endoscapes, B²-r vs soft BCE): metrics + a video-cluster bootstrap.
"""
import json
import sys
from pathlib import Path

import numpy as np

from b2j import metrics as M
from b2j import posthoc as P
from b2j import stats as St
from b2j.data import load_table
from b2j.runs import RESULTS
from b2j.variants import VARIANTS

STUCK_LIMIT = 0.05


def load_runs(block_dir: Path):
    runs = []
    for spec_file in sorted(block_dir.rglob("spec.json")):
        d = spec_file.parent
        if not (d / "summary.json").exists():
            runs.append({"dir": d, "spec": json.loads(spec_file.read_text()), "summary": None})
            continue
        runs.append({"dir": d, "spec": json.loads(spec_file.read_text()),
                     "summary": json.loads((d / "summary.json").read_text())})
    return runs


def _log(d: Path):
    return [json.loads(l) for l in (d / "log.jsonl").read_text().splitlines()]


def _row(name, d, s):
    log = _log(d)
    stuck = log[-1].get("train_stuck_share")
    return {"run": name, "A_epoch": s["A"]["epoch"], "B_epoch": s["B"]["epoch"],
            "val_map_A": s["A"].get("val_map"), "val_nll_B": s["B"].get("val_nll_per_vote"),
            "val_dup_bce": s["B"].get("val_dup_bce"), "nonfinite": s["nonfinite_steps"],
            "stuck_max": None if stuck is None else max(stuck),
            "S_ratio": s["B"].get("train_val_S_ratio"), "sec": s["seconds"], "n_evals": len(log)}


def rows(runs):
    out = []
    for r in runs:
        sp, s, d = r["spec"], r["summary"], r["dir"]
        name = f'{sp["dataset"]}/{d.name}'
        if s is None:
            out.append({"run": name, "status": "NOT FINISHED"})
        elif "rehead" in s:
            out.append(_row(name + " [fine-tune part A]", d / "finetune", s["finetune"]))
            out.append(_row(name + " [re-head part B]", d / "rehead", s["rehead"]))
        else:
            out.append(_row(name, d, s))
    return out


def eps_rule(table):
    """Plan §2 fallback: B²-r family (not B²-M) with any non-finite loss or a stuck share > 5%."""
    hits = []
    for r in table:
        v = r["run"].split("/")[1].split("__")[0]
        if v in VARIANTS and VARIANTS[v].loss.kind == "b2" and not VARIANTS[v].loss.plus_one and "status" not in r:
            if r["nonfinite"] > 0 or (r["stuck_max"] or 0) > STUCK_LIMIT:
                hits.append(r["run"])
    return hits


def module_e_check(block_dir: Path, B=500):
    """Module e on real validation predictions: Endoscapes frozen B²-r vs soft BCE (seed 0)."""
    val = load_table("endoscapes").take("val")
    y = M.majority(val.k, val.n)
    split_t = M.split_target(val.k, val.n)
    res, preds = {}, {}
    for v in ("b2r", "soft_bce"):
        d = block_dir / "endoscapes" / f"{v}__frozen__lr0.0003_wd0.01__seed0"
        a, b = np.load(d / "val_A.npz"), np.load(d / "val_B.npz")
        assert (a["ids"] == val.ids).all()
        mA = P.mean_concentration(v, a["out"])[0]
        mB, SB = P.panel(v, b["out"], b["scale"])
        thr = P.threshold_max_bacc(mA, y)
        pmaj = P.p_majority(P.vote_pmf(mB, SB, val.n), val.n)
        ps = P.p_split3(mB, SB)
        res[v] = {**M.discrimination(mA, y, thr), **M.vote_nll(mB, SB, val.k, val.n), "ece_pmaj": M.ece(pmaj, y),
                  "fcr90_m": M.fcr_at_recall(mB, y), "fcr90_pmaj": M.fcr_at_recall(pmaj, y), "aurc": M.aurc(pmaj, y),
                  **{f"triage_{k}": x for k, x in M.triage(ps, split_t).items()},
                  "split_rate_cov80": M.split_rate_at_coverage(ps, split_t)}
        preds[v] = (mA, y)
    ap_metric = lambda p, rows: M.discrimination(p[0][rows], p[1][rows], np.full(p[0].shape[1], 0.5))["map"]
    boot = St.two_level_bootstrap(ap_metric, {v: [preds[v]] for v in preds}, lambda x: x["b2r"] - x["soft_bce"],
                                  val.cluster, B=B)
    return res, boot


def fmt(x):
    if x is None:
        return "–"
    if isinstance(x, list):
        return "/".join(f"{v:.2f}" for v in x)
    return f"{x:.4f}" if isinstance(x, float) else str(x)


if __name__ == "__main__":
    block = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    bd = RESULTS / block
    runs = load_runs(bd)
    table = rows(runs)
    cols = ["run", "A_epoch", "B_epoch", "val_map_A", "val_nll_B", "val_dup_bce", "nonfinite", "stuck_max", "S_ratio", "sec", "n_evals"]
    lines = [f"# Smoke report: {block} (validation only)", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in table:
        lines.append("| " + " | ".join(fmt(r.get(c, r.get("status") if c == "A_epoch" else None)) for c in cols) + " |")
    hits = eps_rule(table)
    lines += ["", "## ε fallback rule", "TRIGGERED by: " + ", ".join(hits) if hits else "Not triggered (no non-finite loss; stuck share ≤ 5% in every B²-r-family run)."]
    try:
        res, boot = module_e_check(bd)
        keys = list(res["b2r"])
        lines += ["", "## Module e on real validation predictions (Endoscapes, seed 0)", "",
                  "| metric | B²-r | soft BCE |", "|---|---|---|"]
        lines += [f"| {k} | {res['b2r'][k]:.4f} | {res['soft_bce'][k]:.4f} |" for k in keys]
        lines += ["", f"Bootstrap over validation videos (B = 500): Δ mAP (B²-r − soft BCE) = {boot.estimate:+.4f}, "
                      f"95% CI [{boot.lo:+.4f}, {boot.hi:+.4f}] — a pipeline check only, not a result."]
    except FileNotFoundError as e:
        lines += ["", f"Module e check skipped: {e}"]
    text = "\n".join(lines) + "\n"
    (bd / "report.md").write_text(text)
    print(text)
