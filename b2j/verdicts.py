"""Module f: the pre-registered §5.1 readings (05_plan §5, tag prereg-b2r-2026-10-16), one row per endpoint.

  python -m b2j.verdicts <block> [--split val] [--fine-tune-block r2] [--B 2000]

Every endpoint is a difference with its two-level bootstrap 95% CI (stats.two_level_bootstrap) read by
stats.verdict: 'superiority' (predicted direction) or 'difference' (no margins). On validation the output is a
preview only (thresholds and spreads are fitted on the same rows); the verdicts are the test readings of R3,
which need --open-test. Output: <block>/verdicts_<split>.md and .json.
See journal_plan/specs/f_launcher.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from b2j import metrics as M
from b2j import posthoc as P
from b2j import stats as St
from b2j.analysis import boot_strata, guard, nll_rows, run_rows
from b2j.data import ambiguous_mask, load_table
from b2j.runs import RESULTS, RunSpec, tables as run_tables

CORE = ("endoscapes", "cifar10h")


class Block:
    """Predictions of one block on one dataset and split, loaded once per (variant, n_sub, regime)."""

    def __init__(self, block, dataset, split, root=RESULTS):
        self.block, self.ds, self.split, self.root = block, dataset, split, Path(root)
        self.full = load_table(dataset)
        self.ev = self.full.take(split)
        self.y = M.majority(self.ev.k, self.ev.n)
        self.per_vote = dataset == "cifar10h"
        self.strata = boot_strata(dataset, self.ev)
        self._cache = {}

    def runs(self, variant, n_sub=None, regime="frozen"):
        out = []
        for f in sorted((self.root / self.block / self.ds).glob("*/spec.json")):
            s = RunSpec.from_json(json.loads(f.read_text()))
            if (s is not None and s.variant == variant and s.regime == regime and s.n_sub == n_sub
                    and not s.val_full_panel and s.train_frac == 1.0 and (f.parent / "summary.json").exists()):
                out.append((s, f.parent))
        return out

    def preds(self, variant, n_sub=None, regime="frozen"):
        """Per seed: the variant's row of analysis.run_rows (validation-fitted threshold and spread), plus 'noc':
        the B² vote pmf without the scale c."""
        key = (variant, n_sub, regime)
        if key not in self._cache:
            out = []
            for spec, d in self.runs(variant, n_sub, regime):
                p = dict(run_rows(variant, d, self.split, self.ev, run_tables(spec, self.full)[1])[variant])
                if P.spread_rule(variant) == "c":
                    m, S0 = P.mean_spread(variant, np.load(d / f"{self.split}_B.npz")["out"])
                    p["pmf_noc"] = P.vote_pmf(m, S0, self.ev.n)
                out.append(p)
            self._cache[key] = out
        return self._cache[key]

    def nll(self, variant, n_sub=None, regime="frozen", noc=False):
        """Per seed [N, K] vote NLL (per vote on CIFAR-10H)."""
        ps = self.preds(variant, n_sub, regime)
        return [nll_rows({"pmf": p["pmf_noc" if noc else "pmf"]}, self.ev, self.per_vote) for p in ps]

    def boot(self, metric, models, combine, paired=(), B=St.B_DEFAULT):
        if any(len(v) == 0 for v in models.values()):
            raise LookupError("missing runs: " + ", ".join(k for k, v in models.items() if not v))
        return St.two_level_bootstrap(metric, models, combine, self.ev.cluster, self.strata, paired=paired, B=B)


def _mean(a, r):
    return a[r].mean()


def _disc(b, key):
    return lambda p, r: M.discrimination(p["disc"][r], b.y[r], p["thr"])[key]


def _advantage(b, n_sub=None, regime="frozen", mask=None):
    """Models and combine for advantage = NLL(soft BCE + Global-S) − NLL(B²-r) (positive = B²-r better)."""
    models = {f"gs{n_sub}{regime}": b.nll("soft_bce", n_sub, regime), f"b2r{n_sub}{regime}": b.nll("b2r", n_sub, regime)}
    if mask is not None:
        models = {k: [np.where(mask[:, None], a, np.nan) for a in v] for k, v in models.items()}
    return models, (lambda v: v[f"gs{n_sub}{regime}"] - v[f"b2r{n_sub}{regime}"])


def _nanmean(a, r):
    return float(np.nanmean(a[r]))


def endpoints(b: Block, ft: Block | None = None, B=St.B_DEFAULT):
    """Yields (id, contrast, kind, direction, Result | error string) for one dataset (plan §5.1 table)."""
    ds = b.ds

    def run(eid, contrast, kind, direction, fn):
        try:
            return eid, contrast, kind, direction, fn()
        except LookupError as e:
            return eid, contrast, kind, direction, str(e)

    diff = lambda a, c: (lambda v: v[a] - v[c])
    # S0: discrimination on m at checkpoint A, B²-r − soft BCE (open, two-sided)
    for key in ("bacc", "map"):
        yield run("S0", f"Δ {key.upper()}, B²-r − soft BCE", "difference", +1,
                  lambda key=key: b.boot(_disc(b, key), {"b2r": b.preds("b2r"), "soft": b.preds("soft_bce")},
                                         diff("b2r", "soft"), B=B))
    # ST-w: weighted − unweighted (both with the validation BACC threshold), per loss
    for w, u, name in (("b2r_w", "b2r", "B²-r"), ("hard_bce_w", "hard_bce", "hard BCE")):
        yield run("ST-w", f"Δ BACC, {name}: weighted − unweighted", "difference", +1,
                  lambda w=w, u=u: b.boot(_disc(b, "bacc"), {"w": b.preds(w), "u": b.preds(u)}, diff("w", "u"), B=B))
    # ST-KL: Δ vote NLL B²-r − (B²-r + KL), negative = no-KL better; with c (primary) and without c
    for noc, tag in ((False, "with c"), (True, "without c (secondary)")):
        yield run("ST-KL", f"Δ vote NLL, B²-r − B²-r + KL, {tag}", "difference", -1,
                  lambda noc=noc: b.boot(_mean, {"b2r": b.nll("b2r", noc=noc), "kl": b.nll("b2r_kl", noc=noc)},
                                         diff("b2r", "kl"), B=B))
    if ds in ("cifar10h", "mhist"):
        # S4a: advantage at the full panel; all images and the ambiguous subset
        yield run("S4a", "advantage, n = all", "superiority", +1,
                  lambda: b.boot(_mean, *_advantage(b), B=B))
        amb = ambiguous_mask(b.ev)
        if amb.ndim > 1:
            amb = amb.any(1)
        yield run("S4a", f"advantage, n = all, ambiguous subset ({int(amb.sum())} images)", "superiority", +1,
                  lambda: b.boot(_nanmean, *_advantage(b, mask=amb), B=B))
        # S4b: advantage(all) − advantage(3)
        def s4b():
            m_all, c_all = _advantage(b)
            m3, c3 = _advantage(b, 3)
            return b.boot(_mean, {**m_all, **m3}, lambda v: c_all(v) - c3(v), B=B)
        yield run("S4b", "advantage(all) − advantage(n = 3)", "superiority", +1, s4b)
    # S6a (Endoscapes) / S6b (others): Δ FCR at test recall 0.90, P_maj − m within B²-r, same runs (paired seeds)
    if ds == "endoscapes":
        yield run("S4c", "advantage, n = 3", "difference", +1, lambda: b.boot(_mean, *_advantage(b), B=B))
    fcr = lambda p, r: M.fcr_at_recall(p["score"][r], b.y[r])

    def s6():
        ps = b.preds("b2r")
        return b.boot(fcr, {"pmaj": [dict(score=P.p_majority(p["pmf"], b.ev.n)) for p in ps],
                            "m": [dict(score=p["m"]) for p in ps]}, diff("pmaj", "m"), paired=[("pmaj", "m")], B=B)
    if ds == "endoscapes":
        yield run("S6a", "Δ FCR at recall 0.90, P_maj − m (B²-r)", "difference", -1, s6)
    else:
        yield run("S6b", "Δ FCR at recall 0.90, P_maj − m (B²-r), full-panel majority", "superiority", -1, s6)
    # S5c (CIFAR-10H): frozen advantage − fine-tuned advantage, Global-S from the soft-BCE run of the same regime
    if ds == "cifar10h" and ft is not None:
        def s5c():
            mf, cf = _advantage(b)
            mt, ct = _advantage(ft, regime="finetune")
            return b.boot(_mean, {**mf, **mt}, lambda v: cf(v) - ct(v), B=B)
        yield run("S5c", "frozen advantage − fine-tuned advantage", "superiority", +1, s5c)


def read(block, split="val", root=RESULTS, open_test=False, ft_block=None, B=St.B_DEFAULT,
         datasets=("endoscapes", "cifar10h", "mhist")):
    guard(split, open_test)
    rows = []
    for ds in datasets:
        if not (Path(root) / block / ds).exists():
            continue
        b = Block(block, ds, split, root)
        ft = Block(ft_block, ds, split, root) if ft_block and (Path(root) / ft_block / ds).exists() else None
        for eid, contrast, kind, direction, r in endpoints(b, ft, B):
            row = {"id": eid, "dataset": ds, "role": "core" if ds in CORE else "confirmatory", "contrast": contrast,
                   "kind": kind, "direction": direction}
            if isinstance(r, str):
                row.update(reading="not available", note=r)
            else:
                row.update(estimate=r.estimate, lo=r.lo, hi=r.hi, reading=St.verdict(r, kind, direction))
            rows.append(row)
    out = Path(root) / block
    out.mkdir(parents=True, exist_ok=True)
    (out / f"verdicts_{split}.json").write_text(json.dumps(rows, indent=1))
    pred = {(+1, "superiority"): "> 0", (-1, "superiority"): "< 0", (+1, "difference"): "+ = better",
            (-1, "difference"): "− = better"}
    md = [f"# §5.1 readings ({block}, {split})", "",
          "Validation preview only: thresholds and spreads are fitted on these rows; the verdicts are read on test in R3."
          if split == "val" else "Test readings (R3).", "",
          "| ID | dataset | contrast | reading | prediction / sign | estimate | 95% CI |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        est = f"{r['estimate']:+.4f} | [{r['lo']:+.4f}, {r['hi']:+.4f}]" if "estimate" in r else f"– | {r['note']}"
        md.append(f"| {r['id']} | {r['dataset']}{'' if r['role'] == 'core' else ' (conf.)'} | {r['contrast']} | "
                  f"**{r['reading']}** | {pred[(r['direction'], r['kind'])]} | {est} |")
    (out / f"verdicts_{split}.md").write_text("\n".join(md) + "\n")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("block")
    ap.add_argument("--split", default="val")
    ap.add_argument("--open-test", action="store_true", help="R3 only")
    ap.add_argument("--fine-tune-block", default=None, help="block with the R2 fine-tuned runs (S5c)")
    ap.add_argument("--B", type=int, default=St.B_DEFAULT)
    a = ap.parse_args()
    read(a.block, a.split, open_test=a.open_test, ft_block=a.fine_tune_block, B=a.B)
    print((RESULTS / a.block / f"verdicts_{a.split}.md").read_text())
