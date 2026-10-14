"""Module f: one training run = one RunSpec. Builds the splits, trains, writes to results/journal/<block>/<dataset>/<run>.

Regimes:
  frozen   - head on the cached public DINOv3 features (cache/<dataset>_frozen.npz);
  finetune - encoder + head on images;
  rehead   - H5 held-out re-head (plan §3 H5, R2): fine-tune B²-r on part A (80% of training clusters),
             embed every image with the checkpoint-B encoder (kept in memory, never saved), then train a fresh
             frozen head on part B (the other 20%). Its comparator is a `frozen` run with part="B".
Labels: training and validation votes are subsampled to n_sub raters when asked (validation keeps the full
panel if val_full_panel); test is always the full panel and reaches the trainer without labels.
See journal_plan/specs/f_launcher.md.
"""
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from b2j import embed as E
from b2j import train as T
from b2j.data import ImageSet, LabelTable, load_table, subsample

RESULTS = Path(__file__).resolve().parents[1] / "results" / "journal"
PART_A_FRAC = 0.8


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    variant: str
    seed: int = 0
    regime: str = "frozen"            # frozen | finetune | rehead
    n_sub: int | None = None          # raters per image for train (and val); None = full panel
    val_full_panel: bool = False      # sensitivity check: validate on the full panel
    part: str = "all"                 # training rows: all | A | B (H5 re-head split)
    lr: float = 3e-4
    wd: float = 1e-2
    encoder_lr: float | None = None
    epochs: int | None = None         # None = trainer default for the regime
    eps: float | None = None
    block: str = "dev"

    @property
    def name(self) -> str:
        bits = [self.variant, self.regime]
        if self.n_sub is not None:
            bits.append(f"n{self.n_sub}" + ("vfull" if self.val_full_panel else ""))
        if self.part != "all":
            bits.append(f"part{self.part}")
        if self.encoder_lr is not None:
            bits.append(f"elr{self.encoder_lr:g}")
        bits.append(f"lr{self.lr:g}_wd{self.wd:g}")
        if self.epochs is not None:
            bits.append(f"ep{self.epochs}")
        if self.eps is not None:
            bits.append(f"eps{self.eps:g}")
        return "__".join(bits) + f"__seed{self.seed}"

    def out_dir(self, root=RESULTS) -> Path:
        return Path(root) / self.block / self.dataset / self.name


def part_mask(train: LabelTable, part: str, seed: int) -> np.ndarray:
    """Rows of the training table in part A (80% of clusters) or B (the rest), drawn per seed."""
    if part == "all":
        return np.ones(len(train.ids), bool)
    cl = np.unique(train.cluster)
    a = np.random.default_rng(1000 + seed).permutation(cl)[: int(round(PART_A_FRAC * len(cl)))]
    in_a = np.isin(train.cluster, a)
    return in_a if part == "A" else ~in_a


def tables(spec: RunSpec, table: LabelTable | None = None):
    """(train, val, test) label tables for a run: subsampling and part split applied; test full panel."""
    full = table if table is not None else load_table(spec.dataset)
    sub = subsample(full, spec.n_sub, spec.seed) if spec.n_sub is not None else full
    train = sub.take("train")
    train = train.subset(part_mask(train, spec.part, spec.seed))
    val = (full if spec.val_full_panel else sub).take("val")
    return train, val, full.take("test")


def _config(spec: RunSpec, finetune: bool, num_workers: int = 8) -> T.TrainConfig:
    kw = {"lr": spec.lr, "wd": spec.wd, "seed": spec.seed, "eps": spec.eps, "num_workers": num_workers}
    if spec.epochs is not None:
        kw["epochs"] = spec.epochs
    if finetune:
        return T.TrainConfig.finetune(spec.variant, spec.encoder_lr, **kw)
    return T.TrainConfig(spec.variant, **kw)


def _frozen_splits(tabs, features: np.ndarray, all_ids):
    pos = {i: r for r, i in enumerate(all_ids)}
    return [T.split_from_table(t, features[[pos[i] for i in t.ids]], labelled=name != "test")
            for name, t in zip(("train", "val", "test"), tabs)]


def execute(spec: RunSpec, root=RESULTS, device=None, verbose=True, num_workers=8) -> dict:
    """Run one spec end to end; skips (returns the saved summary) if it already finished."""
    out = spec.out_dir(root)
    if (out / "summary.json").exists():
        return json.loads((out / "summary.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(asdict(spec), indent=1))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    full = load_table(spec.dataset)
    tabs = tables(spec, full)

    if spec.regime == "frozen":
        X = E.load_cache(full, E.cache_path(spec.dataset))
        splits = _frozen_splits(tabs, X, full.ids)
        return T.run(_config(spec, False, num_workers), *splits, out, device=device, verbose=verbose)

    if spec.regime not in ("finetune", "rehead"):
        raise ValueError(f"unknown regime {spec.regime}")
    if spec.encoder_lr is None:
        raise ValueError("fine-tuning needs encoder_lr")
    ft_spec = replace(spec, part="A") if spec.regime == "rehead" else spec
    ft_tabs = tables(ft_spec, full)
    splits = [T.split_from_table(t, ImageSet(t, spec.dataset), labelled=name != "test")
              for name, t in zip(("train", "val", "test"), ft_tabs)]
    ft_out = out / "finetune" if spec.regime == "rehead" else out
    summary, model = T.run(_config(spec, True, num_workers), *splits, ft_out, encoder=E.load_encoder(spec.dataset),
                           device=device, return_model_B=True, verbose=verbose)
    if spec.regime == "finetune":
        return summary

    # re-head: features of every image from the checkpoint-B encoder; the encoder is then dropped
    X = E.embed(model.encoder, ImageSet(full, spec.dataset), device, num_workers=num_workers)
    del model
    np.savez(out / "features_ftB.npz", ids=full.ids, X=X)
    part_b = tables(replace(spec, part="B"), full)
    head = T.run(_config(replace(spec, part="B", epochs=None), False, num_workers), *_frozen_splits(part_b, X, full.ids),
                 out / "rehead", device=device, verbose=verbose)
    both = {"finetune": summary, "rehead": head}
    (out / "summary.json").write_text(json.dumps(both, indent=1))
    return both
