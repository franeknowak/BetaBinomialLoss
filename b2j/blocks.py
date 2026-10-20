"""Module f: the run lists of each round block (plan §4.4). R1/R2 blocks use the head lr / wd per loss family
chosen by module g (results/journal/tune/chosen.json). No margins (05_plan amendment 2026-10-16 #1)."""
from b2j.runs import RunSpec
from b2j.variants import MAIN, S_SHRINK, VARIANTS

DATASETS = ("endoscapes", "cifar10h", "mhist")   # R0-R1 blocks, pinned
ROUND_D_DATASETS = ("amibr", "ferplus")        # added in round D (2026-10-19 / 2026-10-20)
SMOKE_ENCODER_LR = 1e-5


def smoke():
    """R0 smoke test (plan §4.4 module f): every frozen variant on every dataset (1 seed), 1 fine-tuning epoch per
    loss (Endoscapes), and the subsampling / full-panel-validation code paths. (Job 7519912 also ran a re-head pass
    and a part-B run; both designs were dropped on 2026-10-16.)"""
    b = "smoke"
    specs = [RunSpec(ds, v, block=b) for ds in DATASETS for v in MAIN]
    specs += [RunSpec("endoscapes", v, regime="finetune", encoder_lr=SMOKE_ENCODER_LR, epochs=1, block=b)
              for v in ("hard_bce", "soft_bce", "b2r")]
    specs += [RunSpec("cifar10h", "b2r", n_sub=3, block=b),
              RunSpec("cifar10h", "b2r", n_sub=3, val_full_panel=True, block=b),
              RunSpec("cifar10h", "soft_bce", n_sub=1, block=b),
              RunSpec("mhist", "dup", n_sub=3, block=b)]
    return specs


def tune():
    from b2j.pilot import tune_specs
    return tune_specs()


def pilot():
    from b2j.pilot import pilot_specs
    return pilot_specs()


SEEDS = range(5)
SUBSAMPLE_N = {"cifar10h": (1, 3, 5, 10), "mhist": (1, 3, 5)}     # plan §4.4 R1; the full panel is the main block
SWEEP = {"datasets": ("endoscapes", "cifar10h"), "fracs": (0.1, 0.3), "variants": ("soft_bce", "b2r")}


def r1(root=None):
    """R1, frozen encoder, 5 seeds (plan §4.4 R1; 05_plan amendment 2026-10-16 #6):
      main         every variant on every dataset (135);
      full panel   CIFAR-10H n = 3 validated on the full panel, soft BCE and B²-r (10; plan amendment 2026-10-14 #5);
      subsampling  soft BCE, B²-r at every n; DUP at n >= 3 (CIFAR-10H 55, MHIST 40);
      size sweep   soft BCE, B²-r on 10% and 30% of the training clusters, Endoscapes + CIFAR-10H (40; the 100%
                   point is the main block).
    Head lr / wd come from module g (tune/chosen.json), inherited by loss family."""
    from b2j.pilot import hyper
    kw = {} if root is None else {"root": root}
    spec = lambda ds, v, s, **x: RunSpec(ds, v, seed=s, block="r1", **hyper(ds, v, **kw), **x)
    out = [spec(ds, v, s) for ds in DATASETS for v in MAIN for s in SEEDS]
    out += [spec("cifar10h", v, s, n_sub=3, val_full_panel=True) for v in ("soft_bce", "b2r") for s in SEEDS]
    out += [spec(ds, v, s, n_sub=n) for ds, ns in SUBSAMPLE_N.items() for n in ns for v in ("soft_bce", "b2r", "dup")
            for s in SEEDS if n >= VARIANTS[v].min_n]
    out += [spec(ds, v, s, train_frac=f) for ds in SWEEP["datasets"] for f in SWEEP["fracs"] for v in SWEEP["variants"]
            for s in SEEDS]
    return out


def d_base(root=None):
    """Round D: the R1 main block (every variant in MAIN, frozen, 5 seeds) on the datasets added in round D (45 runs per
    dataset: AMi-Br, FER+). Needs their rows in tune/chosen.json (block `tune` then `pilot choose`)."""
    from b2j.pilot import hyper
    kw = {} if root is None else {"root": root}
    return [RunSpec(ds, v, seed=s, block="d_base", **hyper(ds, v, **kw)) for ds in ROUND_D_DATASETS for v in MAIN
            for s in SEEDS]


D12 = ("soft_bce", "b2r", "b2ms") + tuple(f"b2ms_var{lam:g}" for lam in S_SHRINK)   # D2b replaced the invalid
#   b2ms_shrink* runs (2026-10-20; moved to results/journal/d12_invalid_shrink/ on the cluster)


def d12(root=None):
    """Round D, D1 + D2b (journal_plan/iterations.md): the mean-spread head without and with spread shrinkage
    (towards the batch-mean log S) at every λ, plus soft BCE and B²-r as baselines in the same block; frozen, full panels, 5 seeds (150 runs)."""
    from b2j.pilot import hyper
    kw = {} if root is None else {"root": root}
    return [RunSpec(ds, v, seed=s, block="d12", **hyper(ds, v, **kw)) for ds in DATASETS + ROUND_D_DATASETS
            for v in D12 for s in SEEDS]


D5 = ("soft_bce", "soft_bce_n", "b2r")
D5_DATASETS = ("ferplus", "cifar10h")   # the datasets whose images have different numbers of votes


def d5(root=None):
    """Round D, D5 (journal_plan/iterations.md): soft BCE weighted by the number of votes (Binomial NLL) beside soft BCE
    and B²-r, on the datasets with a variable number of votes per image; frozen, full panels, 5 seeds (30 runs)."""
    from b2j.pilot import hyper
    kw = {} if root is None else {"root": root}
    return [RunSpec(ds, v, seed=s, block="d5", **hyper(ds, v, **kw)) for ds in D5_DATASETS for v in D5 for s in SEEDS]


BLOCKS = {"smoke": smoke, "tune": tune, "pilot": pilot, "r1": r1, "d_base": d_base, "d12": d12, "d5": d5}
