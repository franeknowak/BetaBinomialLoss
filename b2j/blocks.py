"""Module f: the run lists of each round block (plan §4.4). R1/R2 blocks use the head lr / wd per loss family
chosen by module g (results/journal/tune/chosen.json). No margins (05_plan amendment 2026-10-16 #1)."""
from b2j.runs import RunSpec
from b2j.variants import VARIANTS

DATASETS = ("endoscapes", "cifar10h", "mhist")
SMOKE_ENCODER_LR = 1e-5


def smoke():
    """R0 smoke test (plan §4.4 module f): every frozen variant on every dataset (1 seed), 1 fine-tuning epoch per
    loss (Endoscapes), one re-head pass (CIFAR-10H), and the subsampling / full-panel-validation code paths."""
    b = "smoke"
    specs = [RunSpec(ds, v, block=b) for ds in DATASETS for v in VARIANTS]
    specs += [RunSpec("endoscapes", v, regime="finetune", encoder_lr=SMOKE_ENCODER_LR, epochs=1, block=b)
              for v in ("hard_bce", "soft_bce", "b2r")]
    specs += [RunSpec("cifar10h", "b2r", regime="rehead", encoder_lr=SMOKE_ENCODER_LR, epochs=1, block=b),
              RunSpec("cifar10h", "b2r", part="B", block=b),
              RunSpec("cifar10h", "b2r", n_sub=3, block=b),
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


BLOCKS = {"smoke": smoke, "tune": tune, "pilot": pilot}
