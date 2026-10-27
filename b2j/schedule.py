"""Learning-rate schedule, evaluation points and weight-decay levels (PLAN §1.4, §1.7).

Cosine decay per optimiser step from the peak learning rate to 0, no warmup: lr_t = lr * 0.5 * (1 + cos(pi t / T)),
t = 0 .. T-1 (step t uses lr_t). AdamW (PyTorch) shrinks every weight by (1 - lr_t * wd) at each step, so a
weight-decay *level* L (fraction of a weight left after the run from decay alone) fixes wd through
prod_t (1 - lr_t * wd) = L; it is solved per dataset (T) and learning rate.
"""
import math

import numpy as np

EPOCHS, BATCH, EVALS_PER_EPOCH = 25, 64, 4
LRS = (5e-5, 1e-4, 5e-4, 1e-3)                    # agreed tuning grid
WD_LEVELS = (1.0, 0.75, 0.25)                     # agreed: 100% (wd 0), 75%, 25% of a weight left


def steps_per_epoch(train_size, batch=BATCH):
    return math.ceil(train_size / batch)              # last partial batch kept


def lr_factors(total_steps):
    """Cosine multiplier for every step t = 0 .. T-1."""
    t = np.arange(total_steps)
    return 0.5 * (1 + np.cos(np.pi * t / total_steps))


def eval_steps(train_size, epochs=EPOCHS, evals_per_epoch=EVALS_PER_EPOCH, batch=BATCH):
    """Number of optimiser steps completed at each evaluation (4 per epoch; the last one is the end of training)."""
    spe = steps_per_epoch(train_size, batch)
    steps = [round(e * spe / evals_per_epoch) for e in range(1, epochs * evals_per_epoch + 1)]
    if len(set(steps)) != len(steps) or steps[-1] != epochs * spe:
        raise ValueError(f"evaluation points not distinct for {train_size} training images")
    return steps


def wd_for_level(level, lr, train_size, epochs=EPOCHS, batch=BATCH):
    """AdamW weight decay leaving `level` of a weight after the run from decay alone (0 for level 1)."""
    if level >= 1:
        return 0.0
    y = lr * lr_factors(epochs * steps_per_epoch(train_size, batch))
    lo, hi = 0.0, 1.0 / y.max()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.exp(np.log1p(-y * mid).sum()) > level else (lo, mid)
    return 0.5 * (lo + hi)


def decay_left(wd, lr, train_size, epochs=EPOCHS, batch=BATCH):
    """Fraction of a weight left after the run from decay alone (the inverse check of wd_for_level)."""
    y = lr * lr_factors(epochs * steps_per_epoch(train_size, batch))
    return float(np.exp(np.log1p(-y * wd).sum()))
