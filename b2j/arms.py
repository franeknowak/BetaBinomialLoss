"""Module a: the arm registry (plan §4.1) and the factory that turns an arm + training labels into head + loss.

Post-hoc arms (Global-S, S*(m), S*(m_B²), ensemble) are not trained and live in module d.
"""
from dataclasses import dataclass, replace

from b2j.heads import MultiHead
from b2j.losses import LossConfig, PanelLoss, class_weights, miccai_prior, moment_matched_prior

KL_LAMBDA = 0.05


@dataclass(frozen=True)
class Arm:
    loss: LossConfig
    head_type: str = "mlp"
    prior: str | None = None        # 'moment' (corrected KL) | 'miccai' (legacy B²-M)
    min_n: int = 1                  # DUP is only defined for n_train >= 3


ARMS = {
    "hard_bce":   Arm(LossConfig("hard_bce")),
    "hard_bce_w": Arm(LossConfig("hard_bce", weighted=True)),
    "soft_bce":   Arm(LossConfig("soft_bce")),
    "b2r":        Arm(LossConfig("b2")),
    "b2r_w":      Arm(LossConfig("b2", weighted=True)),
    "b2r_kl":     Arm(LossConfig("b2", kl_lambda=KL_LAMBDA), prior="moment"),
    "b2m":        Arm(LossConfig("b2", eps=1e-6, plus_one=True, weighted=True, kl_lambda=KL_LAMBDA), prior="miccai"),
    "b2r_linear": Arm(LossConfig("b2"), head_type="linear"),
    "dup":        Arm(LossConfig("dup"), min_n=3),
}


def build(arm_name: str, in_dim: int, k_train, n_train, eps: float | None = None, dropout: float = 0.1):
    """Head and loss for one arm. k_train, n_train: [N, K] training labels (weights and priors come from them).

    eps overrides the B²-r positivity constant (the R0 fallback); it never touches B²-M.
    """
    arm = ARMS[arm_name]
    cfg = arm.loss
    if eps is not None and cfg.kind == "b2" and not cfg.plus_one:
        cfg = replace(cfg, eps=eps)
    if int(n_train.min()) < arm.min_n:
        raise ValueError(f"arm {arm_name} needs n_train >= {arm.min_n}")
    weights = class_weights(k_train, n_train) if cfg.weighted else None
    prior = {None: None, "moment": moment_matched_prior, "miccai": miccai_prior}[arm.prior]
    prior = prior(k_train, n_train) if prior else None
    head = MultiHead(in_dim, k_train.shape[1], cfg.out_dim, arm.head_type, dropout=dropout)
    return head, PanelLoss(cfg, weights=weights, prior=prior)
