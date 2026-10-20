"""Module a: the registry of trained model variants (plan §4.1) and the factory that turns a variant + training labels into head + loss.

Post-hoc comparisons (Binomial plug-in, Global-S, soft-BCE ensemble) are not trained and live in module d.
"""
from dataclasses import dataclass, replace

from b2j.heads import MultiHead
from b2j.losses import LossConfig, PanelLoss, class_weights, miccai_prior, moment_matched_prior

KL_LAMBDA = 0.05
S_SHRINK = (0.03, 0.3, 3.0)         # round D, D2: every value is reported (dose-response), none is selected


@dataclass(frozen=True)
class Variant:
    loss: LossConfig
    head_type: str = "mlp"
    prior: str | None = None        # 'moment' (corrected KL) | 'miccai' (legacy B²-M)
    min_n: int = 1                  # DUP is only defined for n_train >= 3


VARIANTS = {
    "hard_bce":   Variant(LossConfig("hard_bce")),
    "hard_bce_w": Variant(LossConfig("hard_bce", weighted=True)),
    "soft_bce":   Variant(LossConfig("soft_bce")),
    "b2r":        Variant(LossConfig("b2")),
    "b2r_w":      Variant(LossConfig("b2", weighted=True)),
    "b2r_kl":     Variant(LossConfig("b2", kl_lambda=KL_LAMBDA), prior="moment"),
    "b2m":        Variant(LossConfig("b2", eps=1e-6, plus_one=True, weighted=True, kl_lambda=KL_LAMBDA), prior="miccai"),
    "b2r_linear": Variant(LossConfig("b2"), head_type="linear"),
    "dup":        Variant(LossConfig("dup"), min_n=3),
    # round D candidates (journal_plan/iterations.md); not in the R1/R2 blocks
    "b2ms":       Variant(LossConfig("b2", param="ms")),
}
# D5: soft BCE weighted by the image's number of votes (= Binomial NLL): isolates B²'s weighting of images by n
VARIANTS["soft_bce_n"] = Variant(LossConfig("soft_bce", n_weight=True))
# D2 (2026-10-20, INVALID design: shrinks towards the across-image prior spread; kept to reproduce the record)
VARIANTS.update({f"b2ms_shrink{lam:g}": Variant(LossConfig("b2", param="ms", s_shrink=lam), prior="moment")
                 for lam in S_SHRINK})
# D2b: shrink log S towards the batch mean (λ -> inf = B² with one learned spread per head)
VARIANTS.update({f"b2ms_var{lam:g}": Variant(LossConfig("b2", param="ms", s_shrink=lam, s_shrink_target="batch"))
                 for lam in S_SHRINK})

MAIN = ("hard_bce", "hard_bce_w", "soft_bce", "b2r", "b2r_w", "b2r_kl", "b2m", "b2r_linear", "dup")   # R0-R2 set


def build(name: str, in_dim: int, k_train, n_train, eps: float | None = None, dropout: float = 0.1):
    """Head and loss for one variant. k_train, n_train: [N, K] training labels (weights and priors come from them).

    eps overrides the B²-r positivity constant (the R0 fallback); it never touches B²-M.
    """
    variant = VARIANTS[name]
    cfg = variant.loss
    if eps is not None and cfg.kind == "b2" and not cfg.plus_one:
        cfg = replace(cfg, eps=eps)
    if int(n_train.min()) < variant.min_n:
        raise ValueError(f"variant {name} needs n_train >= {variant.min_n}")
    weights = class_weights(k_train, n_train) if cfg.weighted else None
    prior = {None: None, "moment": moment_matched_prior, "miccai": miccai_prior}[variant.prior]
    prior = prior(k_train, n_train) if prior else None
    head = MultiHead(in_dim, k_train.shape[1], cfg.out_dim, variant.head_type, dropout=dropout)
    return head, PanelLoss(cfg, weights=weights, prior=prior)
