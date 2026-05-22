import torch


def beta_binomial_nll_per_sample(alpha: torch.Tensor,
                                 y_soft: torch.Tensor,
                                 n_annotators: int = 3) -> torch.Tensor:
    device = alpha.device
    alpha = alpha.cpu()
    y_soft = y_soft.cpu()

    a0 = alpha[:, 0]
    a1 = alpha[:, 1]

    y_soft = y_soft.view(-1)
    k = torch.round(y_soft * n_annotators).to(dtype=torch.long)

    kf = k.to(dtype=a0.dtype)
    nf = torch.tensor(float(n_annotators), dtype=a0.dtype)

    log_comb = torch.lgamma(nf + 1.0) - torch.lgamma(kf + 1.0) - torch.lgamma((nf - kf) + 1.0)

    log_B_num = (
        torch.lgamma(kf + a1)
        + torch.lgamma((nf - kf) + a0)
        - torch.lgamma(nf + a0 + a1)
    )
    log_B_den = (
        torch.lgamma(a1)
        + torch.lgamma(a0)
        - torch.lgamma(a0 + a1)
    )

    log_p = log_comb + log_B_num - log_B_den
    return (-log_p).to(device)


def dirichlet_kl_per_sample(alpha: torch.Tensor,
                            prior_alpha: torch.Tensor | None = None) -> torch.Tensor:
    from torch.distributions import Dirichlet  # lazy import — must not be at module level

    if prior_alpha is None:
        prior_alpha = torch.ones_like(alpha)
    else:
        prior_alpha = torch.as_tensor(prior_alpha, device=alpha.device, dtype=alpha.dtype)
        if prior_alpha.ndim == 1:
            prior_alpha = prior_alpha.view(1, 2).expand_as(alpha)

    device = alpha.device
    posterior = Dirichlet(alpha.cpu())
    prior = Dirichlet(prior_alpha.cpu())
    return torch.distributions.kl.kl_divergence(posterior, prior).to(device)


def evidential_bb_loss(alpha: torch.Tensor,
                       y_soft: torch.Tensor,
                       w_pos: float,
                       w_neg: float,
                       lambda_reg: float = 0.05,
                       use_kl: bool = True,
                       prior_alpha: torch.Tensor | None = None,
                       n_annotators: int = 3) -> torch.Tensor:
    """
    Final composed loss for one head:
        per-sample: w(y_hard) * L_BB + lambda_reg * KL   (KL optional)
    where y_hard is majority vote derived from k.

    w_pos, w_neg are precomputed scalars (e.g., w+ = N/(2N+), w- = N/(2N-)).
    """
    bb = beta_binomial_nll_per_sample(alpha, y_soft, n_annotators=n_annotators)  # (B,)

    k = torch.round(y_soft.view(-1) * n_annotators).to(dtype=torch.long)
    y_hard = (k >= (n_annotators // 2 + 1))  # bool (B,)

    w_pos_t = torch.tensor(float(w_pos), device=alpha.device, dtype=bb.dtype)
    w_neg_t = torch.tensor(float(w_neg), device=alpha.device, dtype=bb.dtype)
    w = torch.where(y_hard, w_pos_t, w_neg_t)  # (B,)

    loss = w * bb  # (B,)

    if use_kl:
        kl = dirichlet_kl_per_sample(alpha, prior_alpha=prior_alpha)  # (B,)
        loss = loss + lambda_reg * kl

    return loss.mean()


#weights = {'C1': (w_pos, w_neg),
#           'C2': (w_pos, w_neg),
#           'C3': (w_pos, w_neg)}
# w_pos e.g. 1.76
# w_neg e.g. 0.54
#prior_alpha = { 'C1': (prior_a0, prior_a1),
#                'C2': (prior_a0, prior_a1),
#                'C3': (prior_a0, prior_a1)}
# prior_a0 = (1-pi)*v e.g. (1-0.05)*2 = 0.95*2 = 1.9
# prior_a1 = pi*v, e.g. 0.05*2 = 0.1

def total_bb_loss(x: torch.Tensor,
                  y: torch.Tensor,
                  weights: dict,
                  lambda_reg: float = 0.05,
                  use_kl: bool = True,
                  prior_alpha: dict | None = None,
                  n_annotators: int = 3) -> torch.Tensor:

    total_loss = 0.0
    for i, key in enumerate(['C1', 'C2', 'C3']):
        w_pos, w_neg = weights[key]

        pa = None
        if use_kl and prior_alpha is not None:
            pa = torch.tensor(prior_alpha[key], device=x[i].device, dtype=x[i].dtype)

        total_loss += evidential_bb_loss(x[i],
                                         y[:, i],
                                         w_pos=w_pos,
                                         w_neg=w_neg,
                                         lambda_reg=lambda_reg,
                                         use_kl=use_kl,
                                         prior_alpha=pa,
                                         n_annotators=n_annotators)
    return total_loss