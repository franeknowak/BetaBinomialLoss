import torch
from scripts.bbkl_loss import total_bb_loss
import torch.nn.functional as F

def build_loss_fn(CONFIG, device):
    """
        Loss function selector for three head classifier. This will break if the classifier is 1 head with N label output. 
    """
    loss_name = CONFIG['TRAIN']['LOSS']
    dataset_name = CONFIG['DATA']['DATASET_NAME']
    if loss_name == 'bbl':
        weights         = CONFIG['DATASETS'][dataset_name]['BBL_WEIGHTS']
        use_kl          = CONFIG['TRAIN']['USE_KL']
        use_prior_alpha = CONFIG['TRAIN']['USE_PRIOR_ALPHA']
        prior_alpha = CONFIG['DATASETS'][dataset_name]['PRIOR_ALPHA']
        if use_kl and use_prior_alpha:
            prior_alpha = { 'C1': ((1 - prior_alpha['PI_C1']) * prior_alpha['NU'], prior_alpha['PI_C1'] * prior_alpha['NU']),
                            'C2': ((1 - prior_alpha['PI_C2']) * prior_alpha['NU'], prior_alpha['PI_C2'] * prior_alpha['NU']),
                            'C3': ((1 - prior_alpha['PI_C3']) * prior_alpha['NU'], prior_alpha['PI_C3'] * prior_alpha['NU'])}
        elif use_kl and not use_prior_alpha:
            prior_alpha = { 'C1': (1, 1),
                            'C2': (1, 1),
                            'C3': (1, 1)}
        elif not use_kl and use_prior_alpha:
            raise ValueError("USE_KL and USE_PRIOR_ALPHA: You can not use prior alpha when kl is not enabled.")

        def loss_fn(output, labels):
            return total_bb_loss(
                output, labels,
                weights=weights,
                use_kl=use_kl,
                prior_alpha=prior_alpha,
            )
        return loss_fn

    elif loss_name == 'bce':
        class_weights = torch.tensor(
            CONFIG['DATASETS'][dataset_name]['BCE_POS_CLASS_WEIGHTS'],
            device=device,
        )

        def loss_fn(output, labels):
            total = 0.0
            for i in range(3):
                total += F.binary_cross_entropy_with_logits(
                    output[i].squeeze(-1),
                    labels[:, i].float(),
                    weight=class_weights[i],
                )
            return total
        return loss_fn

    else:
        raise ValueError(f"Unknown loss: {loss_name}")