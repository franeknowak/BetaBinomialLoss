import torch
from scripts.bbkl_loss import total_bb_loss
import torch.nn.functional as F

def build_loss_fn(CONFIG, device):
    """
        Loss function selector for three head classifier. This will break if the classifier is 1 head with N label output. 
    """
    loss_name = CONFIG['TRAIN']['LOSS']

    if loss_name == 'bbl':
        weights     = CONFIG['TRAIN']['BBL_PARAMS']['WEIGHTS']
        use_kl      = CONFIG['TRAIN']['BBL_PARAMS']['USE_KL']
        prior_alpha = CONFIG['TRAIN']['BBL_PARAMS']['PRIOR_ALPHA']
        if use_kl and prior_alpha is not None:
            prior_alpha = {
                'C1': ((1 - prior_alpha['PI_C1']) * prior_alpha['NU'], prior_alpha['PI_C1'] * prior_alpha['NU']),
                'C2': ((1 - prior_alpha['PI_C2']) * prior_alpha['NU'], prior_alpha['PI_C2'] * prior_alpha['NU']),
                'C3': ((1 - prior_alpha['PI_C3']) * prior_alpha['NU'], prior_alpha['PI_C3'] * prior_alpha['NU']),
            }

        def loss_fn(output, labels):
            return total_bb_loss(
                output, labels,
                weights=weights,
                use_kl=use_kl,
                prior_alpha=prior_alpha,
            )
        return loss_fn

    elif loss_name == 'bce':
        dataset_name  = CONFIG['DATA']['DATASET_NAME']
        class_weights = torch.tensor(
            CONFIG['DATA']['DATASETS'][dataset_name]['CLASS_WEIGHTS'],
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