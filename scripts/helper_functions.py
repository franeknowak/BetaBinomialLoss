import math
import torch

def get_parameter_groups(model, classifier_dict):
    temporal_proc_params = []
    head_params = []

    if classifier_dict['temporal_processing'] == 'gated_pooling' and classifier_dict['head'] == 'evidential':
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("heads"):
                head_params.append(p)
            elif name.startswith("pool"):
                temporal_proc_params.append(p)
    else: raise Exception("Functionality not implemented. Only option available is 'gated_pooling' and 'evidential'")

    return temporal_proc_params, head_params

def get_schedulers(optimizer, CONFIG, len_train_dataloader):
        # Necessary vars
        ACC_STEPS   = CONFIG['TRAIN']['GRADIENT_ACC_BATCH_SIZE'] // CONFIG['TRAIN']['BATCH_SIZE']
        _steps_per_epoch     = math.ceil(len_train_dataloader / ACC_STEPS)
        _warmup_steps        = CONFIG['TRAIN']['WARMUP_EPOCHS'] * _steps_per_epoch
        _decay_epochs        = CONFIG['TRAIN']['EPOCHS'] - CONFIG['TRAIN']['WARMUP_EPOCHS']

        # Cosine has to go first because at __init__ it automatically takes a step
        # Putting it after warmup would store lr=0 as base, messing it up
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(  optimizer,
                                                                        T_max   = max(1, _decay_epochs),
                                                                        eta_min = CONFIG['TRAIN']['ENCODER_LR']['END'])
        
        #
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(   optimizer,
                                                                start_factor = 1e-8,
                                                                end_factor   = 1.0,
                                                                total_iters  = _warmup_steps)
        
        
        
        return ACC_STEPS, warmup_scheduler, cosine_scheduler