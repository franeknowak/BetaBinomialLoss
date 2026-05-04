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
        try:
            # This works if we are settin up the scheduler for encoder pretraining
            eta_min = CONFIG['TRAIN']['ENCODER_LR']['END']
        except:
            # This works for the main experiment with temporal part
            eta_min = CONFIG['TRAIN']['TEMPORAL_LR']['END']
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(  optimizer,
                                                                        T_max   = max(1, _decay_epochs),
                                                                        eta_min = eta_min)
        
        #
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(   optimizer,
                                                                start_factor = 1e-8,
                                                                end_factor   = 1.0,
                                                                total_iters  = _warmup_steps)
        
        
        
        return ACC_STEPS, warmup_scheduler, cosine_scheduler

def inspect_model(model):
    # 1. Top-level parameter breakdown
    print(f"{'Module':<20} {'Total params':>15} {'Trainable':>15} {'Frozen':>15}")
    print("-" * 68)
    grand_total, grand_train = 0, 0
    for name, module in model.named_children():
        total = sum(p.numel() for p in module.parameters())
        train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        grand_total += total
        grand_train += train
        print(f"{name:<20} {total:>15,} {train:>15,} {total-train:>15,}")
    print("-" * 68)
    print(f"{'TOTAL':<20} {grand_total:>15,} {grand_train:>15,} {grand_total-grand_train:>15,}")
    print(f"Trainable fraction: {grand_train/grand_total:.4%}\n")

    # 2. List every trainable parameter (should only be temporal + heads)
    print("Trainable parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {name:<60} {tuple(p.shape)}")

    # 3. Sanity-check that NO encoder param is trainable
    leaked = [n for n, p in model.encoder.named_parameters() if p.requires_grad]
    print(f"\nEncoder params with requires_grad=True: {len(leaked)} "
          f"({'OK' if not leaked else 'LEAK — investigate'})")

    # 4. Train/eval mode check (run after model.train())
    model.train()
    print(f"\nAfter model.train():")
    print(f"  encoder.training:  {model.encoder.training}  "
          f"({'WARNING: DropPath active' if model.encoder.training else 'OK'})")
    print(f"  temporal.training: {model.temporal.training}")
    print(f"  heads.training:    {model.heads.training}")

    # 5. Determinism check on the encoder (catches the DropPath issue empirically)
    model.eval()
    x = torch.randn(1, 3, 256, 256)  # adjust to your input size
    with torch.no_grad():
        z1 = model.encoder(x)
        z2 = model.encoder(x)
    print(f"\nEncoder deterministic in eval: {torch.allclose(z1, z2)}")
    model.train()
    with torch.no_grad():
        z1 = model.encoder(x)
        z2 = model.encoder(x)
    print(f"Encoder deterministic in train: {torch.allclose(z1, z2)} "
          f"(False = DropPath is firing on frozen encoder)")