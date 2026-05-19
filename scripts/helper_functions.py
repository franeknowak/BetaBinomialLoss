import math
import torch

def get_schedulers(optimizer, CONFIG, len_train_dataloader):
    ACC_STEPS        = CONFIG['TRAIN']['GRADIENT_ACC_BATCH_SIZE'] // CONFIG['TRAIN']['BATCH_SIZE']
    _steps_per_epoch = math.ceil(len_train_dataloader / ACC_STEPS)
    _warmup_steps    = CONFIG['TRAIN']['WARMUP_EPOCHS'] * _steps_per_epoch
    _decay_epochs    = max(1, CONFIG['TRAIN']['EPOCHS'] - CONFIG['TRAIN']['WARMUP_EPOCHS'])

    def _cosine_fn(start_lr, end_lr, T_max):
        def fn(epoch):
            cos_val = math.cos(math.pi * epoch / T_max)
            return (end_lr + 0.5 * (start_lr - end_lr) * (1 + cos_val)) / start_lr
        return fn

    # LambdaLR stores base_lrs = TARGET at creation time (before warmup touches the lrs),
    # so base_lr * fn(epoch) = TARGET * (cosine_lr / TARGET) = cosine_lr as intended.
    # Must be created before LinearLR for the same reason.
    lambdas = [_cosine_fn(g['lr'], g['end_lr'], _decay_epochs) for g in optimizer.param_groups]
    cosine_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 1e-8,
        end_factor   = 1.0,
        total_iters  = _warmup_steps,
    )

    return ACC_STEPS, warmup_scheduler, cosine_scheduler

def inspect_model(model, img_size):
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
    x = torch.randn(1, 3, img_size, img_size)
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

def dummy_output_dict(uncerts = False):
    """Creates a dummy output_dict used in training and eval"""
    output_dict = { 'C1':  {'probs':     [],
                            'preds':     []},
                    'C2':  {'probs':     [],
                            'preds':     []},
                    'C3':  {'probs':     [],
                            'preds':     []},
                    'labels':            [],
                    'vid_ids':           [],
                    'frame_ids':         []}
    
    if uncerts:
        for criterion in ['C1', 'C2', 'C3']:
            output_dict[criterion]['uncerts'] = []
    return output_dict

def _split_decay_params(named_params):
    "picks up layernorms and biases from the model. prevents from applying weight decay"
    decay, no_decay = [], []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        # Exclude biases and 1D params (LayerNorm weights) from decay
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay