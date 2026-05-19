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

def print_run_header(model, CONFIG, optimizer):
    W   = 62
    SEP = '═' * W
    DIV = '─' * W

    experiment_name = CONFIG['EXPERIMENT_NAME'] + str(CONFIG['SEED'])
    loss            = CONFIG['TRAIN']['LOSS']
    freeze          = CONFIG['TRAIN']['FREEZE_ENCODER']
    frozen_stages   = CONFIG['MODEL']['ENCODER'].get('FROZEN_STAGES', 0)

    # ── Header ────────────────────────────────────────────────────
    print(f'\n{SEP}')
    print(f'  EXPERIMENT: {experiment_name}')
    print(SEP)

    # ── Data / Loss ───────────────────────────────────────────────
    dataset  = CONFIG['DATA']['DATASET_NAME']
    labels   = CONFIG['DATA']['LABEL_METHOD']
    temporal = CONFIG['DATA']['TEMPORAL']
    print(f"  Data      {dataset}  |  Labels: {labels}  |  Temporal: {temporal}")

    if loss == 'bbl':
        print(f"  Loss      BBL  (KL={CONFIG['TRAIN']['USE_KL']}, Prior={CONFIG['TRAIN']['USE_PRIOR_ALPHA']})")
    else:
        print(f"  Loss      BCE")
    print(f"  Seed      {CONFIG['SEED']}")

    # ── Architecture ──────────────────────────────────────────────
    print(f'\n  ARCHITECTURE')

    enc_name = CONFIG['MODEL']['ENCODER']['NAME']
    if freeze:
        freeze_str = '[FROZEN]'
    elif frozen_stages > 0:
        freeze_str = f'[PARTIAL — stages 0–{frozen_stages - 1} frozen]'
    else:
        freeze_str = '[E2E]'
    print(f"  {'Encoder':<12}{enc_name}")
    print(f"  {'':12}{freeze_str}")

    temp_name = CONFIG['MODEL']['TEMPORAL']['NAME']
    if temp_name == 'lstm':
        lstm     = CONFIG['MODEL']['TEMPORAL']['LSTM']
        temp_str = f"LSTM  hidden={lstm['HIDDEN_SIZE']}  layers={lstm['NUM_LAYERS']}  dropout={lstm['DROPOUT']}"
    elif temp_name == 'gated_pooling':
        gp       = CONFIG['MODEL']['TEMPORAL']['GATED_POOLING']
        temp_str = f"GatedPooling  dropout={gp['DROPOUT']}"
    elif temp_name is None:
        temp_str = 'None  (single-frame)'
    else:
        temp_str = temp_name
    print(f"  {'Temporal':<12}{temp_str}")

    cls_type = 'EvidentialHead' if loss == 'bbl' else 'BCEHead'
    print(f"  {'Classifier':<12}{cls_type}  dropout={CONFIG['MODEL']['CLASSIFIER']['DROPOUT']}")

    # ── Parameters ────────────────────────────────────────────────
    print(f'\n  PARAMETERS')
    print(f"  {'Module':<14}{'Total':>14}{'Trainable':>14}{'Frozen':>14}")
    print(f'  {DIV}')

    grand_total = grand_train = 0
    for name, module in model.named_children():
        if name == 'spatial_pool':
            continue
        total       = sum(p.numel() for p in module.parameters())
        trainable   = sum(p.numel() for p in module.parameters() if p.requires_grad)
        grand_total += total
        grand_train += trainable
        print(f"  {name:<14}{total:>14,}{trainable:>14,}{total - trainable:>14,}")

    print(f'  {DIV}')
    print(f"  {'TOTAL':<14}{grand_total:>14,}{grand_train:>14,}{grand_total - grand_train:>14,}")
    print(f"  Trainable: {grand_train / grand_total:.2%}")

    # Encoder leak: assertion rather than print — should never be silently wrong
    leaked = [n for n, p in model.encoder.named_parameters() if p.requires_grad]
    assert not (freeze and leaked), \
        f"Encoder param leak — {len(leaked)} params have requires_grad=True despite FREEZE_ENCODER=True"

    # ── Training ──────────────────────────────────────────────────
    print(f'\n  TRAINING')
    bs  = CONFIG['TRAIN']['BATCH_SIZE']
    acc = CONFIG['TRAIN']['GRADIENT_ACC_BATCH_SIZE']
    acc_steps = acc // bs
    print(f"  Epochs {CONFIG['TRAIN']['EPOCHS']}  |  "
          f"Batch {bs}  |  "
          f"Acc {acc_steps} steps  |  "
          f"Effective batch {acc}")
    print(f"  Warmup {CONFIG['TRAIN']['WARMUP_EPOCHS']} epoch(s)  |  "
          f"Early stopping patience {CONFIG['TRAIN']['EARLY_PATIENCE']}")

    opt = CONFIG['TRAIN']['OPTIMIZER']
    print(f"  AdamW  β={opt['BETAS']}  ε={opt['EPS']}  WD={opt['WEIGHT_DECAY']}")

    groups = [(g['name'], g['lr'], g['end_lr'])
          for g in optimizer.param_groups
          if not g['name'].endswith('_nd') and len(g['params']) > 0]
    for i, (name, start, end) in enumerate(groups):
        prefix = '  LR' if i == 0 else '    '
        print(f"  {prefix}  {name:<14}{start:.2e} → {end:.2e}")

    print(f'{SEP}\n')

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

def print_epoch_summary(train_r, val_r, epoch, total_epochs, optimizer, is_best):
    W   = 62
    DIV = '─' * W

    print(f'\n{DIV}')
    print(f'  Epoch {epoch + 1:02d}/{total_epochs:02d}')
    print(DIV)

    print(f"  {'':22}{'Train':>9}{'Val':>9}")
    print(f"  {'Loss':<22}{train_r['loss']:>9.4f}{val_r['loss']:>9.4f}")
    print(f"  {'Avg Accuracy':<22}{train_r['avg_accuracy']:>9.4f}{val_r['avg_accuracy']:>9.4f}")
    print(f"  {'Avg BAcc':<22}{train_r['avg_bacc']:>9.4f}{val_r['avg_bacc']:>9.4f}  ←")
    print(f"  {'mAP':<22}{train_r['mAP']:>9.4f}{val_r['mAP']:>9.4f}")
    print()

    print(f"  {'':4}  {'─── Train ──────':^23}  {'─── Val ────────':^23}")
    print(f"  {'':4}  {'Acc':>7}{'BAcc':>8}{'AP':>8}  {'Acc':>7}{'BAcc':>8}{'AP':>8}")
    for c in ['C1', 'C2', 'C3']:
        t = (train_r[f'accuracy_{c}'], train_r[f'bal_accuracy_{c}'], train_r[f'ap_{c}'])
        v = (val_r[f'accuracy_{c}'],   val_r[f'bal_accuracy_{c}'],   val_r[f'ap_{c}'])
        print(f"  {c:<4}  {t[0]:>7.3f}{t[1]:>8.3f}{t[2]:>8.3f}  {v[0]:>7.3f}{v[1]:>8.3f}{v[2]:>8.3f}")

    print()
    groups  = [(g['name'], g['lr']) for g in optimizer.param_groups
               if not g['name'].endswith('_nd') and len(g['params']) > 0]
    lr_str  = '  |  '.join(f"{n}: {lr:.2e}" for n, lr in groups)
    print(f"  LR  {lr_str}")

    print(DIV)
    if is_best:
        print(f"  ★  New best — Val BAcc {val_r['avg_bacc']:.4f} — weights saved")
        print(DIV)
    print()


def print_test_summary(test_r, best_epoch):
    W   = 62
    SEP = '═' * W

    print(f'\n{SEP}')
    print(f'  TEST RESULTS  (weights from epoch {best_epoch})')
    print(SEP)

    print(f"  {'Loss':<22}{test_r['loss']:>9.4f}")
    print(f"  {'Avg Accuracy':<22}{test_r['avg_accuracy']:>9.4f}")
    print(f"  {'Avg BAcc':<22}{test_r['avg_bacc']:>9.4f}")
    print(f"  {'mAP':<22}{test_r['mAP']:>9.4f}")
    print()

    print(f"  {'':4}  {'Acc':>7}{'BAcc':>8}{'AP':>8}")
    for c in ['C1', 'C2', 'C3']:
        print(f"  {c:<4}  {test_r[f'accuracy_{c}']:>7.3f}"
              f"{test_r[f'bal_accuracy_{c}']:>8.3f}"
              f"{test_r[f'ap_{c}']:>8.3f}")

    print(f'\n{SEP}\n')