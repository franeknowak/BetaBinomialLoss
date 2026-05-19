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

def validate_config(CONFIG):
    errors   = []
    warnings = []

    def err(msg):  errors.append(msg)
    def warn(msg): warnings.append(msg)

    def _check_keys(d, *keys, context):
        for k in keys:
            if k not in d:
                err(f"Missing required key '{k}' in {context}")

    # ── Top-level ─────────────────────────────────────────────────────────
    _check_keys(CONFIG, 'EXPERIMENT_NAME', 'SEED', 'DATASET_DIR',
                'ANNOTATIONS_PATH', 'CHECKPOINT_DIR',
                'MODEL', 'DATA', 'TRAIN', 'DATASETS',
                context='config root')

    from pathlib import Path

    if 'ANNOTATIONS_PATH' in CONFIG and not Path(CONFIG['ANNOTATIONS_PATH']).exists():
        warn(f"ANNOTATIONS_PATH does not exist: {CONFIG['ANNOTATIONS_PATH']}")

    if 'CHECKPOINT_DIR' in CONFIG and not Path(CONFIG['CHECKPOINT_DIR']).exists():
        warn(f"CHECKPOINT_DIR does not exist: {CONFIG['CHECKPOINT_DIR']}")

    # ── MODEL.ENCODER ─────────────────────────────────────────────────────
    enc = CONFIG.get('MODEL', {}).get('ENCODER', {})
    if 'MODEL' in CONFIG:
        _check_keys(CONFIG['MODEL'], 'ENCODER', 'TEMPORAL', 'CLASSIFIER', context='MODEL')

    if enc:
        _check_keys(enc, 'NAME', 'IMG_SIZE', context='MODEL.ENCODER')

        if 'NAME' in enc:
            if not any(k in enc['NAME'] for k in ('swinv2', 'dinov3')):
                err(f"MODEL.ENCODER.NAME '{enc['NAME']}' not recognised — must contain 'swinv2' or 'dinov3'")

        if 'IMG_SIZE' in enc:
            if not isinstance(enc['IMG_SIZE'], int) or enc['IMG_SIZE'] <= 0:
                err("MODEL.ENCODER.IMG_SIZE must be a positive integer")

        if 'FROZEN_STAGES' in enc:
            fs = enc['FROZEN_STAGES']
            if not isinstance(fs, int) or fs < 0:
                err("MODEL.ENCODER.FROZEN_STAGES must be a non-negative integer")

        if 'FT_WEIGHTS' in enc:
            warn("MODEL.ENCODER.FT_WEIGHTS is present but FT_WEIGHTS loading is not currently implemented — key will be silently ignored")

    # ── MODEL.TEMPORAL ────────────────────────────────────────────────────
    temp = CONFIG.get('MODEL', {}).get('TEMPORAL', {})
    if temp:
        _check_keys(temp, 'NAME', context='MODEL.TEMPORAL')

        tname = temp.get('NAME')
        if tname not in ('lstm', 'gated_pooling', None):
            err(f"MODEL.TEMPORAL.NAME '{tname}' not recognised — must be 'lstm', 'gated_pooling', or null")

        if tname == 'lstm':
            if 'LSTM' not in temp:
                err("MODEL.TEMPORAL.LSTM block is required when TEMPORAL.NAME='lstm'")
            else:
                lstm = temp['LSTM']
                _check_keys(lstm, 'HIDDEN_SIZE', 'NUM_LAYERS', 'DROPOUT', context='MODEL.TEMPORAL.LSTM')
                if 'HIDDEN_SIZE' in lstm and (not isinstance(lstm['HIDDEN_SIZE'], int) or lstm['HIDDEN_SIZE'] <= 0):
                    err("MODEL.TEMPORAL.LSTM.HIDDEN_SIZE must be a positive integer")
                if 'NUM_LAYERS' in lstm and (not isinstance(lstm['NUM_LAYERS'], int) or lstm['NUM_LAYERS'] <= 0):
                    err("MODEL.TEMPORAL.LSTM.NUM_LAYERS must be a positive integer")
                if 'DROPOUT' in lstm and not (0.0 <= lstm['DROPOUT'] < 1.0):
                    err("MODEL.TEMPORAL.LSTM.DROPOUT must be in [0, 1)")

        if tname == 'gated_pooling':
            if 'GATED_POOLING' not in temp:
                err("MODEL.TEMPORAL.GATED_POOLING block is required when TEMPORAL.NAME='gated_pooling'")
            else:
                gp = temp['GATED_POOLING']
                _check_keys(gp, 'DROPOUT', context='MODEL.TEMPORAL.GATED_POOLING')
                if 'DROPOUT' in gp and not (0.0 <= gp['DROPOUT'] < 1.0):
                    err("MODEL.TEMPORAL.GATED_POOLING.DROPOUT must be in [0, 1)")
        if tname == 'lstm' and 'GATED_POOLING' in temp:
            warn("MODEL.TEMPORAL.GATED_POOLING is defined but TEMPORAL.NAME='lstm' — it will be ignored")
        if tname == 'gated_pooling' and 'LSTM' in temp:
            warn("MODEL.TEMPORAL.LSTM is defined but TEMPORAL.NAME='gated_pooling' — it will be ignored")
        if tname is None and 'LSTM' in temp:
            warn("MODEL.TEMPORAL.LSTM is defined but TEMPORAL.NAME=null (no temporal) — it will be ignored")
        if tname is None and 'GATED_POOLING' in temp:
            warn("MODEL.TEMPORAL.GATED_POOLING is defined but TEMPORAL.NAME=null (no temporal) — it will be ignored")
    # ── MODEL.CLASSIFIER ──────────────────────────────────────────────────
    cls = CONFIG.get('MODEL', {}).get('CLASSIFIER', {})
    if cls:
        _check_keys(cls, 'DROPOUT', context='MODEL.CLASSIFIER')
        if 'DROPOUT' in cls and not (0.0 <= cls['DROPOUT'] < 1.0):
            err("MODEL.CLASSIFIER.DROPOUT must be in [0, 1)")

    # ── DATA ──────────────────────────────────────────────────────────────
    data = CONFIG.get('DATA', {})
    if data:
        _check_keys(data, 'DATASET_NAME', 'TEMPORAL', 'LABEL_CRITERION', 'LABEL_METHOD', context='DATA')

        if 'LABEL_CRITERION' in data and data['LABEL_CRITERION'] not in (None, 0, 1, 2):
            err(f"DATA.LABEL_CRITERION must be null, 0, 1, or 2 — got '{data['LABEL_CRITERION']}'")

        if 'LABEL_METHOD' in data and data['LABEL_METHOD'] not in ('soft', 'hard'):
            err(f"DATA.LABEL_METHOD must be 'soft' or 'hard' — got '{data['LABEL_METHOD']}'")

    # ── TRAIN ─────────────────────────────────────────────────────────────
    train = CONFIG.get('TRAIN', {})
    if train:
        _check_keys(train, 'EPOCHS', 'BATCH_SIZE', 'GRADIENT_ACC_BATCH_SIZE',
                    'EARLY_PATIENCE', 'OPTIMIZER', 'LOSS', 'WARMUP_EPOCHS',
                    'FREEZE_ENCODER', 'LR', context='TRAIN')

        if 'EPOCHS' in train and (not isinstance(train['EPOCHS'], int) or train['EPOCHS'] <= 0):
            err("TRAIN.EPOCHS must be a positive integer")

        if 'BATCH_SIZE' in train and (not isinstance(train['BATCH_SIZE'], int) or train['BATCH_SIZE'] <= 0):
            err("TRAIN.BATCH_SIZE must be a positive integer")

        if 'GRADIENT_ACC_BATCH_SIZE' in train and 'BATCH_SIZE' in train:
            gacc = train['GRADIENT_ACC_BATCH_SIZE']
            bs   = train['BATCH_SIZE']
            if not isinstance(gacc, int) or gacc <= 0:
                err("TRAIN.GRADIENT_ACC_BATCH_SIZE must be a positive integer")
            elif isinstance(bs, int) and bs > 0:
                if gacc < bs:
                    err(f"TRAIN.GRADIENT_ACC_BATCH_SIZE ({gacc}) must be >= BATCH_SIZE ({bs})")
                elif gacc % bs != 0:
                    err(f"TRAIN.GRADIENT_ACC_BATCH_SIZE ({gacc}) must be divisible by BATCH_SIZE ({bs})")

        if 'WARMUP_EPOCHS' in train and 'EPOCHS' in train:
            we = train['WARMUP_EPOCHS']
            if not isinstance(we, int) or we < 0:
                err("TRAIN.WARMUP_EPOCHS must be a non-negative integer")
            elif isinstance(train['EPOCHS'], int) and we >= train['EPOCHS']:
                err(f"TRAIN.WARMUP_EPOCHS ({we}) must be less than TRAIN.EPOCHS ({train['EPOCHS']})")

        if 'EARLY_PATIENCE' in train and (not isinstance(train['EARLY_PATIENCE'], int) or train['EARLY_PATIENCE'] <= 0):
            err("TRAIN.EARLY_PATIENCE must be a positive integer")

        if 'FREEZE_ENCODER' in train and not isinstance(train['FREEZE_ENCODER'], bool):
            err("TRAIN.FREEZE_ENCODER must be a boolean (true/false)")

        # OPTIMIZER
        opt = train.get('OPTIMIZER', {})
        if opt:
            _check_keys(opt, 'EPS', 'BETAS', 'WEIGHT_DECAY', context='TRAIN.OPTIMIZER')
            if 'BETAS' in opt:
                b = opt['BETAS']
                if not (isinstance(b, (list, tuple)) and len(b) == 2 and all(0.0 < v < 1.0 for v in b)):
                    err("TRAIN.OPTIMIZER.BETAS must be a list of 2 floats in (0, 1)")
            if 'EPS' in opt and opt['EPS'] <= 0:
                err("TRAIN.OPTIMIZER.EPS must be positive")
            if 'WEIGHT_DECAY' in opt and opt['WEIGHT_DECAY'] < 0:
                err("TRAIN.OPTIMIZER.WEIGHT_DECAY must be non-negative")

        # LOSS
        loss = train.get('LOSS')
        if loss is not None:
            if loss not in ('bce', 'bbl'):
                err(f"TRAIN.LOSS '{loss}' not recognised — must be 'bce' or 'bbl'")

            if loss == 'bbl':
                _check_keys(train, 'USE_KL', 'USE_PRIOR_ALPHA', context='TRAIN (required when LOSS=bbl)')
                if 'USE_KL' in train and 'USE_PRIOR_ALPHA' in train:
                    if not train['USE_KL'] and train['USE_PRIOR_ALPHA']:
                        err("TRAIN.USE_PRIOR_ALPHA=True requires TRAIN.USE_KL=True")

            if loss == 'bce':
                if train.get('USE_KL') is not None or train.get('USE_PRIOR_ALPHA') is not None:
                    warn("TRAIN.USE_KL / USE_PRIOR_ALPHA are set but ignored when LOSS='bce' — possible copy-paste from a BBL config")

        # LR
        lr = train.get('LR', {})
        if lr:
            _check_keys(lr, 'TEMPORAL', 'CLASSIFIER', context='TRAIN.LR')

            for group in ('TEMPORAL', 'CLASSIFIER'):
                g = lr.get(group, {})
                if g:
                    _check_keys(g, 'TARGET', 'END', context=f'TRAIN.LR.{group}')
                    if 'TARGET' in g and 'END' in g:
                        if g['TARGET'] <= 0:
                            err(f"TRAIN.LR.{group}.TARGET must be positive")
                        if g['END'] <= 0:
                            err(f"TRAIN.LR.{group}.END must be positive")
                        if g['END'] >= g['TARGET']:
                            warn(f"TRAIN.LR.{group}.END ({g['END']}) >= TARGET ({g['TARGET']}) — cosine will ascend, is this intended?")

            freeze = train.get('FREEZE_ENCODER')
            if freeze is False:
                if 'ENCODER' not in lr:
                    err("TRAIN.LR.ENCODER is required when FREEZE_ENCODER=False")
                else:
                    enc_lr = lr['ENCODER']
                    _check_keys(enc_lr, 'TARGET', 'END', context='TRAIN.LR.ENCODER')
                    if 'TARGET' in enc_lr and 'END' in enc_lr and enc_lr['END'] >= enc_lr['TARGET']:
                        warn(f"TRAIN.LR.ENCODER.END ({enc_lr['END']}) >= TARGET ({enc_lr['TARGET']}) — cosine will ascend")

            if freeze is True and 'ENCODER' in lr:
                warn("TRAIN.LR.ENCODER is defined but FREEZE_ENCODER=True — encoder LR block will not be used")

    # ── DATASETS ──────────────────────────────────────────────────────────
    dataset_name = CONFIG.get('DATA', {}).get('DATASET_NAME')
    datasets     = CONFIG.get('DATASETS', {})
    loss         = CONFIG.get('TRAIN', {}).get('LOSS')

    if dataset_name and datasets:
        if dataset_name not in datasets:
            err(f"DATA.DATASET_NAME '{dataset_name}' not found in DATASETS block")
        else:
            ds = datasets[dataset_name]
            _check_keys(ds, 'MEAN', 'STD', 'CENTER_CROP', 'BCE_POS_CLASS_WEIGHTS',
                        context=f'DATASETS.{dataset_name}')

            if loss == 'bbl':
                _check_keys(ds, 'BBL_WEIGHTS', 'PRIOR_ALPHA',
                            context=f'DATASETS.{dataset_name} (required when LOSS=bbl)')
                if 'BBL_WEIGHTS' in ds:
                    _check_keys(ds['BBL_WEIGHTS'], 'C1', 'C2', 'C3',
                                context=f'DATASETS.{dataset_name}.BBL_WEIGHTS')
                if 'PRIOR_ALPHA' in ds:
                    _check_keys(ds['PRIOR_ALPHA'], 'NU', 'PI_C1', 'PI_C2', 'PI_C3',
                                context=f'DATASETS.{dataset_name}.PRIOR_ALPHA')

    # ── Cross-field ────────────────────────────────────────────────────────
    data_temporal  = CONFIG.get('DATA', {}).get('TEMPORAL')
    model_temporal = CONFIG.get('MODEL', {}).get('TEMPORAL', {}).get('NAME')

    if data_temporal is True and model_temporal is None:
        warn("DATA.TEMPORAL=True but MODEL.TEMPORAL.NAME=null — full sequences are loaded but only the last frame is used")
    if data_temporal is False and model_temporal in ('lstm', 'gated_pooling'):
        warn(f"DATA.TEMPORAL=False but MODEL.TEMPORAL.NAME='{model_temporal}' — single frames will be fed as T=1 sequences")

    freeze    = CONFIG.get('TRAIN', {}).get('FREEZE_ENCODER')
    fs        = CONFIG.get('MODEL', {}).get('ENCODER', {}).get('FROZEN_STAGES', 0)
    if freeze is True and isinstance(fs, int) and fs > 0:
        warn("MODEL.ENCODER.FROZEN_STAGES is set but FREEZE_ENCODER=True — FROZEN_STAGES will be ignored")

    # ── Experiment name heuristics ─────────────────────────────────────────
    exp  = CONFIG.get('EXPERIMENT_NAME', '').lower()
    enc_name = CONFIG.get('MODEL', {}).get('ENCODER', {}).get('NAME', '').lower()

    if loss == 'bbl' and 'bce' in exp:
        warn(f"EXPERIMENT_NAME contains 'bce' but LOSS='bbl'")
    if loss == 'bce' and 'bbl' in exp:
        warn(f"EXPERIMENT_NAME contains 'bbl' but LOSS='bce'")

    if model_temporal == 'lstm' and 'notemp' in exp:
        warn("EXPERIMENT_NAME contains 'notemp' but MODEL.TEMPORAL.NAME='lstm'")
    if model_temporal is None and 'lstm' in exp:
        warn("EXPERIMENT_NAME contains 'lstm' but MODEL.TEMPORAL.NAME=null")

    if 'swinv2' in enc_name and 'dino' in exp:
        warn("EXPERIMENT_NAME contains 'dino' but encoder is SwinV2")
    if 'dinov3' in enc_name and 'swin' in exp:
        warn("EXPERIMENT_NAME contains 'swin' but encoder is DINOv3")

    if freeze is True and 'e2e' in exp:
        warn("EXPERIMENT_NAME contains 'e2e' but FREEZE_ENCODER=True")
    if freeze is False and 'frozen' in exp:
        warn("EXPERIMENT_NAME contains 'frozen' but FREEZE_ENCODER=False")

    label_method = CONFIG.get('DATA', {}).get('LABEL_METHOD', '')
    if label_method == 'hard' and 'soft' in exp:
        warn("EXPERIMENT_NAME contains 'soft' but LABEL_METHOD='hard'")
    if label_method == 'soft' and 'hard' in exp:
        warn("EXPERIMENT_NAME contains 'hard' but LABEL_METHOD='soft'")

    # ── Report ─────────────────────────────────────────────────────────────
    if warnings:
        print('\nCONFIG WARNINGS:')
        for w in warnings:
            print(f'  ⚠  {w}')
        print()

    if errors:
        raise ValueError(
            f'\nConfig validation failed with {len(errors)} error(s):\n' +
            '\n'.join(f'  ✗  {e}' for e in errors) + '\n'
        )