print('Importing libraries...')
import json
import argparse
from pathlib import Path
import warnings 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import v2

from scripts.env import set_deterministic_behaviour, get_config
from scripts.dataset import EndoscapesDataset
from scripts.model import build_model
from scripts.helper_functions import get_schedulers, inspect_model, dummy_output_dict, _split_decay_params
from scripts.build_loss import build_loss_fn
from scripts.metrics import update_model_output_dict, calculate_metrics

warnings.filterwarnings("ignore")

############################################################################################
############################################################################################
# Read config
parser = argparse.ArgumentParser(description="Specify relevant config")
parser.add_argument('--config_path', type=str, required=True, help='Path to config YAML file')
args = parser.parse_args()
CONFIG = get_config(args.config_path)

EXPERIMENT_NAME = CONFIG['EXPERIMENT_NAME']+str(CONFIG['SEED'])
DATASET_NAME = CONFIG['DATA']['DATASET_NAME']

# Check CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
else:
    device = torch.device("cpu")

# For reproducible results
set_deterministic_behaviour(CONFIG['SEED'])

############################################################################################
############################################################################################
transforms =  v2.Compose([  v2.CenterCrop(CONFIG['DATASETS'][DATASET_NAME]['CENTER_CROP']),
                            v2.Resize((CONFIG['MODEL']['ENCODER']['IMG_SIZE'], CONFIG['MODEL']['ENCODER']['IMG_SIZE'])),
                            v2.ToImage(),                
                            v2.ToDtype(torch.float32, scale=True),  
                            v2.Normalize(mean=CONFIG['DATASETS'][DATASET_NAME]['MEAN'], std=CONFIG['DATASETS'][DATASET_NAME]['STD'])])

# Datasets
dataset_train = EndoscapesDataset(  split = 'train',
                                    transforms = transforms,
                                    temporal = CONFIG['DATA']['TEMPORAL'],
                                    label_criterion = (CONFIG['DATA']['LABEL_CRITERION'], CONFIG['DATA']['LABEL_METHOD']),
                                    annotations_path = CONFIG['ANNOTATIONS_PATH'],
                                    dataset_dir = CONFIG['DATASET_DIR'])
dataset_val =   EndoscapesDataset(  split = 'val',
                                    transforms = transforms,
                                    temporal = CONFIG['DATA']['TEMPORAL'],
                                    label_criterion = (CONFIG['DATA']['LABEL_CRITERION'], CONFIG['DATA']['LABEL_METHOD']),
                                    annotations_path = CONFIG['ANNOTATIONS_PATH'],
                                    dataset_dir = CONFIG['DATASET_DIR'])
dataset_test =  EndoscapesDataset(  split = 'test',
                                    transforms = transforms,
                                    temporal = CONFIG['DATA']['TEMPORAL'],
                                    label_criterion = (CONFIG['DATA']['LABEL_CRITERION'], CONFIG['DATA']['LABEL_METHOD']),
                                    annotations_path = CONFIG['ANNOTATIONS_PATH'],
                                    dataset_dir = CONFIG['DATASET_DIR'])

# Dataloaders
train_dataloader =  DataLoader( dataset_train,
                                batch_size = CONFIG['TRAIN']['BATCH_SIZE'],
                                pin_memory = True,
                                drop_last= False,
                                shuffle = True)

val_dataloader =    DataLoader( dataset_val,
                                batch_size = CONFIG['TRAIN']['BATCH_SIZE'],
                                pin_memory = True,
                                drop_last= False,
                                shuffle = False)

test_dataloader =   DataLoader( dataset_test,
                                batch_size = CONFIG['TRAIN']['BATCH_SIZE'],
                                pin_memory = True,
                                drop_last= False,
                                shuffle = False)

############################################################################################
############################################################################################
# Init Backbone
model = build_model(CONFIG)
inspect_model(model, img_size=CONFIG['MODEL']['ENCODER']['IMG_SIZE'])

freeze_encoder = CONFIG['TRAIN']['FREEZE_ENCODER']

if not freeze_encoder:
    encoder_decay, encoder_no_decay = _split_decay_params(
        [(n, p) for n, p in model.named_parameters() if n.startswith("encoder")]
    )

temporal_decay, temporal_no_decay = _split_decay_params(
    [(n, p) for n, p in model.named_parameters() if n.startswith("temporal")]
)
classifier_decay, classifier_no_decay = _split_decay_params(
    [(n, p) for n, p in model.named_parameters() if n.startswith("heads")]
)

# Sanity check: all trainable params are accounted for
accounted = temporal_decay + temporal_no_decay + classifier_decay + classifier_no_decay
if not freeze_encoder:
    accounted += encoder_decay + encoder_no_decay
expected = sum(p.numel() for p in accounted)
actual   = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert expected == actual, f"Parameter accounting mismatch: {expected} vs {actual}"

TEMPORAL_LR    = CONFIG['TRAIN']['LR']['TEMPORAL']
CLASSIFIER_LR  = CONFIG['TRAIN']['LR']['CLASSIFIER']
WD             = CONFIG['TRAIN']['OPTIMIZER']['WEIGHT_DECAY']

param_groups = [
    {"params": temporal_decay,      "lr": TEMPORAL_LR['TARGET'],   "end_lr": TEMPORAL_LR['END'],   "weight_decay": WD,  "name": "temporal"},
    {"params": temporal_no_decay,   "lr": TEMPORAL_LR['TARGET'],   "end_lr": TEMPORAL_LR['END'],   "weight_decay": 0.0, "name": "temporal_nd"},
    {"params": classifier_decay,    "lr": CLASSIFIER_LR['TARGET'], "end_lr": CLASSIFIER_LR['END'], "weight_decay": WD,  "name": "classifier"},
    {"params": classifier_no_decay, "lr": CLASSIFIER_LR['TARGET'], "end_lr": CLASSIFIER_LR['END'], "weight_decay": 0.0, "name": "classifier_nd"},
]

if not freeze_encoder:
    ENCODER_LR = CONFIG['TRAIN']['LR']['ENCODER']
    param_groups = [
        {"params": encoder_decay,    "lr": ENCODER_LR['TARGET'], "end_lr": ENCODER_LR['END'], "weight_decay": WD,  "name": "encoder"},
        {"params": encoder_no_decay, "lr": ENCODER_LR['TARGET'], "end_lr": ENCODER_LR['END'], "weight_decay": 0.0, "name": "encoder_nd"},
    ] + param_groups

optimizer = optim.AdamW(
    param_groups,
    betas=CONFIG['TRAIN']['OPTIMIZER']['BETAS'],
    eps=CONFIG['TRAIN']['OPTIMIZER']['EPS'],
)

model.to(device)

ACCUMULATION_STEPS, warmup_scheduler, cosine_scheduler = get_schedulers(optimizer, CONFIG, len(train_dataloader))

loss_fn = build_loss_fn(CONFIG, device)
EVIDENTIAL = (CONFIG['TRAIN']['LOSS'] == 'bbl')

############################################################################################
############################################################################################
# MODEL TRAINING AND EVALUATION
results_dict = {}

best_bacc_across_epochs = -1.0
best_epoch = 0
epochs_without_improvement = 0

EPOCHS = CONFIG['TRAIN']['EPOCHS']

for epoch in range(EPOCHS):
        print(f"Epoch: {epoch+1:02}/{EPOCHS:02}")

        print("Training")
        train_loss_sum = 0.0
        len_train_loader = len(train_dataloader)
        train_output_dict = dummy_output_dict(uncerts=EVIDENTIAL)

        model.train()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        for idx, (images, labels, vid_id, frame_id) in enumerate(train_dataloader):
                print(f'\r{idx+1}/{len_train_loader}', end='', flush=True)

                images, labels = images.to(device), labels.to(device)
                output = model(images)
                train_loss_per_acc_batch= loss_fn(output, labels) / ACCUMULATION_STEPS
                
                train_loss_per_acc_batch.backward()

                if (idx + 1) % ACCUMULATION_STEPS == 0 or (idx + 1) == len_train_loader:
                        optimizer.step()
                        if epoch < CONFIG['TRAIN']['WARMUP_EPOCHS']:
                                warmup_scheduler.step()
                        optimizer.zero_grad()

                # Populate the output dict with probs and preds per batch per class
                train_output_dict = update_model_output_dict(output, train_output_dict, evidential = EVIDENTIAL)
                train_output_dict['labels'].append(labels.detach().cpu())
                train_output_dict['vid_ids'].append(vid_id)
                train_output_dict['frame_ids'].append(frame_id)
                train_loss_sum += train_loss_per_acc_batch.item() * ACCUMULATION_STEPS

        results, train_output_dict = calculate_metrics(train_output_dict, evidential = EVIDENTIAL)

        avg_train_loss = train_loss_sum / len_train_loader
        results['loss'] = round(avg_train_loss, 4)

        print(f"\n--- Training Metrics ---")
        print(f"Train Avg Accuracy              {results['avg_accuracy']:.4f}")
        print(f"Train Avg BAcc                  {results['avg_bacc']:.4f}")
        print(f"Train mAP                       {results['mAP']:.4f}")
        print(f"Train Loss:                     {results['loss']:.4f}\n")
        print(f"Train C1 Accuracy               {results['accuracy_C1']:.4f}")
        print(f"Train C2 Accuracy               {results['accuracy_C2']:.4f}")
        print(f"Train C3 Accuracy               {results['accuracy_C3']:.4f}\n")
        print(f"Train C1 Balanced Accuracy:     {results['bal_accuracy_C1']:.4f}")
        print(f"Train C2 Balanced Accuracy:     {results['bal_accuracy_C2']:.4f}")
        print(f"Train C3 Balanced Accuracy:     {results['bal_accuracy_C3']:.4f}\n")
        print(f"Train C1 AP:                    {results['ap_C1']:.4f}")
        print(f"Train C2 AP:                    {results['ap_C2']:.4f}")
        print(f"Train C3 AP:                    {results['ap_C3']:.4f}")
        print(f"------------------------\n")
        results_dict[f"Epoch {epoch+1} Train"] = results

        print('Validation')
        val_loss_sum = 0.0
        len_val_loader = len(val_dataloader)
        val_output_dict = dummy_output_dict(uncerts = EVIDENTIAL)

        model.eval()
        torch.cuda.synchronize()

        with torch.inference_mode():
                for idx, (images, labels, vid_id, frame_id) in enumerate(val_dataloader):
                        print(f'\r{idx+1}/{len_val_loader}', end='', flush=True)

                        images, labels = images.to(device), labels.to(device)
                        output = model(images)
                        val_loss_per_batch = loss_fn(output, labels)

                        val_output_dict = update_model_output_dict(output, val_output_dict, evidential = EVIDENTIAL)
                        val_output_dict['labels'].append(labels.detach().cpu())
                        val_output_dict['vid_ids'].append(vid_id)
                        val_output_dict['frame_ids'].append(frame_id)
                        val_loss_sum += val_loss_per_batch.item()

        results, val_output_dict = calculate_metrics(val_output_dict, evidential = EVIDENTIAL)
        avg_val_loss = val_loss_sum / len_val_loader
        results['loss'] = round(avg_val_loss, 4)
        print(f"\n--- Validation Metrics ---")
        print(f"Val Avg Accuracy              {results['avg_accuracy']:.4f}")
        print(f"Val Avg BAcc                  {results['avg_bacc']:.4f}")
        print(f"Val mAP                       {results['mAP']:.4f}")
        print(f"Val Loss:                     {results['loss']:.4f}\n")
        print(f"Val C1 Accuracy               {results['accuracy_C1']:.4f}")
        print(f"Val C2 Accuracy               {results['accuracy_C2']:.4f}")
        print(f"Val C3 Accuracy               {results['accuracy_C3']:.4f}\n")
        print(f"Val C1 Balanced Accuracy:     {results['bal_accuracy_C1']:.4f}")
        print(f"Val C2 Balanced Accuracy:     {results['bal_accuracy_C2']:.4f}")
        print(f"Val C3 Balanced Accuracy:     {results['bal_accuracy_C3']:.4f}\n")
        print(f"Val C1 AP:                    {results['ap_C1']:.4f}")
        print(f"Val C2 AP:                    {results['ap_C2']:.4f}")
        print(f"Val C3 AP:                    {results['ap_C3']:.4f}")
        print(f"------------------------\n")

        results['saved'] = {    'C1': { 'probs':     val_output_dict['C1']['probs'].tolist(),
                                        'preds':     val_output_dict['C1']['preds'].tolist()},
                                'C2': { 'probs':     val_output_dict['C2']['probs'].tolist(),
                                        'preds':     val_output_dict['C2']['preds'].tolist()},
                                'C3': { 'probs':     val_output_dict['C3']['probs'].tolist(),
                                        'preds':     val_output_dict['C3']['preds'].tolist()},
                                'labels':            val_output_dict['labels'].tolist(),
                                'vid_ids':           val_output_dict['vid_ids'].tolist(),
                                'frame_ids':         val_output_dict['frame_ids'].tolist()}
        if EVIDENTIAL:
               results['saved']['C1']['uncerts'] = val_output_dict['C1']['uncerts'].tolist()
               results['saved']['C2']['uncerts'] = val_output_dict['C2']['uncerts'].tolist()       
               results['saved']['C3']['uncerts'] = val_output_dict['C3']['uncerts'].tolist()        
        results_dict[f"Epoch {epoch+1} Val"] = results

        # Save results
        with open(Path('./results') / f'{EXPERIMENT_NAME}_results.json', 'w') as file:
                json.dump(results_dict, file, indent=4)

        # Cosine LR deacay
        if epoch >= CONFIG['TRAIN']['WARMUP_EPOCHS']:
                cosine_scheduler.step()

        # Log the current lr of each param group for visibility
        logged = {      g['name']: g['lr']
                        for g in optimizer.param_groups
                        if not g['name'].endswith('_nd') and len(g['params']) > 0}
        lr_str = '  |  '.join(f"{name}: {lr:.2e}" for name, lr in logged.items())
        print(f"LR — {lr_str}\n")

        # Save weights of the best epoch
        if results['avg_bacc'] > best_bacc_across_epochs:
                best_bacc_across_epochs = results['avg_bacc']
                best_epoch = epoch+1
                epochs_without_improvement = 0
                print(f"New best result (Epoch {best_epoch}), saving weights...")
                checkpoint_path = Path(CONFIG['CHECKPOINT_DIR']) / f'{EXPERIMENT_NAME}.pt'
                torch.save(model.state_dict(), checkpoint_path)
        else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement}/{CONFIG['TRAIN']['EARLY_PATIENCE']} epochs\n")
                if epochs_without_improvement >= CONFIG['TRAIN']['EARLY_PATIENCE']:
                        print(f"Early stopping triggered. Best epoch was {best_epoch} with BAcc {best_bacc_across_epochs:.4f}")
                        break
        
print(f"Testing @ epoch {best_epoch}")
test_loss_sum = 0.0
len_test_loader = len(test_dataloader)
test_output_dict = dummy_output_dict(uncerts = EVIDENTIAL)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.to(device)

model.eval()
torch.cuda.synchronize()

with torch.inference_mode():
    for idx, (images, labels, vid_id, frame_id) in enumerate(test_dataloader):
        print(f'\r{idx+1}/{len_test_loader}', end='', flush=True)

        images, labels = images.to(device), labels.to(device)
        output = model(images)
        test_loss_per_batch = loss_fn(output, labels)

        test_output_dict = update_model_output_dict(output, test_output_dict, evidential = EVIDENTIAL)
        test_output_dict['labels'].append(labels.detach().cpu())
        test_output_dict['vid_ids'].append(vid_id)
        test_output_dict['frame_ids'].append(frame_id)
        test_loss_sum += test_loss_per_batch.item()

results, test_output_dict = calculate_metrics(test_output_dict, evidential = EVIDENTIAL)
avg_test_loss = test_loss_sum / len_test_loader
results['loss'] = round(avg_test_loss, 4)

print(f"\n--- Testing Metrics ---")
print(f"Test Avg Accuracy              {results['avg_accuracy']:.4f}")
print(f"Test Avg BAcc                  {results['avg_bacc']:.4f}")
print(f"Test mAP                       {results['mAP']:.4f}")
print(f"Test Loss:                     {results['loss']:.4f}\n")
print(f"Test C1 Accuracy               {results['accuracy_C1']:.4f}")
print(f"Test C2 Accuracy               {results['accuracy_C2']:.4f}")
print(f"Test C3 Accuracy               {results['accuracy_C3']:.4f}\n")
print(f"Test C1 Balanced Accuracy:     {results['bal_accuracy_C1']:.4f}")
print(f"Test C2 Balanced Accuracy:     {results['bal_accuracy_C2']:.4f}")
print(f"Test C3 Balanced Accuracy:     {results['bal_accuracy_C3']:.4f}\n")
print(f"Test C1 AP:                    {results['ap_C1']:.4f}")
print(f"Test C2 AP:                    {results['ap_C2']:.4f}")
print(f"Test C3 AP:                    {results['ap_C3']:.4f}")
print(f"------------------------\n")
results['saved'] = {'C1': { 'probs':     test_output_dict['C1']['probs'].tolist(),
                            'preds':     test_output_dict['C1']['preds'].tolist()},
                    'C2': { 'probs':     test_output_dict['C2']['probs'].tolist(),
                            'preds':     test_output_dict['C2']['preds'].tolist()},
                    'C3': { 'probs':     test_output_dict['C3']['probs'].tolist(),
                            'preds':     test_output_dict['C3']['preds'].tolist()},
                    'labels':            test_output_dict['labels'].tolist(),
                    'vid_ids':           test_output_dict['vid_ids'].tolist(),
                    'frame_ids':         test_output_dict['frame_ids'].tolist()}
if EVIDENTIAL:
       results['saved']['C1']['uncerts'] = test_output_dict['C1']['uncerts'].tolist()
       results['saved']['C2']['uncerts'] = test_output_dict['C2']['uncerts'].tolist()       
       results['saved']['C3']['uncerts'] = test_output_dict['C3']['uncerts'].tolist()    
results_dict[f"Epoch {best_epoch} Test"] = results

with open(Path('./results') / f'{EXPERIMENT_NAME}_results.json', 'w') as file:
    json.dump(results_dict, file, indent=4)