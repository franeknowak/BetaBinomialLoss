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
from scripts.encoder_swinv2 import build_swinv2_encoder
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
transforms =  v2.Compose([  v2.CenterCrop(CONFIG['DATA']['DATASETS'][DATASET_NAME]['CENTER_CROP']),
                            v2.Resize((CONFIG['DATA']['DATASETS'][DATASET_NAME]['RESIZE'], CONFIG['DATA']['DATASETS'][DATASET_NAME]['RESIZE'])),
                            v2.ToImage(),                
                            v2.ToDtype(torch.float32, scale=True),  
                            v2.Normalize(mean=CONFIG['DATA']['DATASETS'][DATASET_NAME]['MEAN'], std=CONFIG['DATA']['DATASETS'][DATASET_NAME]['STD'])])

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
model = build_swinv2_encoder(CONFIG['ENCODER']['NAME'],
                             CONFIG['ENCODER']['FROZEN_STAGES'])

# Separate parameter groups for adjusted learning rate
backbone_params = []
head_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if name.startswith("head"):
        head_params.append(param)
    else:
        backbone_params.append(param)

optimizer = optim.AdamW(    [{"params": backbone_params, "lr": CONFIG['TRAIN']['OPTIMIZER']['ENCODER_LR']},
                             {"params": head_params, "lr": CONFIG['TRAIN']['OPTIMIZER']['CLASSIFIER_LR']}],
                             betas = CONFIG['TRAIN']['OPTIMIZER']['BETAS'],
                             eps = CONFIG['TRAIN']['OPTIMIZER']['EPS'],
                             weight_decay=CONFIG['TRAIN']['OPTIMIZER']['WEIGHT_DECAY'])
model.to(device)

class_weights = torch.tensor(CONFIG['DATA']['DATASETS'][DATASET_NAME]['CLASS_WEIGHTS']).to(device) # weights, specific to BCE, taken from official endoscapes implementation repository
bce_loss = nn.BCEWithLogitsLoss(weight=class_weights).to(device)

############################################################################################
############################################################################################
# MODEL TRAINING AND EVALUATION
results_dict = {}

best_bacc_across_epochs = -1.0
best_epoch = 0
epochs_without_improvement = 0

EPOCHS = CONFIG['TRAIN']['EPOCHS']
ACCUMULATION_STEPS = CONFIG['TRAIN']['GRADIENT_ACC_BATCH_SIZE'] // CONFIG['TRAIN']['BATCH_SIZE']

for epoch in range(EPOCHS):
        print(f"Epoch: {epoch+1:02}/{EPOCHS:02}")
        print("Training")
        train_loss_sum = 0.0
        len_train_loader = len(train_dataloader)
        train_output_dict = {'C1':  {'probs':     [],
                                     'preds':     []},
                             'C2':  {'probs':     [],
                                     'preds':     []},
                             'C3':  {'probs':     [],
                                     'preds':     []},
                             'labels':            [],
                             'vid_ids':           [],
                             'frame_ids':         []}
        model.train()
        optimizer.zero_grad()
        for idx, (images, labels, vid_id, frame_id) in enumerate(train_dataloader):
                print(f'\r{idx+1}/{len_train_loader}', end='', flush=True)

                images, labels = images.to(device), labels.to(device)
                torch.cuda.synchronize()

                output = model(images)

                train_loss_per_acc_batch = bce_loss(output, labels) / ACCUMULATION_STEPS
                train_loss_per_acc_batch.backward()

                if (idx + 1) % ACCUMULATION_STEPS == 0 or (idx + 1) == len_train_loader:
                        optimizer.step()
                        optimizer.zero_grad()

                # Populate the output dict with probs and preds per batch per class
                train_output_dict = update_model_output_dict(output, train_output_dict)
                train_output_dict['labels'].append(labels.detach().cpu())
                train_output_dict['vid_ids'].append(vid_id)
                train_output_dict['frame_ids'].append(frame_id)
                train_loss_sum += train_loss_per_acc_batch.item() * ACCUMULATION_STEPS

        results, train_output_dict = calculate_metrics(train_output_dict)

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
        val_output_dict = {     'C1':  {'probs':     [],
                                        'preds':     []},
                                'C2':  {'probs':     [],
                                        'preds':     []},
                                'C3':  {'probs':     [],
                                        'preds':     []},
                                'labels':            [],
                                'vid_ids':           [],
                                'frame_ids':         []}
        model.eval()
        with torch.inference_mode():
                for idx, (images, labels, vid_id, frame_id) in enumerate(val_dataloader):
                        print(f'\r{idx+1}/{len_val_loader}', end='', flush=True)
                        images, labels = images.to(device), labels.to(device)
                        torch.cuda.synchronize()

                        output = model(images)

                        val_loss_per_batch = bce_loss(output, labels) 

                        val_output_dict = update_model_output_dict(output, val_output_dict)
                        val_output_dict['labels'].append(labels.detach().cpu())
                        val_output_dict['vid_ids'].append(vid_id)
                        val_output_dict['frame_ids'].append(frame_id)
                        val_loss_sum += val_loss_per_batch.item()

        results, val_output_dict = calculate_metrics(val_output_dict)
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
        results_dict[f"Epoch {epoch+1} Val"] = results

        # Save results
        with open(Path('./results') / f'{EXPERIMENT_NAME}_results.json', 'w') as file:
                json.dump(results_dict, file, indent=4)

        # Save weights of the best epoch
        if results['avg_bacc'] >= best_bacc_across_epochs:
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
test_output_dict = {    'C1':  {'probs':     [],
                                'preds':     []},
                        'C2':  {'probs':     [],
                                'preds':     []},
                        'C3':  {'probs':     [],
                                'preds':     []},
                        'labels':            [],
                        'vid_ids':           [],
                        'frame_ids':         []}
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()
with torch.inference_mode():
    for idx, (images, labels, vid_id, frame_id) in enumerate(test_dataloader):
        print(f'\r{idx+1}/{len_test_loader}', end='', flush=True)
        images, labels = images.to(device), labels.to(device)
        torch.cuda.synchronize()
        
        output = model(images)

        test_loss_per_batch = bce_loss(output, labels)

        test_output_dict = update_model_output_dict(output, test_output_dict)
        test_output_dict['labels'].append(labels.detach().cpu())
        test_output_dict['vid_ids'].append(vid_id)
        test_output_dict['frame_ids'].append(frame_id)
        test_loss_sum += test_loss_per_batch.item()

results, test_output_dict = calculate_metrics(test_output_dict)
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

results_dict[f"Epoch {best_epoch} Test"] = results
with open(Path('./results') / f'{EXPERIMENT_NAME}_results.json', 'w') as file:
    json.dump(results_dict, file, indent=4)
