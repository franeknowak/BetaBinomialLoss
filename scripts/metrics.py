import numpy as np
from sklearn.metrics import average_precision_score
import torch

def p_k_ge_2_from_a(a0: torch.Tensor, a1: torch.Tensor, eps: float = 1e-12):
    S = (a0 + a1).clamp_min(eps)

    E2 = a1 * (a1 + 1.0) / (S * (S + 1.0))
    E3 = a1 * (a1 + 1.0) * (a1 + 2.0) / (S * (S + 1.0) * (S + 2.0))

    return 3.0 * E2 - 2.0 * E3

@torch.no_grad()
def update_model_output_dict(output, model_output_dict, class_keys=("C1", "C2", "C3"), evidential = False):
    """
    output: indexable like [C1_alpha, C2_alpha, C3_alpha], each alpha shape (B, K)
    train_output_dict: dict to be updated in-place (and returned)
    """
    
    # Account for the difference in number of heads and output values in said heads
    if isinstance(output, torch.Tensor):
        output = [output[:, i] for i in range(output.shape[1])]

    for i, key in enumerate(class_keys):
        if evidential:
            alpha = output[i]  # shape (B, 2) = [a0, a1]
            a0, a1 = alpha[:, 0], alpha[:, 1]
            prob = p_k_ge_2_from_a(a0, a1)
            uncert = 2.0 / (a0 + a1)
        else:
            prob = torch.sigmoid(output[i].view(-1))
        
        pred = torch.round(prob)      

        bucket = model_output_dict[key]
        bucket["probs"].append(prob.detach().cpu())
        bucket["preds"].append(pred.detach().cpu())
        if evidential:
            bucket["uncerts"].append(uncert.detach().cpu())

    return model_output_dict

def get_recall(true, predicted):
    """Calculates recall (True Positive Rate)."""
    # Avoid division by zero if there are no positive samples
    true_positives = np.sum((true == 1) & (predicted == 1))
    actual_positives = np.sum(true == 1)
    return true_positives / actual_positives if actual_positives > 0 else 0

def get_specificity(true, predicted):
    """Calculates specificity (True Negative Rate)."""
    # Avoid division by zero if there are no negative samples
    true_negatives = np.sum((true == 0) & (predicted == 0))
    actual_negatives = np.sum(true == 0)
    return true_negatives / actual_negatives if actual_negatives > 0 else 0


# Main metric functions (rewritten for binary classification)
def calculate_binary_metrics(y_true, y_pred, y_pred_probs):
    """
    Calculates balanced accuracy and average precision for a binary task.

    Args:
        y_true (list or np.array): List of ground truth label tensors/arrays.
        y_pred (list or np.array): List of predicted label tensors/arrays (0s and 1s).
        y_pred_probs (list or np.array): List of prediction probability tensors/arrays.

    Returns:
        tuple: A tuple containing (balanced_accuracy, average_precision).
    """
    # Consolidate list of tensors/arrays into a single flat numpy array
    true_labels = np.concatenate([np.array(x).flatten() for x in y_true])
    predicted_labels = np.concatenate([np.array(x).flatten() for x in y_pred])
    predicted_probs = np.concatenate([np.array(x).flatten() for x in y_pred_probs])

    # 1. Calculate Balanced Accuracy
    recall = get_recall(true_labels, predicted_labels)
    specificity = get_specificity(true_labels, predicted_labels)
    balanced_accuracy = (recall + specificity) / 2
    accuracy = np.sum(true_labels == predicted_labels) / len(true_labels)


    # 2. Calculate Average Precision (AP)
    average_precision = average_precision_score(true_labels, predicted_probs)

    return accuracy, balanced_accuracy, average_precision

def calculate_metrics(model_output_dict, class_keys=("C1", "C2", "C3"), evidential = False):
    # Concat the lists in output_dict to form a single tensor
    for c in class_keys:
        desired = ['probs', 'preds', 'uncerts'] if evidential else ['probs', 'preds']
        for key in desired:
            model_output_dict[c][key] = torch.cat(model_output_dict[c][key], dim=0)
    model_output_dict['labels'] = torch.round(torch.cat(model_output_dict['labels'], dim=0)).squeeze(1)
    model_output_dict['vid_ids'] = torch.cat(model_output_dict['vid_ids'], dim=0)
    model_output_dict['frame_ids'] = torch.cat(model_output_dict['frame_ids'], dim=0)
    
    n = len(class_keys)
    results = {'avg_accuracy':      0.0,
               'avg_bacc':          0.0,
               'mAP':               0.0,
               'accuracy_C1':       0.0,
               'accuracy_C2':       0.0,
               'accuracy_C3':       0.0,
               'bal_accuracy_C1':   0.0,
               'bal_accuracy_C2':   0.0,
               'bal_accuracy_C3':   0.0,
               'ap_C1':             0.0,
               'ap_C2':             0.0,
               'ap_C3':             0.0}
    if evidential:
        results.update({'avg_uncert_pos': 0.0, 'avg_uncert_neg': 0.0})
        
    for i, key in enumerate(class_keys):
        accuracy, balanced_accuracy, average_precision = calculate_binary_metrics(model_output_dict['labels'][:,i], model_output_dict[key]['preds'], model_output_dict[key]['probs'])
        results['accuracy_'+key] = accuracy
        results['bal_accuracy_'+key] = balanced_accuracy
        results['ap_'+key] = average_precision
        results['avg_accuracy']+=accuracy
        results['avg_bacc']+=balanced_accuracy
        results['mAP']+=average_precision

        if evidential:
            uncert_pos = float(model_output_dict[key]['uncerts'][model_output_dict['labels'][:,i] == 1.0].mean())
            uncert_neg = float(model_output_dict[key]['uncerts'][model_output_dict['labels'][:,i] == 0.0].mean())
            results['avg_uncert_pos']+=uncert_pos
            results['avg_uncert_neg']+=uncert_neg

    # Get averages across classes
    results['avg_accuracy']/=n
    results['avg_bacc']/=n
    results['mAP']/=n
    if evidential:
        results['avg_uncert_neg']/=n
        results['avg_uncert_pos']/=n

    # Round to four significatnt numbers
    for key in results.keys():
        results[key] = round(results[key],4)

    return results, model_output_dict