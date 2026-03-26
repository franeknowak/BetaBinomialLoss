import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score

##### GENERAL FUNCTIONS #####
def find_best_threshold_bacc(probs: torch.Tensor, labels: torch.Tensor, n_thresh=4001):
    """
    probs: 1D tensor of probabilities (N,)
    labels: 1D tensor of {0,1} (N,)
    """
    thresholds = torch.linspace(0, 1, n_thresh)
    best_thr = 0.5
    best_bacc = -1.0

    for thr in thresholds:
        preds = (probs >= thr).long()
        bacc = balanced_accuracy_score(labels.numpy(), preds.numpy())
        if bacc > best_bacc:
            best_bacc = bacc
            best_thr = thr.item()

    return best_thr, best_bacc

##### PLOTTING FUNCTIONS #####
def plot_general_training_info(results, save_path):
    train_acc = []
    train_avg_bal_acc = []
    train_avg_map = []
    train_loss = []
    val_acc = []
    val_avg_bal_acc = []
    val_avg_map = []
    val_loss = []
    for key in results.keys():
        if 'Train' in key:
            train_acc.append(results[key]['avg_accuracy'])
            train_avg_bal_acc.append(results[key]['avg_bacc'])
            train_avg_map.append(results[key]['mAP'])
            train_loss.append(results[key]['loss'])
        if 'Val' in key:
            val_acc.append(results[key]['avg_accuracy'])
            val_avg_bal_acc.append(results[key]['avg_bacc'])
            val_avg_map.append(results[key]['mAP'])
            val_loss.append(results[key]['loss'])
    
    x = range(1, 11)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # ----------------------------------------------------------
    # 1. Loss
    axs[0, 0].plot(x, train_loss, label='Train')
    axs[0, 0].plot(x, val_loss, label='Val')
    axs[0, 0].set_title('Loss: Train vs Val')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # ----------------------------------------------------------
    # 2. mAP
    axs[0, 1].plot(x, train_avg_map, label='Train')
    axs[0, 1].plot(x, val_avg_map, label='Val')
    axs[0, 1].set_title('mAP: Train vs Val')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('mAP')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # ----------------------------------------------------------
    # 3. Accuracy
    axs[1, 0].plot(x, train_acc, label='Train')
    axs[1, 0].plot(x, val_acc, label='Val')
    axs[1, 0].set_title('Accuracy: Train vs Val')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # ----------------------------------------------------------
    # 4. Balanced Accuracy
    axs[1, 1].plot(x, train_avg_bal_acc, label='Train')
    axs[1, 1].plot(x, val_avg_bal_acc, label='Val')
    axs[1, 1].set_title('bACC: Train vs Val')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('bACC')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

def plot_ROC_curves(val_results,
                    test_results,
                    save_path = None):
    """
    Plotting ROC curves for model, without custom thresholding, without uncertainty thresholding.
    """
    labels = ["C1", "C2", "C3"]
    AUC_results = {}
    # Create the two panels once: val (left) and test (right)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Add diagonals once
    for ax, title in zip(axs, ["Val ROC Curve", "Test ROC Curve"]):
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(True)
        ax.set_aspect("equal", adjustable="box")

    # Loop over criteria and overlay ROC curves on each panel
    for label_index, label in enumerate(labels):
        # ------------------------- VAL -------------------------
        p_val = torch.tensor(val_results["saved"][label]["probs"], dtype=torch.float32)
        y_val = torch.tensor(val_results["saved"]["labels"], dtype=torch.float32)[:, label_index]

        fpr, tpr, _ = roc_curve(y_val.numpy(),
                                p_val.numpy())
        roc_auc = auc(fpr, tpr)
        axs[0].plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
        AUC_results[f"{label}_val"] = round(roc_auc,3)

        # ------------------------- TEST ------------------------
        p_test = torch.tensor(test_results["saved"][label]["probs"], dtype=torch.float32)
        y_test = torch.tensor(test_results["saved"]["labels"], dtype=torch.float32)[:, label_index]

        fpr, tpr, _ = roc_curve(y_test.numpy(),
                                p_test.numpy())
        roc_auc = auc(fpr, tpr)
        axs[1].plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
        AUC_results[f"{label}_test"] = round(roc_auc,3)

    # Legends once per panel
    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return AUC_results

def confusion_matrix_plot(y_val,
                          y_test,
                          p_val,
                          p_test,
                          best_bacc_class_thr,
                          save_path):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    # ----------------------------------------------------------
    # 1. VAL CM DEFAULT
    cm = confusion_matrix(y_val, (p_val>0.5).float())
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=axs[0, 0], colorbar=False)
    axs[0, 0].set_title("Val Default")
    # ----------------------------------------------------------
    # 2. VAL CM BACC THRESH
    cm = confusion_matrix(y_val, (p_val>best_bacc_class_thr).float())
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=axs[0, 1], colorbar=False)
    axs[0, 1].set_title("Val BACC Thresholded")
    # ----------------------------------------------------------
    # 3. TEST CM DEFAULT
    cm = confusion_matrix(y_test, (p_test>0.5)) 
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=axs[1, 0], colorbar=False)
    axs[1, 0].set_title("Test Default")
    # ----------------------------------------------------------
    # 4. TEST CM BACC THRESH
    cm = confusion_matrix(y_test, (p_test>best_bacc_class_thr))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=axs[1, 1], colorbar=False)
    axs[1, 1].set_title("Test BACC Thresholded")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")