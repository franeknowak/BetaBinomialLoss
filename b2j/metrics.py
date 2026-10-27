"""Phase-1 metrics (PLAN §1.9), per label, against the majority vote 1[k > n/2] at the 0.5 cut-off.

Per label: ACC, BACC, sensitivity, specificity, F1, AP, AUROC (threshold-free), MSE of the predicted fraction against
k/n, vote NLL (mean over images). A value that cannot be computed is NaN. "macro" = plain mean over labels (NaN if any
label is NaN; how to treat NaN is decided at analysis). CIFAR-10H and FER+ also: top-1 accuracy = the head with the
highest predicted fraction, correct if it is one of the classes with the most votes (ties: any of them counts).
"""
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from b2j.losses import predicted_fraction, vote_nll

SINGLE_LABEL = ("cifar10h", "ferplus")             # one true class per image: top-1 accuracy reported
PER_LABEL = ("acc", "bacc", "sensitivity", "specificity", "f1", "ap", "auroc", "mse", "vote_nll",
             "n_pred_pos", "n_ref_pos")


def _div(a, b):
    return a / b if b > 0 else float("nan")


def binary_metrics(p, y, frac, nll):
    """One label. p: predicted fraction [N]; y: majority-vote reference {0,1} [N]; frac: k/n [N]; nll: vote NLL [N]."""
    pred = p > 0.5
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    sens, spec = _div(tp, tp + fn), _div(tn, tn + fp)
    prec = _div(tp, tp + fp)
    f1 = float("nan") if np.isnan(prec) or np.isnan(sens) else _div(2 * tp, 2 * tp + fp + fn)   # NaN if precision undefined
    both = 0 < y.sum() < len(y)
    return {"acc": _div(tp + tn, len(y)), "bacc": (sens + spec) / 2, "sensitivity": sens, "specificity": spec,
            "f1": f1,
            "ap": float(average_precision_score(y, p)) if both else float("nan"),
            "auroc": float(roc_auc_score(y, p)) if both else float("nan"),
            "mse": float(np.mean((p - frac) ** 2)), "vote_nll": float(np.mean(nll)),
            "n_pred_pos": tp + fp, "n_ref_pos": tp + fn}


def top1_accuracy(p, k):
    """p, k: [N, K]. Correct if argmax_j p is among the classes with the most votes. Tied vote counts: any of the tied
    classes counts as correct. Tied predicted probabilities (practically impossible): the first class is predicted."""
    pred = p.argmax(1)
    return float((k[np.arange(len(k)), pred] == k.max(1)).mean())


def compute(loss_name, out, k, n, labels, dataset):
    """Metrics for one split. out: head outputs [N, K, w] (torch); k, n: [N, K] (torch). Returns a plain dict."""
    with torch.no_grad():
        p = predicted_fraction(loss_name, out).cpu().numpy()
        nll = vote_nll(loss_name, out, k, n).cpu().numpy()
    k, n = k.double().cpu().numpy(), n.double().cpu().numpy()
    y, frac = (k > n / 2).astype(int), k / n
    per = {lab: binary_metrics(p[:, j], y[:, j], frac[:, j], nll[:, j]) for j, lab in enumerate(labels)}
    res = {"per_label": per,
           "macro": {m: float(np.mean([per[lab][m] for lab in labels])) for m in PER_LABEL[:9]},
           "n_images": int(len(k))}
    if dataset in SINGLE_LABEL:
        res["top1"] = top1_accuracy(p, k)
    return res


def macro_bacc(loss_name, out, k, n):
    """Validation BACC used for checkpoint selection (macro over labels)."""
    with torch.no_grad():
        p = predicted_fraction(loss_name, out).cpu().numpy()
    k, n = k.double().cpu().numpy(), n.double().cpu().numpy()
    y, pred = k > n / 2, p > 0.5
    sens = (pred & y).sum(0) / y.sum(0)
    spec = (~pred & ~y).sum(0) / (~y).sum(0)
    return float(np.mean((sens + spec) / 2))
