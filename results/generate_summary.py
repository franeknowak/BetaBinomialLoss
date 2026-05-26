"""
Generate a comprehensive experiment summary document with metrics tables
and training curve plots for all runs across all experiment folders.
"""

import json
import re
import os
import base64
from io import BytesIO
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = Path(__file__).parent
PLOTS_DIR = RESULTS_DIR / "summary_plots"
PLOTS_DIR.mkdir(exist_ok=True)

TRAIN_COL = "#4C72B0"
VAL_COL   = "#DD8452"
BEST_COL  = "#2ca02c"

GENERAL_METRICS = [
    ("Accuracy",          "avg_accuracy"),
    ("Balanced Accuracy", "avg_bacc"),
    ("mAP",               "mAP"),
    ("Loss",              "loss"),
]

PCT_KEYS = {"avg_accuracy", "avg_bacc", "mAP",
            "accuracy_C1", "accuracy_C2", "accuracy_C3",
            "bal_accuracy_C1", "bal_accuracy_C2", "bal_accuracy_C3",
            "ap_C1", "ap_C2", "ap_C3"}


def fmt(val, key):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    if key in PCT_KEYS:
        return f"{val*100:.2f}%"
    return f"{val:.4f}"


def parse_results(path):
    with open(path) as f:
        data = json.load(f)

    train_rows, val_rows = [], []
    test_row, test_epoch = None, None

    for key, metrics in data.items():
        m = re.match(r"Epoch (\d+) (Train|Val|Test)", key, re.IGNORECASE)
        if not m:
            if re.match(r"^test$", key, re.IGNORECASE):
                test_row = metrics
            continue
        epoch, split = int(m.group(1)), m.group(2).lower()
        row = {"epoch": epoch, **{k: v for k, v in metrics.items() if k != "saved"}}
        if split == "train":
            train_rows.append(row)
        elif split == "val":
            val_rows.append(row)
        elif split == "test":
            test_row = {k: v for k, v in metrics.items() if k != "saved"}
            test_epoch = epoch

    train_df = pd.DataFrame(train_rows).sort_values("epoch").reset_index(drop=True)
    val_df   = pd.DataFrame(val_rows).sort_values("epoch").reset_index(drop=True)
    return train_df, val_df, test_row, test_epoch


def make_curves(train_df, val_df, best_epoch, title, plot_path):
    epochs = train_df["epoch"].values

    fig = plt.figure(figsize=(12, 4.5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(epochs, train_df["loss"], color=TRAIN_COL, lw=2, label="Train")
    ax1.plot(val_df["epoch"].values, val_df["loss"], color=VAL_COL, lw=2, label="Val")
    if best_epoch is not None:
        ax1.axvline(best_epoch, color=BEST_COL, ls="--", lw=1.4, label=f"Best val (ep {best_epoch})")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss vs Epoch"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(epochs, train_df["avg_bacc"], color=TRAIN_COL, lw=2, label="Train")
    ax2.plot(val_df["epoch"].values, val_df["avg_bacc"], color=VAL_COL, lw=2, label="Val")
    if best_epoch is not None:
        ax2.axvline(best_epoch, color=BEST_COL, ls="--", lw=1.4, label=f"Best val (ep {best_epoch})")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Avg Balanced Accuracy")
    ax2.set_title("Avg Balanced Accuracy vs Epoch"); ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def metrics_table_md(train_df, val_df, test_row, best_epoch, best_val_row):
    """Return a markdown table with best-train / best-val / test metrics."""
    best_train_row = train_df.loc[train_df["epoch"] == best_epoch].iloc[0] if best_epoch in train_df["epoch"].values else None

    rows = []
    for label, key in GENERAL_METRICS:
        tv = best_train_row.get(key) if best_train_row is not None else None
        vv = best_val_row.get(key)
        te = test_row.get(key) if test_row else None
        rows.append(f"| {label} | {fmt(tv, key)} | {fmt(vv, key)} | {fmt(te, key)} |")

    header = (
        f"| Metric | Best Train (Ep {best_epoch}) | Best Val (Ep {best_epoch}) | Test |\n"
        f"|--------|----------------------------|-----------------------------|------|\n"
    )
    return header + "\n".join(rows)


def per_class_table_md(test_row, best_val_row):
    classes = ["C1", "C2", "C3"]
    header = "| Class | Accuracy | Balanced Accuracy | AP |\n|-------|----------|-------------------|----|\n"
    rows_val, rows_test = [], []
    for c in classes:
        acc_v  = best_val_row.get(f"accuracy_{c}")
        bacc_v = best_val_row.get(f"bal_accuracy_{c}")
        ap_v   = best_val_row.get(f"ap_{c}")
        rows_val.append(f"| {c} | {fmt(acc_v,'avg_accuracy')} | {fmt(bacc_v,'avg_bacc')} | {fmt(ap_v,'mAP')} |")

        if test_row:
            acc_t  = test_row.get(f"accuracy_{c}")
            bacc_t = test_row.get(f"bal_accuracy_{c}")
            ap_t   = test_row.get(f"ap_{c}")
            rows_test.append(f"| {c} | {fmt(acc_t,'avg_accuracy')} | {fmt(bacc_t,'avg_bacc')} | {fmt(ap_t,'mAP')} |")

    val_table  = "**Best Val per-class:**\n\n" + header + "\n".join(rows_val)
    test_table = ("**Test per-class:**\n\n" + header + "\n".join(rows_test)) if rows_test else ""
    return val_table + "\n\n" + test_table


def process_file(json_path):
    """Return a markdown section string for a single run."""
    experiment = json_path.parent.name
    run_name   = json_path.stem  # e.g. notemp_bce_0_results

    train_df, val_df, test_row, test_epoch = parse_results(json_path)

    if val_df.empty:
        return f"### {run_name}\n\n_No validation data found._\n\n---\n"

    best_idx     = val_df["avg_bacc"].idxmax()
    best_val_row = val_df.loc[best_idx].to_dict()
    best_epoch   = int(best_val_row["epoch"])

    # Save plot
    safe_name  = f"{experiment}__{run_name}"
    plot_path  = PLOTS_DIR / f"{safe_name}.png"
    plot_title = f"{experiment} / {run_name}"
    make_curves(train_df, val_df, best_epoch, plot_title, plot_path)
    rel_plot   = plot_path.relative_to(RESULTS_DIR)

    # Build markdown
    lines = []
    lines.append(f"### Run: `{run_name}`")
    lines.append(f"\n**Best validation epoch:** {best_epoch}  "
                 f"(avg_bacc = {best_val_row['avg_bacc']*100:.2f}%)")

    if test_epoch is not None:
        lines.append(f"  \n**Test evaluated at epoch:** {test_epoch}")

    lines.append("\n#### Summary Metrics\n")
    lines.append(metrics_table_md(train_df, val_df, test_row, best_epoch, best_val_row))

    lines.append("\n#### Per-Class Metrics\n")
    lines.append(per_class_table_md(test_row, best_val_row))

    lines.append("\n#### Training Curves\n")
    lines.append(f"![Training curves for {run_name}]({rel_plot})\n")

    lines.append("\n---\n")
    return "\n".join(lines)


def main():
    # Collect all JSON result files, grouped by experiment folder
    experiments = {}
    for json_path in sorted(RESULTS_DIR.rglob("*_results.json")):
        exp = json_path.parent.name
        experiments.setdefault(exp, []).append(json_path)

    doc_lines = [
        "# Experiment Results Summary",
        "",
        "This document summarises all training runs. "
        "For each run it reports: best validation epoch, "
        "accuracy / balanced accuracy / mAP / loss at that epoch for "
        "train, val and test sets, per-class metrics, and training curves.",
        "",
        "---",
        "",
    ]

    for exp_name, files in experiments.items():
        doc_lines.append(f"## Experiment: `{exp_name}`")
        doc_lines.append("")
        for fpath in sorted(files):
            doc_lines.append(process_file(fpath))

    output_path = RESULTS_DIR / "experiment_summary.md"
    output_path.write_text("\n".join(doc_lines))
    print(f"Summary written to: {output_path}")
    print(f"Plots saved to:     {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
