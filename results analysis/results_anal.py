import json
import argparse
import numpy as np
import torch
import shutil
import pandas as pd
from pathlib import Path
from old.results_anal_scripts import (plot_general_training_info,
                                          plot_ROC_curves,
                                          find_best_threshold_bacc,
                                          confusion_matrix_plot)
from sklearn.metrics import balanced_accuracy_score, average_precision_score
import warnings
warnings.filterwarnings("ignore")

def main(results_file):
    general_metrics = { 'bacc_class_thresh':    {}}

    ###### OPEN RESULTS ######
    print(f"Opening file...")
    PWD = Path.cwd()
    results_dir = PWD / 'results'
    with open(results_dir / results_file, 'r') as f:
        results = json.load(f)

    ###### CREATE SUBFOLDER WHERE RESULTS ARE SAVED ######
    print(f"Creating folder with saved results...")
    experiment_name = results_file.split('.')[0]
    saved_results_path = results_dir / experiment_name
    if saved_results_path.exists():
        shutil.rmtree(saved_results_path)
    saved_results_path.mkdir(parents=True)

    ###### PLOT GENERAL TRAINING INFO ######
    print(f"Plotting general training info...")
    plot_general_training_info(results, save_path = saved_results_path / 'general_training_info_plot')

    ###### FIND THE BEST EPOCH ######
    print(f"Finding best training, val, test epoch...")
    train_avg_bal_acc = []
    val_avg_bal_acc = []
    for key in results.keys():
        if 'Train' in key:
            train_avg_bal_acc.append(results[key]['avg_bacc'])
        if 'Val' in key:
            val_avg_bal_acc.append(results[key]['avg_bacc'])
    
    ###### EXTRACT BEST RESULTS ######
    print(f"Extracting best results...")
    # Train
    best_train_index = np.argmax(train_avg_bal_acc)
    train_epochs = [x for x in list(results.keys()) if 'Train' in x]
    train_results = results[train_epochs[best_train_index]]
    # Val
    best_val_index = np.argmax(val_avg_bal_acc)
    val_epochs = [x for x in list(results.keys()) if 'Val' in x]
    val_results = results[val_epochs[best_val_index]]
    # Test
    test_results = results[list(results.keys())[-1]]

    data = {
    "Split\Metric": ["Epoch", "Accuracy", "Balanced Acc.", "mAP"],
    "Train": [int(best_train_index+1), 100*train_results['avg_accuracy'], 100*train_results['avg_bacc'], 100*train_results['mAP']],
    "Val":   [int(best_val_index+1),   100*val_results['avg_accuracy'],   100*val_results['avg_bacc'],   100*val_results['mAP']],
    "Test":  [int(best_val_index+1),   100*test_results['avg_accuracy'],  100*test_results['avg_bacc'],  100*test_results['mAP']]
    }
    print(f"Saving basic results...")
    basic_results = pd.DataFrame(data).set_index("Split\Metric").T
    basic_results.to_excel(saved_results_path / "basic_results.xlsx", index=True)

    ###### PLOT AND SAVE ROC CURVES ######
    print(f"Plotting and saving ROC curves...")
    AUC_results = plot_ROC_curves(  val_results,
                                    test_results,
                                    save_path = saved_results_path / 'ROC_curves')
    general_metrics['AUC'] = AUC_results

    ###### EXTRACT CLASS THRESHOLD RESULTS ######
    print(f"==Starting general evaluation on classification and uncertainty thresholds...==")
    val_bacc_class_thresh_list = []
    test_bacc_class_thresh_list = []

    for label_idx, label in enumerate(['C1', 'C2', 'C3']):
        ###### LOAD RESULTS ######
        print(f"Loading results for {label}...")
        p_val = torch.tensor(val_results['saved'][label]["probs"],   dtype=torch.float32)
        y_val = torch.tensor(val_results['saved']["labels"],  dtype=torch.float32)[:,label_idx]
        u_val = torch.tensor(val_results['saved'][label]["uncerts"], dtype=torch.float32)

        p_test = torch.tensor(test_results['saved'][label]["probs"],   dtype=torch.float32)
        y_test = torch.tensor(test_results['saved']["labels"],  dtype=torch.float32)[:,label_idx]
        u_test = torch.tensor(test_results['saved'][label]["uncerts"], dtype=torch.float32)

        ###### GET BEST BACC CLASSIFCICATION THRESHOLD ######
        print(f"Getting best classification threshold for {label}...")
        best_bacc_class_thr, best_val_bacc = find_best_threshold_bacc(p_val, y_val)
        general_metrics['bacc_class_thresh'][label] = round(best_bacc_class_thr,4)

        ###### GET BACC CLASS THRESH RESULTS ######
        best_val_bacc = balanced_accuracy_score(y_val, (p_val >= best_bacc_class_thr).long())
        best_val_map = average_precision_score(y_val, p_val)
        val_acc = ((p_val >= best_bacc_class_thr).long() == y_val).float().mean().item()
        val_bacc_class_thresh_list.append(round(val_acc*100,2)) # ACC
        val_bacc_class_thresh_list.append(round(best_val_bacc*100,2)) # BACC
        val_bacc_class_thresh_list.append(round(best_val_map*100,2)) # MAP

        best_test_bacc = balanced_accuracy_score(y_test, (p_test >= best_bacc_class_thr).long())
        best_test_map = average_precision_score(y_test, p_test)
        test_acc = ((p_test >= best_bacc_class_thr).long() == y_test).float().mean().item()
        test_bacc_class_thresh_list.append(round(test_acc*100,2)) # ACC
        test_bacc_class_thresh_list.append(round(best_test_bacc*100,2)) # BACC
        test_bacc_class_thresh_list.append(round(best_test_map*100,2)) # MAP

        ###### SAVE CONFUSION MATRICES ######
        print(f"Plotting confusion matrices for {label}...")
        confusion_matrix_plot(  y_val,
                                y_test,
                                p_val,
                                p_test,
                                best_bacc_class_thr,
                                save_path = saved_results_path / f"{label}_confusion_matrices")

    print(f"SAVING RESULTS FOR CLASS THRESH...")
    class_thresh_data = {  "Split\Metric": ["C1 ACC", "C1 BACC", "C1 MAP", "C2 ACC", "C2 BACC", "C2 MAP", "C3 ACC", "C3 BACC", "C3 MAP"],
                                "Val":   val_bacc_class_thresh_list,
                                "Test":  test_bacc_class_thresh_list}
    class_thresh_results = pd.DataFrame(class_thresh_data).set_index("Split\Metric").T
    class_thresh_results.to_excel(saved_results_path / "class_thresh_results.xlsx", index=True)
    
    print(f"SAVING GENERAL METRICS...")
    with open(saved_results_path / "general_metrics.json", "w") as f:
        json.dump(general_metrics, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on results JSON")
    parser.add_argument(    "--results_file",
                            type=str,
                            required=True,
                            help="Results JSON filename saved in /results folder")

    args = parser.parse_args()

    main(args.results_file)