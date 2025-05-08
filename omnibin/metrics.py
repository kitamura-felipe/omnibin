import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, matthews_corrcoef, roc_curve,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve
from matplotlib.backends.backend_pdf import PdfPages

def generate_binary_classification_report(y_true, y_scores, output_path="omnibin_report.pdf", n_bootstrap=1000, random_seed=42):
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set default DPI for all figures
    plt.rcParams['figure.dpi'] = 300
    
    thresholds = np.linspace(0, 1, 100)
    metrics_by_threshold = []

    for t in tqdm(thresholds, desc="Calculating metrics across thresholds"):
        y_pred = (y_scores >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        sens = recall_score(y_true, y_pred)
        spec = recall_score(y_true, y_pred, pos_label=0)
        ppv = precision_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        metrics_by_threshold.append([t, acc, sens, spec, ppv, mcc, f1])

    metrics_df = pd.DataFrame(metrics_by_threshold, columns=[
        "Threshold", "Accuracy", "Sensitivity", "Specificity",
        "PPV", "MCC", "F1 Score"
    ])

    def bootstrap_metric(metric_func, y_true, y_scores, n_boot=1000):
        stats = []
        for _ in tqdm(range(n_boot), desc="Bootstrap iterations", leave=False):
            indices = np.random.choice(range(len(y_true)), len(y_true), replace=True)
            try:
                stats.append(metric_func(y_true[indices], y_scores[indices]))
            except:
                continue
        return np.percentile(stats, [2.5, 97.5])

    def bootstrap_curves(y_true, y_scores, n_boot=1000):
        tprs = []
        fprs = []
        precisions = []
        recalls = []
        
        # Get the base curves to determine common points
        base_fpr, base_tpr, _ = roc_curve(y_true, y_scores)
        base_precision, base_recall, _ = precision_recall_curve(y_true, y_scores)
        
        # Create common x-axis points
        common_fpr = np.linspace(0, 1, 100)
        common_recall = np.linspace(0, 1, 100)
        
        for _ in tqdm(range(n_boot), desc="Bootstrap iterations for curves", leave=False):
            indices = np.random.choice(range(len(y_true)), len(y_true), replace=True)
            try:
                # ROC curve
                fpr, tpr, _ = roc_curve(y_true[indices], y_scores[indices])
                tpr_interp = np.interp(common_fpr, fpr, tpr)
                tprs.append(tpr_interp)
                
                # PR curve - handle precision interpolation carefully
                precision, recall, _ = precision_recall_curve(y_true[indices], y_scores[indices])
                # Sort by recall to ensure proper interpolation
                sort_idx = np.argsort(recall)
                recall = recall[sort_idx]
                precision = precision[sort_idx]
                # Interpolate precision values
                precision_interp = np.interp(common_recall, recall, precision)
                precisions.append(precision_interp)
            except:
                continue
        
        # Calculate confidence intervals
        tpr_ci = np.percentile(tprs, [2.5, 97.5], axis=0)
        precision_ci = np.percentile(precisions, [2.5, 97.5], axis=0)
        
        return tpr_ci, precision_ci, common_fpr, common_recall

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    best_thresh = roc_thresholds[np.argmax(j_scores)]
    y_pred_opt = (y_scores >= best_thresh).astype(int)

    metrics_summary = {
        "Accuracy": accuracy_score(y_true, y_pred_opt),
        "Sensitivity": recall_score(y_true, y_pred_opt),
        "Specificity": recall_score(y_true, y_pred_opt, pos_label=0),
        "PPV": precision_score(y_true, y_pred_opt, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred_opt),
        "F1 Score": f1_score(y_true, y_pred_opt),
        "AUC-ROC": roc_auc_score(y_true, y_scores),
        "AUC-PR": average_precision_score(y_true, y_scores)
    }

    conf_intervals = {}
    for name, func in {
        "Accuracy": lambda yt, ys: accuracy_score(yt, ys >= best_thresh),
        "Sensitivity": lambda yt, ys: recall_score(yt, ys >= best_thresh),
        "Specificity": lambda yt, ys: recall_score(yt, ys >= best_thresh, pos_label=0),
        "PPV": lambda yt, ys: precision_score(yt, ys >= best_thresh, zero_division=0),
        "MCC": lambda yt, ys: matthews_corrcoef(yt, ys >= best_thresh),
        "F1 Score": lambda yt, ys: f1_score(yt, ys >= best_thresh),
        "AUC-ROC": lambda yt, ys: roc_auc_score(yt, ys),
        "AUC-PR": lambda yt, ys: average_precision_score(yt, ys)
    }.items():
        ci = bootstrap_metric(func, y_true, y_scores, n_boot=n_bootstrap)
        conf_intervals[name] = ci

    # Create output directory for individual plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    with PdfPages(output_path) as pdf:
        # ROC and PR Curves with proper confidence intervals
        plt.figure(figsize=(12, 5), dpi=300)
        
        # Calculate confidence intervals for curves
        tpr_ci, precision_ci, common_fpr, common_recall = bootstrap_curves(y_true, y_scores, n_boot=n_bootstrap)
        
        plt.subplot(1, 2, 1)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.plot(fpr, tpr, label="ROC curve")
        plt.fill_between(common_fpr, tpr_ci[0], tpr_ci[1], alpha=0.3)
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        plt.subplot(1, 2, 2)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.plot(recall, precision, label="PR curve")
        plt.fill_between(common_recall, precision_ci[0], precision_ci[1], alpha=0.3)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "roc_pr.png"), dpi=300, bbox_inches='tight')
        pdf.savefig(dpi=300)
        plt.close()

        # Metrics vs Threshold
        plt.figure(figsize=(10, 6), dpi=300)
        for col in metrics_df.columns[1:]:
            plt.plot(metrics_df["Threshold"], metrics_df[col], label=col)
        plt.xlabel("Threshold")
        plt.ylabel("Metric Value")
        plt.title("Metrics Across Thresholds")
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "metrics_threshold.png"), dpi=300, bbox_inches='tight')
        pdf.savefig(dpi=300)
        plt.close()

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred_opt)
        plt.figure(figsize=(5, 4), dpi=300)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix (Optimal Threshold)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
        pdf.savefig(dpi=300)
        plt.close()

        # Calibration Plot
        plt.figure(figsize=(6, 6), dpi=300)
        prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=10, strategy='uniform')
        plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Calibration Plot')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "calibration.png"), dpi=300, bbox_inches='tight')
        pdf.savefig(dpi=300)
        plt.close()

        # Metrics Summary Table
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        ax.axis("off")
        table_data = [
            [k, f"{v:.3f}", f"[{conf_intervals[k][0]:.3f}, {conf_intervals[k][1]:.3f}]"]
            for k, v in metrics_summary.items()
        ]
        table = ax.table(cellText=table_data, colLabels=["Metric", "Value", "95% CI"], loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        ax.set_title("Performance Metrics at Optimal Threshold", fontweight="bold")
        plt.savefig(os.path.join(plots_dir, "metrics_summary.png"), dpi=300, bbox_inches='tight')
        pdf.savefig(dpi=300)
        plt.close()

        # Prediction Distribution Histogram
        plt.figure(figsize=(10, 6), dpi=300)
        plt.hist(y_scores[y_true == 1], bins=50, alpha=0.5, label='Positive Class', color='blue')
        plt.hist(y_scores[y_true == 0], bins=50, alpha=0.5, label='Negative Class', color='red')
        plt.axvline(x=best_thresh, color='black', linestyle='--', label=f'Optimal Threshold ({best_thresh:.3f})')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title('Distribution of Predictions')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "prediction_distribution.png"), dpi=300, bbox_inches='tight')
        pdf.savefig(dpi=300)
        plt.close()

    return output_path