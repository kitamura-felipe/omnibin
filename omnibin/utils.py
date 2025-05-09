import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, matthews_corrcoef, roc_curve,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve
from enum import Enum
import os

class ColorScheme(Enum):
    DEFAULT = {
        'positive_class': 'tab:blue',
        'negative_class': 'tab:orange',
        'roc_curve': 'tab:blue',
        'pr_curve': 'tab:blue',
        'threshold_line': 'black',
        'calibration_curve': 'tab:blue',
        'calibration_reference': 'gray',
        'metrics_colors': ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange', 'tab:brown', 'tab:pink'],
        'cmap': 'Blues'
    }
    
    MONOCHROME = {
        'positive_class': '#404040',
        'negative_class': '#808080',
        'roc_curve': '#000000',
        'pr_curve': '#000000',
        'threshold_line': '#000000',
        'calibration_curve': '#000000',
        'calibration_reference': '#808080',
        'metrics_colors': ['#000000', '#404040', '#606060', '#808080', '#A0A0A0', '#C0C0C0', '#E0E0E0'],
        'cmap': 'Greys'
    }
    
    VIBRANT = {
        'positive_class': '#FF6B6B',
        'negative_class': '#4ECDC4',
        'roc_curve': '#FF6B6B',
        'pr_curve': '#4ECDC4',
        'threshold_line': '#2C3E50',
        'calibration_curve': '#FF6B6B',
        'calibration_reference': '#95A5A6',
        'metrics_colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5', '#9B59B6'],
        'cmap': 'Greens'
    }

def calculate_metrics_by_threshold(y_true, y_scores):
    """Calculate various metrics across different thresholds."""
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

    return pd.DataFrame(metrics_by_threshold, columns=[
        "Threshold", "Accuracy", "Sensitivity", "Specificity",
        "PPV", "MCC", "F1 Score"
    ])

def bootstrap_metric(metric_func, y_true, y_scores, n_boot=1000):
    """Calculate bootstrap confidence intervals for a given metric."""
    stats = []
    for _ in tqdm(range(n_boot), desc="Bootstrap iterations", leave=False):
        indices = np.random.choice(range(len(y_true)), len(y_true), replace=True)
        try:
            stats.append(metric_func(y_true[indices], y_scores[indices]))
        except:
            continue
    return np.percentile(stats, [2.5, 97.5])

def bootstrap_curves(y_true, y_scores, n_boot=1000):
    """Calculate bootstrap confidence intervals for ROC and PR curves."""
    tprs = []
    fprs = []
    precisions = []
    recalls = []
    
    base_fpr, base_tpr, _ = roc_curve(y_true, y_scores)
    base_precision, base_recall, _ = precision_recall_curve(y_true, y_scores)
    
    common_fpr = np.linspace(0, 1, 100)
    common_recall = np.linspace(0, 1, 100)
    
    for _ in tqdm(range(n_boot), desc="Bootstrap iterations for curves", leave=False):
        indices = np.random.choice(range(len(y_true)), len(y_true), replace=True)
        try:
            fpr, tpr, _ = roc_curve(y_true[indices], y_scores[indices])
            tpr_interp = np.interp(common_fpr, fpr, tpr)
            tprs.append(tpr_interp)
            
            precision, recall, _ = precision_recall_curve(y_true[indices], y_scores[indices])
            sort_idx = np.argsort(recall)
            recall = recall[sort_idx]
            precision = precision[sort_idx]
            precision_interp = np.interp(common_recall, recall, precision)
            precisions.append(precision_interp)
        except:
            continue
    
    tpr_ci = np.percentile(tprs, [2.5, 97.5], axis=0)
    precision_ci = np.percentile(precisions, [2.5, 97.5], axis=0)
    
    return tpr_ci, precision_ci, common_fpr, common_recall

def calculate_optimal_threshold(y_true, y_scores):
    """Calculate the optimal threshold using ROC curve."""
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    return roc_thresholds[np.argmax(j_scores)]

def calculate_metrics_summary(y_true, y_scores, best_thresh):
    """Calculate summary metrics at the optimal threshold."""
    y_pred_opt = (y_scores >= best_thresh).astype(int)
    
    return {
        "Accuracy": accuracy_score(y_true, y_pred_opt),
        "Sensitivity": recall_score(y_true, y_pred_opt),
        "Specificity": recall_score(y_true, y_pred_opt, pos_label=0),
        "PPV": precision_score(y_true, y_pred_opt, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred_opt),
        "F1 Score": f1_score(y_true, y_pred_opt),
        "AUC-ROC": roc_auc_score(y_true, y_scores),
        "AUC-PR": average_precision_score(y_true, y_scores)
    }

def calculate_confidence_intervals(y_true, y_scores, best_thresh, n_bootstrap=1000):
    """Calculate confidence intervals for all metrics."""
    metric_functions = {
        "Accuracy": lambda yt, ys: accuracy_score(yt, ys >= best_thresh),
        "Sensitivity": lambda yt, ys: recall_score(yt, ys >= best_thresh),
        "Specificity": lambda yt, ys: recall_score(yt, ys >= best_thresh, pos_label=0),
        "PPV": lambda yt, ys: precision_score(yt, ys >= best_thresh, zero_division=0),
        "MCC": lambda yt, ys: matthews_corrcoef(yt, ys >= best_thresh),
        "F1 Score": lambda yt, ys: f1_score(yt, ys >= best_thresh),
        "AUC-ROC": lambda yt, ys: roc_auc_score(yt, ys),
        "AUC-PR": lambda yt, ys: average_precision_score(yt, ys)
    }
    
    return {
        name: bootstrap_metric(func, y_true, y_scores, n_boot=n_bootstrap)
        for name, func in metric_functions.items()
    }

def create_output_directories(output_path):
    """Create necessary output directories for plots and PDF."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    return plots_dir

def plot_roc_pr_curves(y_true, y_scores, tpr_ci, precision_ci, common_fpr, common_recall, colors, dpi, plots_dir):
    """Generate ROC and PR curves with confidence intervals."""
    plt.figure(figsize=(12, 5), dpi=dpi)
    
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.plot(fpr, tpr, label="ROC curve", color=colors['roc_curve'])
    plt.fill_between(common_fpr, tpr_ci[0], tpr_ci[1], alpha=0.3, color=colors['roc_curve'])
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.plot(recall, precision, label="PR curve", color=colors['pr_curve'])
    plt.fill_between(common_recall, precision_ci[0], precision_ci[1], alpha=0.3, color=colors['pr_curve'])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    
    plt.savefig(os.path.join(plots_dir, "roc_pr.png"), dpi=dpi, bbox_inches='tight')
    return plt.gcf()

def plot_metrics_threshold(metrics_df, colors, dpi, plots_dir):
    """Generate metrics vs threshold plot."""
    plt.figure(figsize=(10, 6), dpi=dpi)
    for i, col in enumerate(metrics_df.columns[1:]):
        plt.plot(metrics_df["Threshold"], metrics_df[col], label=col, 
                color=colors['metrics_colors'][i % len(colors['metrics_colors'])])
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Metrics Across Thresholds")
    plt.legend()
    
    plt.savefig(os.path.join(plots_dir, "metrics_threshold.png"), dpi=dpi, bbox_inches='tight')
    return plt.gcf()

def plot_confusion_matrix(y_true, y_scores, best_thresh, colors, dpi, plots_dir):
    """Generate confusion matrix plot."""
    cm = confusion_matrix(y_true, y_scores >= best_thresh)
    plt.figure(figsize=(5, 4), dpi=dpi)
    sns.heatmap(cm, annot=True, fmt="d", cmap=colors['cmap'], cbar=False, annot_kws={"size": 12})
    plt.title("Confusion Matrix (Optimal Threshold)", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=dpi, bbox_inches='tight')
    return plt.gcf()

def plot_calibration(y_true, y_scores, colors, dpi, plots_dir):
    """Generate calibration plot."""
    plt.figure(figsize=(6, 6), dpi=dpi)
    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=10, strategy='uniform')
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve', color=colors['calibration_curve'])
    plt.plot([0, 1], [0, 1], linestyle='--', color=colors['calibration_reference'])
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Plot')
    plt.legend()
    
    plt.savefig(os.path.join(plots_dir, "calibration.png"), dpi=dpi, bbox_inches='tight')
    return plt.gcf()

def plot_metrics_summary(metrics_summary, conf_intervals, dpi, plots_dir):
    """Generate metrics summary table plot."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
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
    
    plt.savefig(os.path.join(plots_dir, "metrics_summary.png"), dpi=dpi, bbox_inches='tight')
    return plt.gcf()

def plot_prediction_distribution(y_true, y_scores, best_thresh, colors, dpi, plots_dir):
    """Generate prediction distribution histogram."""
    plt.figure(figsize=(10, 6), dpi=dpi)
    plt.hist(y_scores[y_true == 1], bins=50, alpha=0.5, label='Positive Class', color=colors['positive_class'])
    plt.hist(y_scores[y_true == 0], bins=50, alpha=0.5, label='Negative Class', color=colors['negative_class'])
    plt.axvline(x=best_thresh, color=colors['threshold_line'], linestyle='--', 
                label=f'Optimal Threshold ({best_thresh:.3f})')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Predictions')
    plt.legend()
    
    plt.savefig(os.path.join(plots_dir, "prediction_distribution.png"), dpi=dpi, bbox_inches='tight')
    return plt.gcf() 