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
from enum import Enum
from .utils import (
    ColorScheme, calculate_metrics_by_threshold, bootstrap_curves,
    calculate_optimal_threshold, calculate_metrics_summary,
    calculate_confidence_intervals, create_output_directories,
    plot_roc_pr_curves, plot_metrics_threshold, plot_confusion_matrix,
    plot_calibration, plot_metrics_summary, plot_prediction_distribution
)

def generate_binary_classification_report(y_true, y_scores, output_path="omnibin_report.pdf", n_bootstrap=1000, random_seed=42, dpi=300, color_scheme=ColorScheme.DEFAULT):
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Set DPI for all figures
    plt.rcParams['figure.dpi'] = dpi
    
    # Get color scheme
    colors = color_scheme.value

    # Calculate metrics and optimal threshold
    metrics_df = calculate_metrics_by_threshold(y_true, y_scores)
    best_thresh = calculate_optimal_threshold(y_true, y_scores)
    metrics_summary = calculate_metrics_summary(y_true, y_scores, best_thresh)
    conf_intervals = calculate_confidence_intervals(y_true, y_scores, best_thresh, n_bootstrap)

    # Create output directories
    plots_dir = create_output_directories(output_path)

    # Calculate confidence intervals for curves
    tpr_ci, precision_ci, common_fpr, common_recall = bootstrap_curves(y_true, y_scores, n_boot=n_bootstrap)

    with PdfPages(output_path) as pdf:
        # Generate and save all plots
        plots = [
            plot_roc_pr_curves(y_true, y_scores, tpr_ci, precision_ci, common_fpr, common_recall, colors, dpi, plots_dir),
            plot_metrics_threshold(metrics_df, colors, dpi, plots_dir),
            plot_confusion_matrix(y_true, y_scores, best_thresh, colors, dpi, plots_dir),
            plot_calibration(y_true, y_scores, colors, dpi, plots_dir),
            plot_metrics_summary(metrics_summary, conf_intervals, dpi, plots_dir),
            plot_prediction_distribution(y_true, y_scores, best_thresh, colors, dpi, plots_dir)
        ]
        
        # Save all plots to PDF
        for plot in plots:
            pdf.savefig(plot, dpi=dpi)
            plt.close(plot)

    return output_path