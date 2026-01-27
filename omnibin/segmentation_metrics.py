import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .segmentation_utils import (
    SegmentationColorScheme, calculate_segmentation_metrics,
    calculate_multiclass_segmentation_metrics,
    calculate_segmentation_confidence_intervals, create_output_directories,
    plot_segmentation_comparison, plot_confusion_matrix_segmentation,
    plot_dice_distribution, plot_metrics_summary_segmentation,
    plot_surface_distance_histogram, plot_metrics_bar_chart, dice_score
)


def validate_segmentation_inputs(y_true, y_pred):
    """Validate input arrays for segmentation analysis."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true has shape {y_true.shape}, y_pred has shape {y_pred.shape}")

    if y_true.ndim < 2:
        raise ValueError("Segmentation masks must be at least 2D (height x width)")

    if y_true.ndim > 3:
        raise ValueError("Segmentation masks must be at most 3D (depth x height x width)")

    return y_true, y_pred


def generate_segmentation_report(y_true, y_pred, output_path="segmentation_report.pdf",
                                 n_bootstrap=500, random_seed=42, dpi=300,
                                 color_scheme=SegmentationColorScheme.DEFAULT,
                                 class_names=None, slice_idx=None):
    """
    Generate a comprehensive segmentation analysis report with visualizations and metrics.

    Parameters
    ----------
    y_true : ndarray
        Ground truth segmentation mask (2D or 3D binary array)
    y_pred : ndarray
        Predicted segmentation mask (2D or 3D binary array)
    output_path : str, optional
        Path for the output PDF report (default: "segmentation_report.pdf")
    n_bootstrap : int, optional
        Number of bootstrap iterations for confidence intervals (default: 500)
    random_seed : int, optional
        Random seed for reproducibility (default: 42)
    dpi : int, optional
        DPI for plot resolution (default: 300)
    color_scheme : SegmentationColorScheme, optional
        Color scheme for visualizations (default: SegmentationColorScheme.DEFAULT)
    class_names : dict, optional
        Dictionary mapping class indices to names (for multi-class segmentation)
    slice_idx : int, optional
        For 3D data, which slice to visualize (default: slice with most foreground)

    Returns
    -------
    str
        Path to the generated PDF report

    Raises
    ------
    ValueError
        If input validation fails

    Examples
    --------
    >>> import numpy as np
    >>> # Create synthetic 2D segmentation masks
    >>> y_true = np.zeros((256, 256), dtype=np.uint8)
    >>> y_true[100:150, 100:150] = 1
    >>> y_pred = np.zeros((256, 256), dtype=np.uint8)
    >>> y_pred[95:155, 95:155] = 1
    >>> report_path = generate_segmentation_report(y_true, y_pred)
    """
    # Validate inputs
    y_true, y_pred = validate_segmentation_inputs(y_true, y_pred)

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Set DPI for all figures
    plt.rcParams['figure.dpi'] = dpi

    # Get color scheme
    colors = color_scheme.value

    # Calculate metrics
    metrics_summary = calculate_segmentation_metrics(y_true, y_pred)

    # Calculate confidence intervals (with reduced bootstrap for segmentation)
    conf_intervals = calculate_segmentation_confidence_intervals(y_true, y_pred, n_bootstrap)

    # Create output directories
    plots_dir = create_output_directories(output_path)

    # Calculate per-slice Dice for 3D data
    dice_per_slice = None
    if y_true.ndim == 3:
        dice_per_slice = []
        for i in range(y_true.shape[0]):
            if np.sum(y_true[i]) > 0 or np.sum(y_pred[i]) > 0:
                dice_per_slice.append(dice_score(y_true[i], y_pred[i]))

    with PdfPages(output_path) as pdf:
        # Generate and save all plots
        plots = [
            plot_segmentation_comparison(y_true, y_pred, colors, dpi, plots_dir, slice_idx),
            plot_confusion_matrix_segmentation(y_true, y_pred, colors, dpi, plots_dir),
            plot_metrics_bar_chart(metrics_summary, colors, dpi, plots_dir),
            plot_surface_distance_histogram(y_true, y_pred, colors, dpi, plots_dir),
            plot_metrics_summary_segmentation(metrics_summary, conf_intervals, dpi, plots_dir)
        ]

        # Add Dice distribution for 3D data
        if dice_per_slice is not None and len(dice_per_slice) > 1:
            plots.insert(2, plot_dice_distribution(np.array(dice_per_slice), colors, dpi, plots_dir))

        # Save all plots to PDF
        for plot in plots:
            pdf.savefig(plot, dpi=dpi)
            plt.close(plot)

    return output_path


def generate_multiclass_segmentation_report(y_true, y_pred, output_path="multiclass_segmentation_report.pdf",
                                            n_bootstrap=500, random_seed=42, dpi=300,
                                            color_scheme=SegmentationColorScheme.DEFAULT,
                                            class_names=None):
    """
    Generate a comprehensive multi-class segmentation analysis report.

    Parameters
    ----------
    y_true : ndarray
        Ground truth segmentation mask with integer class labels
    y_pred : ndarray
        Predicted segmentation mask with integer class labels
    output_path : str, optional
        Path for the output PDF report (default: "multiclass_segmentation_report.pdf")
    n_bootstrap : int, optional
        Number of bootstrap iterations for confidence intervals (default: 500)
    random_seed : int, optional
        Random seed for reproducibility (default: 42)
    dpi : int, optional
        DPI for plot resolution (default: 300)
    color_scheme : SegmentationColorScheme, optional
        Color scheme for visualizations (default: SegmentationColorScheme.DEFAULT)
    class_names : dict, optional
        Dictionary mapping class indices to names

    Returns
    -------
    str
        Path to the generated PDF report
    """
    # Validate inputs
    y_true, y_pred = validate_segmentation_inputs(y_true, y_pred)

    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)

    plt.rcParams['figure.dpi'] = dpi
    colors = color_scheme.value

    # Calculate per-class metrics
    class_metrics = calculate_multiclass_segmentation_metrics(y_true, y_pred, class_names)

    # Create output directories
    plots_dir = create_output_directories(output_path)

    with PdfPages(output_path) as pdf:
        # Plot per-class Dice scores
        fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)

        class_keys = [k for k in class_metrics.keys() if k != "Mean"]
        dice_scores = [class_metrics[k]["Dice Score"] for k in class_keys]
        iou_scores = [class_metrics[k]["IoU (Jaccard)"] for k in class_keys]

        x = np.arange(len(class_keys))
        width = 0.35

        ax.bar(x - width / 2, dice_scores, width, label='Dice Score', color=colors['ground_truth'])
        ax.bar(x + width / 2, iou_scores, width, label='IoU', color=colors['prediction'])

        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Segmentation Metrics', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(class_keys, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)

        # Add mean lines
        ax.axhline(y=class_metrics["Mean"]["Dice Score"], color='red', linestyle='--',
                   label=f'Mean Dice: {class_metrics["Mean"]["Dice Score"]:.3f}')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "per_class_metrics.png"), dpi=dpi, bbox_inches='tight')
        pdf.savefig(fig, dpi=dpi)
        plt.close(fig)

        # Summary table
        fig, ax = plt.subplots(figsize=(14, 8), dpi=dpi)
        ax.axis("off")

        table_data = []
        for cls_name, metrics in class_metrics.items():
            if cls_name == "Mean":
                row = [cls_name, f"{metrics['Dice Score']:.4f}", f"{metrics['IoU (Jaccard)']:.4f}",
                       "-", "-", "-", "-"]
            else:
                row = [cls_name,
                       f"{metrics['Dice Score']:.4f}",
                       f"{metrics['IoU (Jaccard)']:.4f}",
                       f"{metrics['Sensitivity (Recall)']:.4f}",
                       f"{metrics['Specificity']:.4f}",
                       f"{metrics['Precision (PPV)']:.4f}",
                       f"{metrics['Volumetric Similarity']:.4f}"]
            table_data.append(row)

        table = ax.table(cellText=table_data,
                         colLabels=["Class", "Dice", "IoU", "Sensitivity", "Specificity", "Precision", "Vol. Sim."],
                         loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Style header
        for j in range(7):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        ax.set_title("Multi-Class Segmentation Performance Metrics", fontweight="bold", fontsize=14, pad=20)

        plt.savefig(os.path.join(plots_dir, "multiclass_summary.png"), dpi=dpi, bbox_inches='tight')
        pdf.savefig(fig, dpi=dpi)
        plt.close(fig)

    return output_path


# Import os for the multiclass report function
import os
