import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .detection_utils import (
    DetectionColorScheme, calculate_detection_metrics,
    calculate_detection_confidence_intervals, create_output_directories,
    plot_precision_recall_curve_detection, plot_froc_curve,
    plot_confidence_distribution, plot_detections_per_image,
    plot_metrics_summary_detection, plot_iou_distribution
)


def validate_detection_inputs(predictions, ground_truths):
    """
    Validate input data for detection analysis.

    Parameters
    ----------
    predictions : list of lists
        List of predictions for each image, where each prediction is a dict
        containing 'box' (or 'bbox') and 'score'
    ground_truths : list of lists
        List of ground truth boxes for each image [x1, y1, x2, y2]

    Returns
    -------
    tuple
        Validated (predictions, ground_truths)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Number of prediction lists ({len(predictions)}) must match "
                         f"number of ground truth lists ({len(ground_truths)})")

    if len(predictions) == 0:
        raise ValueError("At least one image/sample is required")

    # Validate predictions format
    for img_idx, preds in enumerate(predictions):
        for pred_idx, pred in enumerate(preds):
            if not isinstance(pred, dict):
                raise ValueError(f"Prediction {pred_idx} in image {img_idx} must be a dictionary")

            if 'box' not in pred and 'bbox' not in pred:
                raise ValueError(f"Prediction {pred_idx} in image {img_idx} must contain 'box' or 'bbox' key")

            box = pred.get('box', pred.get('bbox'))
            if len(box) != 4:
                raise ValueError(f"Box in prediction {pred_idx}, image {img_idx} must have 4 values [x1,y1,x2,y2]")

    # Validate ground truths format
    for img_idx, gts in enumerate(ground_truths):
        for gt_idx, gt in enumerate(gts):
            if len(gt) != 4:
                raise ValueError(f"Ground truth {gt_idx} in image {img_idx} must have 4 values [x1,y1,x2,y2]")

    return predictions, ground_truths


def generate_detection_report(predictions, ground_truths, output_path="detection_report.pdf",
                              n_bootstrap=500, random_seed=42, dpi=300,
                              color_scheme=DetectionColorScheme.DEFAULT,
                              iou_thresholds=[0.5, 0.75]):
    """
    Generate a comprehensive object detection analysis report with visualizations and metrics.

    Parameters
    ----------
    predictions : list of lists
        List of predictions for each image. Each prediction is a dict with:
        - 'box' or 'bbox': [x1, y1, x2, y2] bounding box coordinates
        - 'score': confidence score (0-1)
    ground_truths : list of lists
        List of ground truth bounding boxes for each image [x1, y1, x2, y2]
    output_path : str, optional
        Path for the output PDF report (default: "detection_report.pdf")
    n_bootstrap : int, optional
        Number of bootstrap iterations for confidence intervals (default: 500)
    random_seed : int, optional
        Random seed for reproducibility (default: 42)
    dpi : int, optional
        DPI for plot resolution (default: 300)
    color_scheme : DetectionColorScheme, optional
        Color scheme for visualizations (default: DetectionColorScheme.DEFAULT)
    iou_thresholds : list, optional
        IoU thresholds for AP calculation (default: [0.5, 0.75])

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
    >>> # Predictions for 3 images
    >>> predictions = [
    ...     [{'box': [10, 10, 50, 50], 'score': 0.9}, {'box': [60, 60, 100, 100], 'score': 0.7}],
    ...     [{'box': [20, 20, 80, 80], 'score': 0.85}],
    ...     []
    ... ]
    >>> # Ground truths for 3 images
    >>> ground_truths = [
    ...     [[12, 12, 48, 48], [65, 65, 95, 95]],
    ...     [[25, 25, 75, 75]],
    ...     [[30, 30, 70, 70]]
    ... ]
    >>> report_path = generate_detection_report(predictions, ground_truths)
    """
    # Validate inputs
    predictions, ground_truths = validate_detection_inputs(predictions, ground_truths)

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Set DPI for all figures
    plt.rcParams['figure.dpi'] = dpi

    # Get color scheme
    colors = color_scheme.value

    # Calculate metrics
    metrics_summary = calculate_detection_metrics(predictions, ground_truths, iou_thresholds)

    # Calculate confidence intervals
    conf_intervals = calculate_detection_confidence_intervals(predictions, ground_truths, n_bootstrap)

    # Create output directories
    plots_dir = create_output_directories(output_path)

    with PdfPages(output_path) as pdf:
        # Generate and save all plots
        plots = [
            plot_precision_recall_curve_detection(predictions, ground_truths, iou_thresholds, colors, dpi, plots_dir),
            plot_froc_curve(predictions, ground_truths, colors, dpi, plots_dir),
            plot_iou_distribution(predictions, ground_truths, colors, dpi, plots_dir),
            plot_confidence_distribution(predictions, ground_truths, colors, dpi, plots_dir),
            plot_detections_per_image(predictions, ground_truths, colors, dpi, plots_dir),
            plot_metrics_summary_detection(metrics_summary, conf_intervals, dpi, plots_dir)
        ]

        # Save all plots to PDF
        for plot in plots:
            pdf.savefig(plot, dpi=dpi)
            plt.close(plot)

    return output_path


def generate_lesion_detection_report(predictions, ground_truths, output_path="lesion_detection_report.pdf",
                                     n_bootstrap=500, random_seed=42, dpi=300,
                                     color_scheme=DetectionColorScheme.DEFAULT,
                                     use_distance=False, distance_threshold=None):
    """
    Generate a detection report optimized for medical lesion detection.

    This is specifically designed for healthcare applications where:
    - FROC curves are the standard metric
    - Distance-based matching may be preferred over IoU
    - False positive rates per image are critical

    Parameters
    ----------
    predictions : list of lists
        List of predictions for each image/scan. Each prediction is a dict with:
        - 'box' or 'center': bounding box or center point
        - 'score': confidence score (0-1)
    ground_truths : list of lists
        List of ground truth annotations for each image/scan
    output_path : str, optional
        Path for the output PDF report
    n_bootstrap : int, optional
        Number of bootstrap iterations for confidence intervals
    random_seed : int, optional
        Random seed for reproducibility
    dpi : int, optional
        DPI for plot resolution
    color_scheme : DetectionColorScheme, optional
        Color scheme for visualizations
    use_distance : bool, optional
        If True, use distance-based matching instead of IoU
    distance_threshold : float, optional
        Distance threshold for matching (only used if use_distance=True)

    Returns
    -------
    str
        Path to the generated PDF report
    """
    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)

    plt.rcParams['figure.dpi'] = dpi
    colors = color_scheme.value

    # Calculate metrics with medical imaging focus
    from .detection_utils import (
        calculate_froc_curve, calculate_froc_score,
        calculate_precision_recall_curve, calculate_ap
    )

    # FROC analysis
    sensitivities, fp_per_image, thresholds = calculate_froc_curve(
        predictions, ground_truths, use_distance=use_distance,
        distance_threshold=distance_threshold
    )
    froc_score = calculate_froc_score(sensitivities, fp_per_image)

    # Standard detection metrics
    metrics_summary = calculate_detection_metrics(predictions, ground_truths)
    metrics_summary["FROC Score"] = froc_score

    # Sensitivities at specific FP rates (medical imaging standard)
    fp_rates = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    for fp_rate in fp_rates:
        idx = np.searchsorted(fp_per_image, fp_rate)
        if idx < len(sensitivities):
            metrics_summary[f"Sens@{fp_rate}FP"] = sensitivities[min(idx, len(sensitivities) - 1)]

    conf_intervals = calculate_detection_confidence_intervals(predictions, ground_truths, n_bootstrap)

    plots_dir = create_output_directories(output_path)

    with PdfPages(output_path) as pdf:
        # FROC curve (primary metric for medical imaging)
        fig = plot_froc_curve(predictions, ground_truths, colors, dpi, plots_dir)
        pdf.savefig(fig, dpi=dpi)
        plt.close(fig)

        # Sensitivity at different FP rates
        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
        sens_at_fps = [metrics_summary.get(f"Sens@{fp}FP", 0) for fp in fp_rates]
        bars = ax.bar([str(fp) for fp in fp_rates], sens_at_fps, color=colors['froc'], edgecolor='black')

        for bar, val in zip(bars, sens_at_fps):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('False Positives per Image', fontsize=12)
        ax.set_ylabel('Sensitivity', fontsize=12)
        ax.set_title('Sensitivity at Different FP Rates', fontsize=14)
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3)

        plt.savefig(os.path.join(plots_dir, "sensitivity_at_fp.png"), dpi=dpi, bbox_inches='tight')
        pdf.savefig(fig, dpi=dpi)
        plt.close(fig)

        # Additional plots
        plots = [
            plot_precision_recall_curve_detection(predictions, ground_truths, [0.5, 0.75], colors, dpi, plots_dir),
            plot_confidence_distribution(predictions, ground_truths, colors, dpi, plots_dir),
            plot_metrics_summary_detection(metrics_summary, conf_intervals, dpi, plots_dir)
        ]

        for plot in plots:
            pdf.savefig(plot, dpi=dpi)
            plt.close(plot)

    return output_path


# Import os for the lesion detection report function
import os
