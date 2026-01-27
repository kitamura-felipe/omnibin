import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .regression_utils import (
    RegressionColorScheme, calculate_regression_metrics,
    calculate_regression_confidence_intervals, create_output_directories,
    plot_scatter_regression, plot_residuals, plot_qq, plot_bland_altman,
    plot_error_distribution, plot_metrics_summary_regression, plot_prediction_intervals
)


def validate_regression_inputs(y_true, y_pred):
    """Validate input arrays for regression analysis."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)} samples, y_pred has {len(y_pred)} samples")

    if len(y_true) < 3:
        raise ValueError("At least 3 samples are required for regression analysis")

    if not np.issubdtype(y_true.dtype, np.number) or not np.issubdtype(y_pred.dtype, np.number):
        raise ValueError("Both y_true and y_pred must contain numeric values")

    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Input arrays cannot contain NaN values")

    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Input arrays cannot contain infinite values")

    return y_true, y_pred


def generate_regression_report(y_true, y_pred, output_path="regression_report.pdf",
                               n_bootstrap=1000, random_seed=42, dpi=300,
                               color_scheme=RegressionColorScheme.DEFAULT):
    """
    Generate a comprehensive regression analysis report with visualizations and metrics.

    Parameters
    ----------
    y_true : array-like
        True/actual values (ground truth)
    y_pred : array-like
        Predicted values from the model
    output_path : str, optional
        Path for the output PDF report (default: "regression_report.pdf")
    n_bootstrap : int, optional
        Number of bootstrap iterations for confidence intervals (default: 1000)
    random_seed : int, optional
        Random seed for reproducibility (default: 42)
    dpi : int, optional
        DPI for plot resolution (default: 300)
    color_scheme : RegressionColorScheme, optional
        Color scheme for visualizations (default: RegressionColorScheme.DEFAULT)

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
    >>> y_true = np.random.randn(100) * 10 + 50
    >>> y_pred = y_true + np.random.randn(100) * 2
    >>> report_path = generate_regression_report(y_true, y_pred)
    """
    # Validate inputs
    y_true, y_pred = validate_regression_inputs(y_true, y_pred)

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Set DPI for all figures
    plt.rcParams['figure.dpi'] = dpi

    # Get color scheme
    colors = color_scheme.value

    # Calculate metrics
    metrics_summary = calculate_regression_metrics(y_true, y_pred)

    # Calculate confidence intervals
    conf_intervals = calculate_regression_confidence_intervals(y_true, y_pred, n_bootstrap)

    # Create output directories
    plots_dir = create_output_directories(output_path)

    with PdfPages(output_path) as pdf:
        # Generate and save all plots
        plots = [
            plot_scatter_regression(y_true, y_pred, colors, dpi, plots_dir),
            plot_residuals(y_true, y_pred, colors, dpi, plots_dir),
            plot_qq(y_true, y_pred, colors, dpi, plots_dir),
            plot_bland_altman(y_true, y_pred, colors, dpi, plots_dir),
            plot_error_distribution(y_true, y_pred, colors, dpi, plots_dir),
            plot_prediction_intervals(y_true, y_pred, colors, dpi, plots_dir),
            plot_metrics_summary_regression(metrics_summary, conf_intervals, dpi, plots_dir)
        ]

        # Save all plots to PDF
        for plot in plots:
            pdf.savefig(plot, dpi=dpi)
            plt.close(plot)

    return output_path
