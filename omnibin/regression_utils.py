import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    max_error, median_absolute_error
)
from scipy import stats
from enum import Enum
import os


class RegressionColorScheme(Enum):
    DEFAULT = {
        'scatter': 'tab:blue',
        'regression_line': 'tab:red',
        'residual_line': 'tab:orange',
        'reference_line': 'gray',
        'histogram': 'tab:blue',
        'qq_line': 'tab:red',
        'bland_altman_mean': 'tab:blue',
        'bland_altman_limits': 'tab:red',
        'metrics_colors': ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange', 'tab:brown'],
        'cmap': 'Blues'
    }

    MONOCHROME = {
        'scatter': '#404040',
        'regression_line': '#000000',
        'residual_line': '#606060',
        'reference_line': '#808080',
        'histogram': '#404040',
        'qq_line': '#000000',
        'bland_altman_mean': '#404040',
        'bland_altman_limits': '#000000',
        'metrics_colors': ['#000000', '#404040', '#606060', '#808080', '#A0A0A0', '#C0C0C0'],
        'cmap': 'Greys'
    }

    VIBRANT = {
        'scatter': '#4ECDC4',
        'regression_line': '#FF6B6B',
        'residual_line': '#45B7D1',
        'reference_line': '#95A5A6',
        'histogram': '#4ECDC4',
        'qq_line': '#FF6B6B',
        'bland_altman_mean': '#4ECDC4',
        'bland_altman_limits': '#FF6B6B',
        'metrics_colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5'],
        'cmap': 'Greens'
    }


def calculate_regression_metrics(y_true, y_pred):
    """Calculate all regression metrics."""
    n = len(y_true)
    p = 1  # number of predictors (assuming simple regression)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Adjusted R² (for simple regression with 1 predictor)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

    # Handle MAPE with zero values
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    except:
        mape = np.nan

    ev = explained_variance_score(y_true, y_pred)
    max_err = max_error(y_true, y_pred)
    med_ae = median_absolute_error(y_true, y_pred)

    # Normalized RMSE (as percentage of range)
    y_range = np.max(y_true) - np.min(y_true)
    nrmse = (rmse / y_range * 100) if y_range > 0 else np.nan

    # Coefficient of Variation of RMSE
    cv_rmse = (rmse / np.mean(y_true) * 100) if np.mean(y_true) != 0 else np.nan

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
        "Adjusted R²": adj_r2,
        "MAPE (%)": mape,
        "Explained Variance": ev,
        "Max Error": max_err,
        "Median AE": med_ae,
        "NRMSE (%)": nrmse,
        "CV-RMSE (%)": cv_rmse
    }


def bootstrap_regression_metric(metric_func, y_true, y_pred, n_boot=1000):
    """Calculate bootstrap confidence intervals for a regression metric."""
    stats_list = []
    for _ in tqdm(range(n_boot), desc="Bootstrap iterations", leave=False):
        indices = np.random.choice(range(len(y_true)), len(y_true), replace=True)
        try:
            stats_list.append(metric_func(y_true[indices], y_pred[indices]))
        except:
            continue
    if len(stats_list) == 0:
        return [np.nan, np.nan]
    return np.percentile(stats_list, [2.5, 97.5])


def calculate_regression_confidence_intervals(y_true, y_pred, n_bootstrap=1000):
    """Calculate confidence intervals for all regression metrics."""

    def safe_mape(yt, yp):
        try:
            return mean_absolute_percentage_error(yt, yp) * 100
        except:
            return np.nan

    def safe_nrmse(yt, yp):
        y_range = np.max(yt) - np.min(yt)
        if y_range > 0:
            return np.sqrt(mean_squared_error(yt, yp)) / y_range * 100
        return np.nan

    def safe_cv_rmse(yt, yp):
        mean_val = np.mean(yt)
        if mean_val != 0:
            return np.sqrt(mean_squared_error(yt, yp)) / mean_val * 100
        return np.nan

    metric_functions = {
        "MAE": lambda yt, yp: mean_absolute_error(yt, yp),
        "MSE": lambda yt, yp: mean_squared_error(yt, yp),
        "RMSE": lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
        "R²": lambda yt, yp: r2_score(yt, yp),
        "Adjusted R²": lambda yt, yp: 1 - (1 - r2_score(yt, yp)) * (len(yt) - 1) / (len(yt) - 2) if len(yt) > 2 else r2_score(yt, yp),
        "MAPE (%)": safe_mape,
        "Explained Variance": lambda yt, yp: explained_variance_score(yt, yp),
        "Max Error": lambda yt, yp: max_error(yt, yp),
        "Median AE": lambda yt, yp: median_absolute_error(yt, yp),
        "NRMSE (%)": safe_nrmse,
        "CV-RMSE (%)": safe_cv_rmse
    }

    return {
        name: bootstrap_regression_metric(func, y_true, y_pred, n_boot=n_bootstrap)
        for name, func in metric_functions.items()
    }


def create_output_directories(output_path):
    """Create necessary output directories for plots and PDF."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plots_dir = os.path.join(output_dir, "plots") if output_dir else "plots"
    os.makedirs(plots_dir, exist_ok=True)

    return plots_dir


def plot_scatter_regression(y_true, y_pred, colors, dpi, plots_dir):
    """Generate scatter plot with regression line."""
    plt.figure(figsize=(8, 8), dpi=dpi)

    plt.scatter(y_true, y_pred, alpha=0.5, color=colors['scatter'], label='Predictions')

    # Add perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], '--', color=colors['reference_line'],
             label='Perfect Prediction', linewidth=2)

    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
    line = slope * np.array([min_val, max_val]) + intercept
    plt.plot([min_val, max_val], line, color=colors['regression_line'],
             label=f'Regression (R={r_value:.3f})', linewidth=2)

    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title('Predicted vs True Values', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(plots_dir, "scatter_regression.png"), dpi=dpi, bbox_inches='tight')
    return plt.gcf()


def plot_residuals(y_true, y_pred, colors, dpi, plots_dir):
    """Generate residual plots."""
    residuals = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=dpi)

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, color=colors['scatter'])
    axes[0].axhline(y=0, color=colors['residual_line'], linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title('Residuals vs Predicted Values', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Residuals histogram
    axes[1].hist(residuals, bins=30, color=colors['histogram'], alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color=colors['residual_line'], linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Residuals', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "residuals.png"), dpi=dpi, bbox_inches='tight')
    return fig


def plot_qq(y_true, y_pred, colors, dpi, plots_dir):
    """Generate Q-Q plot for residuals."""
    residuals = y_pred - y_true

    plt.figure(figsize=(7, 7), dpi=dpi)

    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(plots_dir, "qq_plot.png"), dpi=dpi, bbox_inches='tight')
    return plt.gcf()


def plot_bland_altman(y_true, y_pred, colors, dpi, plots_dir):
    """Generate Bland-Altman plot for agreement analysis (common in healthcare)."""
    mean_vals = (y_true + y_pred) / 2
    diff = y_pred - y_true

    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    plt.figure(figsize=(10, 6), dpi=dpi)

    plt.scatter(mean_vals, diff, alpha=0.5, color=colors['scatter'])
    plt.axhline(y=mean_diff, color=colors['bland_altman_mean'], linestyle='-',
                label=f'Mean: {mean_diff:.3f}', linewidth=2)
    plt.axhline(y=mean_diff + 1.96 * std_diff, color=colors['bland_altman_limits'],
                linestyle='--', label=f'+1.96 SD: {mean_diff + 1.96 * std_diff:.3f}', linewidth=2)
    plt.axhline(y=mean_diff - 1.96 * std_diff, color=colors['bland_altman_limits'],
                linestyle='--', label=f'-1.96 SD: {mean_diff - 1.96 * std_diff:.3f}', linewidth=2)

    plt.xlabel('Mean of True and Predicted', fontsize=12)
    plt.ylabel('Difference (Predicted - True)', fontsize=12)
    plt.title('Bland-Altman Plot', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(plots_dir, "bland_altman.png"), dpi=dpi, bbox_inches='tight')
    return plt.gcf()


def plot_error_distribution(y_true, y_pred, colors, dpi, plots_dir):
    """Generate error distribution plots."""
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    pct_errors = np.abs(errors / y_true) * 100 if np.all(y_true != 0) else None

    n_plots = 3 if pct_errors is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), dpi=dpi)

    # Error histogram
    axes[0].hist(errors, bins=30, color=colors['histogram'], alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color=colors['residual_line'], linestyle='--', linewidth=2)
    axes[0].set_xlabel('Error', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Error Distribution', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Absolute error histogram
    axes[1].hist(abs_errors, bins=30, color=colors['histogram'], alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Absolute Error', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Absolute Error Distribution', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    # Percentage error histogram (if applicable)
    if pct_errors is not None:
        # Cap at 100% for visualization
        pct_errors_capped = np.clip(pct_errors, 0, 100)
        axes[2].hist(pct_errors_capped, bins=30, color=colors['histogram'], alpha=0.7, edgecolor='black')
        axes[2].set_xlabel('Absolute Percentage Error (%)', fontsize=12)
        axes[2].set_ylabel('Frequency', fontsize=12)
        axes[2].set_title('Percentage Error Distribution', fontsize=14)
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "error_distribution.png"), dpi=dpi, bbox_inches='tight')
    return fig


def plot_metrics_summary_regression(metrics_summary, conf_intervals, dpi, plots_dir):
    """Generate metrics summary table plot for regression."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    ax.axis("off")

    table_data = []
    for k, v in metrics_summary.items():
        ci = conf_intervals.get(k, [np.nan, np.nan])
        if np.isnan(v):
            table_data.append([k, "N/A", "N/A"])
        else:
            table_data.append([k, f"{v:.4f}", f"[{ci[0]:.4f}, {ci[1]:.4f}]"])

    table = ax.table(cellText=table_data, colLabels=["Metric", "Value", "95% CI"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for j in range(3):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax.set_title("Regression Performance Metrics", fontweight="bold", fontsize=14, pad=20)

    plt.savefig(os.path.join(plots_dir, "metrics_summary.png"), dpi=dpi, bbox_inches='tight')
    return fig


def plot_prediction_intervals(y_true, y_pred, colors, dpi, plots_dir):
    """Generate prediction interval visualization."""
    # Sort by true values for better visualization
    sort_idx = np.argsort(y_true)
    y_true_sorted = y_true[sort_idx]
    y_pred_sorted = y_pred[sort_idx]

    residuals = y_pred - y_true
    std_residuals = np.std(residuals)

    plt.figure(figsize=(12, 6), dpi=dpi)

    x_range = np.arange(len(y_true_sorted))

    plt.plot(x_range, y_true_sorted, 'o', color=colors['scatter'], alpha=0.5,
             label='True Values', markersize=4)
    plt.plot(x_range, y_pred_sorted, '-', color=colors['regression_line'],
             label='Predictions', linewidth=1)

    # Add prediction intervals (95%)
    upper = y_pred_sorted + 1.96 * std_residuals
    lower = y_pred_sorted - 1.96 * std_residuals
    plt.fill_between(x_range, lower, upper, alpha=0.2, color=colors['regression_line'],
                     label='95% Prediction Interval')

    plt.xlabel('Sample Index (sorted by true value)', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Predictions with 95% Prediction Intervals', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(plots_dir, "prediction_intervals.png"), dpi=dpi, bbox_inches='tight')
    return plt.gcf()
