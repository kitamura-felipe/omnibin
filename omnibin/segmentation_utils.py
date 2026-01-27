import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from enum import Enum
import os


class SegmentationColorScheme(Enum):
    DEFAULT = {
        'true_positive': '#2ecc71',
        'false_positive': '#e74c3c',
        'false_negative': '#3498db',
        'true_negative': '#ecf0f1',
        'ground_truth': 'tab:blue',
        'prediction': 'tab:orange',
        'overlap': 'tab:green',
        'metrics_colors': ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange', 'tab:brown'],
        'cmap': 'Blues',
        'cmap_error': 'RdYlGn'
    }

    MONOCHROME = {
        'true_positive': '#404040',
        'false_positive': '#808080',
        'false_negative': '#606060',
        'true_negative': '#f0f0f0',
        'ground_truth': '#000000',
        'prediction': '#606060',
        'overlap': '#404040',
        'metrics_colors': ['#000000', '#404040', '#606060', '#808080', '#A0A0A0', '#C0C0C0'],
        'cmap': 'Greys',
        'cmap_error': 'Greys'
    }

    VIBRANT = {
        'true_positive': '#2ecc71',
        'false_positive': '#e74c3c',
        'false_negative': '#3498db',
        'true_negative': '#ecf0f1',
        'ground_truth': '#4ECDC4',
        'prediction': '#FF6B6B',
        'overlap': '#96CEB4',
        'metrics_colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5'],
        'cmap': 'Greens',
        'cmap_error': 'RdYlGn'
    }


def dice_score(y_true, y_pred, smooth=1e-7):
    """
    Calculate Dice Similarity Coefficient (F1 for segmentation).

    DSC = 2 * |X ∩ Y| / (|X| + |Y|)
    """
    y_true = y_true.flatten().astype(bool)
    y_pred = y_pred.flatten().astype(bool)

    intersection = np.sum(y_true & y_pred)
    return (2.0 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def iou_score(y_true, y_pred, smooth=1e-7):
    """
    Calculate Intersection over Union (Jaccard Index).

    IoU = |X ∩ Y| / |X ∪ Y|
    """
    y_true = y_true.flatten().astype(bool)
    y_pred = y_pred.flatten().astype(bool)

    intersection = np.sum(y_true & y_pred)
    union = np.sum(y_true | y_pred)
    return (intersection + smooth) / (union + smooth)


def pixel_accuracy(y_true, y_pred):
    """Calculate overall pixel accuracy."""
    y_true = y_true.flatten().astype(bool)
    y_pred = y_pred.flatten().astype(bool)
    return np.mean(y_true == y_pred)


def sensitivity(y_true, y_pred, smooth=1e-7):
    """
    Calculate sensitivity (True Positive Rate / Recall).

    Sensitivity = TP / (TP + FN)
    """
    y_true = y_true.flatten().astype(bool)
    y_pred = y_pred.flatten().astype(bool)

    tp = np.sum(y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    return (tp + smooth) / (tp + fn + smooth)


def specificity(y_true, y_pred, smooth=1e-7):
    """
    Calculate specificity (True Negative Rate).

    Specificity = TN / (TN + FP)
    """
    y_true = y_true.flatten().astype(bool)
    y_pred = y_pred.flatten().astype(bool)

    tn = np.sum(~y_true & ~y_pred)
    fp = np.sum(~y_true & y_pred)
    return (tn + smooth) / (tn + fp + smooth)


def precision_seg(y_true, y_pred, smooth=1e-7):
    """
    Calculate precision (Positive Predictive Value).

    Precision = TP / (TP + FP)
    """
    y_true = y_true.flatten().astype(bool)
    y_pred = y_pred.flatten().astype(bool)

    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    return (tp + smooth) / (tp + fp + smooth)


def volumetric_similarity(y_true, y_pred):
    """
    Calculate Volumetric Similarity.

    VS = 1 - |V_pred - V_true| / (V_pred + V_true)
    """
    v_true = np.sum(y_true.astype(bool))
    v_pred = np.sum(y_pred.astype(bool))

    if v_true + v_pred == 0:
        return 1.0

    return 1 - np.abs(v_pred - v_true) / (v_pred + v_true)


def get_surface_points(mask):
    """Extract surface/boundary points from a binary mask."""
    if mask.ndim == 2:
        # 2D: use edge detection
        eroded = ndimage.binary_erosion(mask)
        surface = mask.astype(int) - eroded.astype(int)
    else:
        # 3D: use morphological operations
        eroded = ndimage.binary_erosion(mask)
        surface = mask.astype(int) - eroded.astype(int)

    return np.argwhere(surface > 0)


def hausdorff_distance(y_true, y_pred):
    """
    Calculate Hausdorff Distance between two binary masks.

    HD(A, B) = max(h(A, B), h(B, A))
    where h(A, B) = max_{a ∈ A} min_{b ∈ B} ||a - b||
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    # Get surface points
    surface_true = get_surface_points(y_true)
    surface_pred = get_surface_points(y_pred)

    if len(surface_true) == 0 or len(surface_pred) == 0:
        return np.nan

    # Calculate directed Hausdorff distances
    hd_true_to_pred = directed_hausdorff(surface_true, surface_pred)[0]
    hd_pred_to_true = directed_hausdorff(surface_pred, surface_true)[0]

    return max(hd_true_to_pred, hd_pred_to_true)


def hausdorff_distance_95(y_true, y_pred):
    """
    Calculate 95th percentile Hausdorff Distance.

    More robust to outliers than standard HD.
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    surface_true = get_surface_points(y_true)
    surface_pred = get_surface_points(y_pred)

    if len(surface_true) == 0 or len(surface_pred) == 0:
        return np.nan

    # Calculate all pairwise distances
    from scipy.spatial.distance import cdist
    distances_true_to_pred = cdist(surface_true, surface_pred).min(axis=1)
    distances_pred_to_true = cdist(surface_pred, surface_true).min(axis=1)

    all_distances = np.concatenate([distances_true_to_pred, distances_pred_to_true])

    return np.percentile(all_distances, 95)


def average_surface_distance(y_true, y_pred):
    """
    Calculate Average Surface Distance (ASD).

    ASD = (1/|S_A| + 1/|S_B|) * (Σ d(a, B) + Σ d(b, A))
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    surface_true = get_surface_points(y_true)
    surface_pred = get_surface_points(y_pred)

    if len(surface_true) == 0 or len(surface_pred) == 0:
        return np.nan

    from scipy.spatial.distance import cdist
    distances_true_to_pred = cdist(surface_true, surface_pred).min(axis=1)
    distances_pred_to_true = cdist(surface_pred, surface_true).min(axis=1)

    return (np.mean(distances_true_to_pred) + np.mean(distances_pred_to_true)) / 2


def calculate_segmentation_metrics(y_true, y_pred):
    """Calculate all segmentation metrics."""
    return {
        "Dice Score": dice_score(y_true, y_pred),
        "IoU (Jaccard)": iou_score(y_true, y_pred),
        "Pixel Accuracy": pixel_accuracy(y_true, y_pred),
        "Sensitivity (Recall)": sensitivity(y_true, y_pred),
        "Specificity": specificity(y_true, y_pred),
        "Precision (PPV)": precision_seg(y_true, y_pred),
        "Volumetric Similarity": volumetric_similarity(y_true, y_pred),
        "Hausdorff Distance": hausdorff_distance(y_true, y_pred),
        "Hausdorff Distance 95%": hausdorff_distance_95(y_true, y_pred),
        "Avg Surface Distance": average_surface_distance(y_true, y_pred)
    }


def calculate_multiclass_segmentation_metrics(y_true, y_pred, class_names=None):
    """
    Calculate segmentation metrics for multi-class segmentation.

    Parameters
    ----------
    y_true : ndarray
        Ground truth labels (integer encoded)
    y_pred : ndarray
        Predicted labels (integer encoded)
    class_names : list, optional
        Names for each class

    Returns
    -------
    dict
        Metrics for each class and mean metrics
    """
    unique_classes = np.unique(np.concatenate([y_true.flatten(), y_pred.flatten()]))
    unique_classes = unique_classes[unique_classes != 0]  # Exclude background

    if class_names is None:
        class_names = {c: f"Class {c}" for c in unique_classes}

    results = {}
    all_dice = []
    all_iou = []

    for cls in unique_classes:
        y_true_cls = (y_true == cls)
        y_pred_cls = (y_pred == cls)

        cls_metrics = calculate_segmentation_metrics(y_true_cls, y_pred_cls)
        cls_name = class_names.get(cls, f"Class {cls}")
        results[cls_name] = cls_metrics

        all_dice.append(cls_metrics["Dice Score"])
        all_iou.append(cls_metrics["IoU (Jaccard)"])

    # Calculate mean metrics
    results["Mean"] = {
        "Dice Score": np.mean(all_dice),
        "IoU (Jaccard)": np.mean(all_iou)
    }

    return results


def bootstrap_segmentation_metric(metric_func, y_true, y_pred, n_boot=1000):
    """Calculate bootstrap confidence intervals for a segmentation metric."""
    # For segmentation, we bootstrap over slices/samples if 3D, or patches if 2D
    stats_list = []

    # If 2D, we'll bootstrap by randomly sampling patches
    if y_true.ndim == 2:
        h, w = y_true.shape
        patch_size = min(h, w) // 4

        for _ in tqdm(range(n_boot), desc="Bootstrap iterations", leave=False):
            # Random patch
            y_start = np.random.randint(0, h - patch_size)
            x_start = np.random.randint(0, w - patch_size)

            y_true_patch = y_true[y_start:y_start + patch_size, x_start:x_start + patch_size]
            y_pred_patch = y_pred[y_start:y_start + patch_size, x_start:x_start + patch_size]

            try:
                stats_list.append(metric_func(y_true_patch, y_pred_patch))
            except:
                continue
    elif y_true.ndim == 3:
        # Bootstrap over slices
        n_slices = y_true.shape[0]
        for _ in tqdm(range(n_boot), desc="Bootstrap iterations", leave=False):
            indices = np.random.choice(n_slices, n_slices, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            try:
                stats_list.append(metric_func(y_true_boot, y_pred_boot))
            except:
                continue
    else:
        # Fallback: bootstrap over flattened pixels
        flat_true = y_true.flatten()
        flat_pred = y_pred.flatten()
        n_pixels = len(flat_true)
        sample_size = min(n_pixels, 10000)

        for _ in tqdm(range(n_boot), desc="Bootstrap iterations", leave=False):
            indices = np.random.choice(n_pixels, sample_size, replace=True)
            try:
                stats_list.append(metric_func(
                    flat_true[indices].reshape(-1),
                    flat_pred[indices].reshape(-1)
                ))
            except:
                continue

    if len(stats_list) == 0:
        return [np.nan, np.nan]
    return np.percentile(stats_list, [2.5, 97.5])


def calculate_segmentation_confidence_intervals(y_true, y_pred, n_bootstrap=1000):
    """Calculate confidence intervals for all segmentation metrics."""
    # Only bootstrap metrics that make sense
    metric_functions = {
        "Dice Score": dice_score,
        "IoU (Jaccard)": iou_score,
        "Pixel Accuracy": pixel_accuracy,
        "Sensitivity (Recall)": sensitivity,
        "Specificity": specificity,
        "Precision (PPV)": precision_seg,
        "Volumetric Similarity": volumetric_similarity
    }

    return {
        name: bootstrap_segmentation_metric(func, y_true, y_pred, n_boot=n_bootstrap)
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


def plot_segmentation_comparison(y_true, y_pred, colors, dpi, plots_dir, slice_idx=None):
    """Generate side-by-side comparison of ground truth and prediction."""
    # Handle 3D volumes by selecting a slice
    if y_true.ndim == 3:
        if slice_idx is None:
            # Select slice with most positive pixels
            slice_sums = np.sum(y_true, axis=(1, 2))
            slice_idx = np.argmax(slice_sums)
        y_true_2d = y_true[slice_idx]
        y_pred_2d = y_pred[slice_idx]
        title_suffix = f" (Slice {slice_idx})"
    else:
        y_true_2d = y_true
        y_pred_2d = y_pred
        title_suffix = ""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=dpi)

    # Ground Truth
    axes[0].imshow(y_true_2d, cmap='gray')
    axes[0].set_title(f'Ground Truth{title_suffix}', fontsize=12)
    axes[0].axis('off')

    # Prediction
    axes[1].imshow(y_pred_2d, cmap='gray')
    axes[1].set_title(f'Prediction{title_suffix}', fontsize=12)
    axes[1].axis('off')

    # Overlay
    overlay = np.zeros((*y_true_2d.shape, 3))
    y_true_bool = y_true_2d.astype(bool)
    y_pred_bool = y_pred_2d.astype(bool)

    # True Positive (green)
    tp = y_true_bool & y_pred_bool
    overlay[tp] = [0.18, 0.8, 0.44]  # Green

    # False Positive (red)
    fp = ~y_true_bool & y_pred_bool
    overlay[fp] = [0.91, 0.3, 0.24]  # Red

    # False Negative (blue)
    fn = y_true_bool & ~y_pred_bool
    overlay[fn] = [0.2, 0.6, 0.86]  # Blue

    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay (TP=Green, FP=Red, FN=Blue){title_suffix}', fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "segmentation_comparison.png"), dpi=dpi, bbox_inches='tight')
    return fig


def plot_confusion_matrix_segmentation(y_true, y_pred, colors, dpi, plots_dir):
    """Generate pixel-wise confusion matrix."""
    y_true_flat = y_true.flatten().astype(bool)
    y_pred_flat = y_pred.flatten().astype(bool)

    tp = np.sum(y_true_flat & y_pred_flat)
    tn = np.sum(~y_true_flat & ~y_pred_flat)
    fp = np.sum(~y_true_flat & y_pred_flat)
    fn = np.sum(y_true_flat & ~y_pred_flat)

    cm = np.array([[tn, fp], [fn, tp]])

    plt.figure(figsize=(6, 5), dpi=dpi)
    sns.heatmap(cm, annot=True, fmt='d', cmap=colors['cmap'], cbar=False,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'],
                annot_kws={"size": 12})
    plt.title('Pixel-wise Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)

    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=dpi, bbox_inches='tight')
    return plt.gcf()


def plot_dice_distribution(dice_scores_per_slice, colors, dpi, plots_dir):
    """Generate distribution of Dice scores across slices (for 3D data)."""
    plt.figure(figsize=(10, 5), dpi=dpi)

    plt.subplot(1, 2, 1)
    plt.hist(dice_scores_per_slice, bins=20, color=colors['ground_truth'], alpha=0.7, edgecolor='black')
    plt.xlabel('Dice Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Dice Scores per Slice', fontsize=14)
    plt.axvline(x=np.mean(dice_scores_per_slice), color='red', linestyle='--',
                label=f'Mean: {np.mean(dice_scores_per_slice):.3f}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.boxplot(dice_scores_per_slice, orientation='vertical')
    plt.ylabel('Dice Score', fontsize=12)
    plt.title('Dice Score Box Plot', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "dice_distribution.png"), dpi=dpi, bbox_inches='tight')
    return plt.gcf()


def plot_metrics_summary_segmentation(metrics_summary, conf_intervals, dpi, plots_dir):
    """Generate metrics summary table plot for segmentation."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    ax.axis("off")

    table_data = []
    for k, v in metrics_summary.items():
        ci = conf_intervals.get(k, [np.nan, np.nan])
        if np.isnan(v):
            table_data.append([k, "N/A", "N/A"])
        else:
            # Format distance metrics differently
            if "Distance" in k:
                table_data.append([k, f"{v:.2f}", "N/A"])
            else:
                ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if not np.isnan(ci[0]) else "N/A"
                table_data.append([k, f"{v:.4f}", ci_str])

    table = ax.table(cellText=table_data, colLabels=["Metric", "Value", "95% CI"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for j in range(3):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax.set_title("Segmentation Performance Metrics", fontweight="bold", fontsize=14, pad=20)

    plt.savefig(os.path.join(plots_dir, "metrics_summary.png"), dpi=dpi, bbox_inches='tight')
    return fig


def plot_surface_distance_histogram(y_true, y_pred, colors, dpi, plots_dir):
    """Generate histogram of surface distances."""
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    surface_true = get_surface_points(y_true)
    surface_pred = get_surface_points(y_pred)

    if len(surface_true) == 0 or len(surface_pred) == 0:
        fig = plt.figure(figsize=(10, 5), dpi=dpi)
        plt.text(0.5, 0.5, "No surface points found", ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.savefig(os.path.join(plots_dir, "surface_distance.png"), dpi=dpi, bbox_inches='tight')
        return fig

    from scipy.spatial.distance import cdist
    distances_true_to_pred = cdist(surface_true, surface_pred).min(axis=1)
    distances_pred_to_true = cdist(surface_pred, surface_true).min(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)

    axes[0].hist(distances_true_to_pred, bins=30, color=colors['ground_truth'], alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Distance', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Surface Distance: Ground Truth to Prediction', fontsize=12)
    axes[0].axvline(x=np.mean(distances_true_to_pred), color='red', linestyle='--',
                    label=f'Mean: {np.mean(distances_true_to_pred):.2f}')
    axes[0].legend()

    axes[1].hist(distances_pred_to_true, bins=30, color=colors['prediction'], alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Distance', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Surface Distance: Prediction to Ground Truth', fontsize=12)
    axes[1].axvline(x=np.mean(distances_pred_to_true), color='red', linestyle='--',
                    label=f'Mean: {np.mean(distances_pred_to_true):.2f}')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "surface_distance.png"), dpi=dpi, bbox_inches='tight')
    return fig


def plot_metrics_bar_chart(metrics_summary, colors, dpi, plots_dir):
    """Generate bar chart of key metrics."""
    # Select metrics suitable for bar chart (0-1 range)
    bar_metrics = {k: v for k, v in metrics_summary.items()
                   if not np.isnan(v) and "Distance" not in k}

    plt.figure(figsize=(10, 6), dpi=dpi)

    names = list(bar_metrics.keys())
    values = list(bar_metrics.values())

    bars = plt.bar(names, values, color=colors['metrics_colors'][:len(names)], edgecolor='black')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Segmentation Metrics Summary', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.15)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "metrics_bar_chart.png"), dpi=dpi, bbox_inches='tight')
    return plt.gcf()
