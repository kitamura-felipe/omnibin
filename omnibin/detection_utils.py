import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from enum import Enum
import os


class DetectionColorScheme(Enum):
    DEFAULT = {
        'precision_recall': 'tab:blue',
        'froc': 'tab:red',
        'confidence': 'tab:green',
        'true_positive': '#2ecc71',
        'false_positive': '#e74c3c',
        'false_negative': '#3498db',
        'reference_line': 'gray',
        'metrics_colors': ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange', 'tab:brown'],
        'cmap': 'Blues'
    }

    MONOCHROME = {
        'precision_recall': '#000000',
        'froc': '#404040',
        'confidence': '#606060',
        'true_positive': '#404040',
        'false_positive': '#808080',
        'false_negative': '#606060',
        'reference_line': '#808080',
        'metrics_colors': ['#000000', '#404040', '#606060', '#808080', '#A0A0A0', '#C0C0C0'],
        'cmap': 'Greys'
    }

    VIBRANT = {
        'precision_recall': '#4ECDC4',
        'froc': '#FF6B6B',
        'confidence': '#45B7D1',
        'true_positive': '#2ecc71',
        'false_positive': '#e74c3c',
        'false_negative': '#3498db',
        'reference_line': '#95A5A6',
        'metrics_colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5'],
        'cmap': 'Greens'
    }


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two bounding boxes.

    Parameters
    ----------
    box1 : array-like
        First bounding box [x1, y1, x2, y2]
    box2 : array-like
        Second bounding box [x1, y1, x2, y2]

    Returns
    -------
    float
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


def match_detections_to_ground_truth(predictions, ground_truths, iou_threshold=0.5, use_distance=False,
                                     distance_threshold=None):
    """
    Match predicted detections to ground truth annotations.

    Parameters
    ----------
    predictions : list
        List of predictions, each containing {'box': [x1,y1,x2,y2], 'score': float}
        or {'center': [x, y], 'score': float} for point-based detection
    ground_truths : list
        List of ground truth boxes [x1, y1, x2, y2] or points [x, y]
    iou_threshold : float, optional
        IoU threshold for matching (default: 0.5)
    use_distance : bool, optional
        If True, use distance-based matching instead of IoU (default: False)
    distance_threshold : float, optional
        Distance threshold for matching when use_distance=True

    Returns
    -------
    tuple
        (tp, fp, fn, matched_gt_indices, matched_scores)
    """
    if len(predictions) == 0:
        return 0, 0, len(ground_truths), [], []

    if len(ground_truths) == 0:
        return 0, len(predictions), 0, [], [p.get('score', 1.0) for p in predictions]

    # Sort predictions by confidence score (descending)
    sorted_preds = sorted(predictions, key=lambda x: x.get('score', 1.0), reverse=True)

    matched_gt = set()
    tp = 0
    fp = 0
    matched_scores = []

    for pred in sorted_preds:
        best_match = -1
        best_score = -1 if not use_distance else float('inf')

        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue

            if use_distance:
                pred_point = pred.get('center', pred.get('point'))
                gt_point = gt if len(gt) == 2 else [(gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2]
                dist = calculate_distance(pred_point, gt_point)

                if dist < best_score and (distance_threshold is None or dist <= distance_threshold):
                    best_score = dist
                    best_match = gt_idx
            else:
                pred_box = pred.get('box', pred.get('bbox'))
                iou = calculate_iou(pred_box, gt)

                if iou > best_score and iou >= iou_threshold:
                    best_score = iou
                    best_match = gt_idx

        if best_match >= 0:
            tp += 1
            matched_gt.add(best_match)
            matched_scores.append(pred.get('score', 1.0))
        else:
            fp += 1

    fn = len(ground_truths) - len(matched_gt)

    return tp, fp, fn, list(matched_gt), matched_scores


def calculate_precision_recall_curve(all_predictions, all_ground_truths, iou_threshold=0.5):
    """
    Calculate precision-recall curve across all images/samples.

    Parameters
    ----------
    all_predictions : list of lists
        List of predictions for each image
    all_ground_truths : list of lists
        List of ground truth annotations for each image
    iou_threshold : float, optional
        IoU threshold for matching (default: 0.5)

    Returns
    -------
    tuple
        (precisions, recalls, thresholds)
    """
    # Collect all predictions with their image index
    all_detections = []
    total_gt = 0

    for img_idx, (preds, gts) in enumerate(zip(all_predictions, all_ground_truths)):
        for pred in preds:
            all_detections.append({
                'img_idx': img_idx,
                'score': pred.get('score', 1.0),
                'box': pred.get('box', pred.get('bbox'))
            })
        total_gt += len(gts)

    if total_gt == 0 or len(all_detections) == 0:
        return np.array([1.0]), np.array([0.0]), np.array([1.0])

    # Sort by confidence
    all_detections.sort(key=lambda x: x['score'], reverse=True)

    # Track which GTs are matched per image
    matched_gt_per_image = {i: set() for i in range(len(all_predictions))}

    precisions = []
    recalls = []
    thresholds = []
    tp_cumsum = 0
    fp_cumsum = 0

    for det in all_detections:
        img_idx = det['img_idx']
        gts = all_ground_truths[img_idx]

        # Find best matching GT
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gts):
            if gt_idx in matched_gt_per_image[img_idx]:
                continue

            iou = calculate_iou(det['box'], gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp_cumsum += 1
            matched_gt_per_image[img_idx].add(best_gt_idx)
        else:
            fp_cumsum += 1

        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / total_gt

        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(det['score'])

    return np.array(precisions), np.array(recalls), np.array(thresholds)


def calculate_ap(precisions, recalls):
    """
    Calculate Average Precision using 11-point interpolation or all-point interpolation.

    Parameters
    ----------
    precisions : array-like
        Precision values
    recalls : array-like
        Recall values

    Returns
    -------
    float
        Average Precision
    """
    # Add sentinel values
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure monotonically decreasing precision
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Find points where recall changes
    recall_changes = np.where(recalls[1:] != recalls[:-1])[0]

    # Sum areas
    ap = np.sum((recalls[recall_changes + 1] - recalls[recall_changes]) * precisions[recall_changes + 1])

    return ap


def calculate_froc_curve(all_predictions, all_ground_truths, iou_threshold=0.5, use_distance=False,
                         distance_threshold=None):
    """
    Calculate Free-Response ROC (FROC) curve - commonly used in medical imaging.

    FROC plots sensitivity vs. average false positives per image.

    Parameters
    ----------
    all_predictions : list of lists
        List of predictions for each image
    all_ground_truths : list of lists
        List of ground truth annotations for each image
    iou_threshold : float, optional
        IoU threshold for matching (default: 0.5)
    use_distance : bool, optional
        If True, use distance-based matching (default: False)
    distance_threshold : float, optional
        Distance threshold for matching

    Returns
    -------
    tuple
        (sensitivities, fp_per_image, thresholds)
    """
    # Collect all predictions with scores
    all_scores = []
    for preds in all_predictions:
        for pred in preds:
            all_scores.append(pred.get('score', 1.0))

    if len(all_scores) == 0:
        return np.array([0.0]), np.array([0.0]), np.array([1.0])

    # Get unique thresholds
    thresholds = np.unique(all_scores)
    thresholds = np.sort(thresholds)[::-1]  # Descending

    n_images = len(all_predictions)
    total_gt = sum(len(gts) for gts in all_ground_truths)

    if total_gt == 0:
        return np.array([0.0]), np.array([0.0]), thresholds

    sensitivities = []
    fp_per_image = []

    for thresh in thresholds:
        total_tp = 0
        total_fp = 0

        for preds, gts in zip(all_predictions, all_ground_truths):
            # Filter predictions by threshold
            filtered_preds = [p for p in preds if p.get('score', 1.0) >= thresh]

            tp, fp, fn, _, _ = match_detections_to_ground_truth(
                filtered_preds, gts, iou_threshold, use_distance, distance_threshold
            )

            total_tp += tp
            total_fp += fp

        sensitivity = total_tp / total_gt if total_gt > 0 else 0
        avg_fp = total_fp / n_images if n_images > 0 else 0

        sensitivities.append(sensitivity)
        fp_per_image.append(avg_fp)

    return np.array(sensitivities), np.array(fp_per_image), thresholds


def calculate_froc_score(sensitivities, fp_per_image, fp_rates=[0.125, 0.25, 0.5, 1, 2, 4, 8]):
    """
    Calculate FROC score as average sensitivity at predefined FP rates.

    This is commonly used in medical imaging competitions (e.g., LUNA16).

    Parameters
    ----------
    sensitivities : array-like
        Sensitivity values from FROC curve
    fp_per_image : array-like
        False positives per image from FROC curve
    fp_rates : list, optional
        FP rates at which to measure sensitivity

    Returns
    -------
    float
        FROC score (average sensitivity at specified FP rates)
    """
    sensitivities_at_fps = []

    for fp_rate in fp_rates:
        # Find sensitivity at this FP rate (interpolate if needed)
        idx = np.searchsorted(fp_per_image, fp_rate)

        if idx == 0:
            sens = sensitivities[0]
        elif idx >= len(fp_per_image):
            sens = sensitivities[-1]
        else:
            # Linear interpolation
            x1, x2 = fp_per_image[idx - 1], fp_per_image[idx]
            y1, y2 = sensitivities[idx - 1], sensitivities[idx]

            if x2 - x1 > 0:
                sens = y1 + (y2 - y1) * (fp_rate - x1) / (x2 - x1)
            else:
                sens = y1

        sensitivities_at_fps.append(sens)

    return np.mean(sensitivities_at_fps)


def calculate_detection_metrics(all_predictions, all_ground_truths, iou_thresholds=[0.5, 0.75]):
    """
    Calculate comprehensive detection metrics.

    Parameters
    ----------
    all_predictions : list of lists
        List of predictions for each image
    all_ground_truths : list of lists
        List of ground truth annotations for each image
    iou_thresholds : list, optional
        IoU thresholds for calculating AP (default: [0.5, 0.75])

    Returns
    -------
    dict
        Dictionary containing all metrics
    """
    metrics = {}

    # Calculate metrics at each IoU threshold
    for iou_thresh in iou_thresholds:
        precisions, recalls, _ = calculate_precision_recall_curve(
            all_predictions, all_ground_truths, iou_thresh
        )
        ap = calculate_ap(precisions, recalls)
        metrics[f"AP@{int(iou_thresh * 100)}"] = ap

    # Calculate mAP (mean over IoU thresholds)
    ap_values = [metrics[f"AP@{int(t * 100)}"] for t in iou_thresholds]
    metrics["mAP"] = np.mean(ap_values)

    # Calculate FROC metrics
    sensitivities, fp_per_image, _ = calculate_froc_curve(all_predictions, all_ground_truths)
    metrics["FROC Score"] = calculate_froc_score(sensitivities, fp_per_image)

    # Calculate overall TP, FP, FN at IoU=0.5
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for preds, gts in zip(all_predictions, all_ground_truths):
        tp, fp, fn, _, _ = match_detections_to_ground_truth(preds, gts, iou_threshold=0.5)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Calculate detection-level metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics["Precision@50"] = precision
    metrics["Recall@50"] = recall
    metrics["F1@50"] = f1

    # Detection rate (percentage of images with at least one detection)
    images_with_detection = sum(1 for preds in all_predictions if len(preds) > 0)
    metrics["Detection Rate"] = images_with_detection / len(all_predictions) if len(all_predictions) > 0 else 0

    # Average FP per image
    total_fp_all = sum(len(preds) for preds in all_predictions) - total_tp
    metrics["Avg FP/Image"] = total_fp_all / len(all_predictions) if len(all_predictions) > 0 else 0

    return metrics


def bootstrap_detection_metric(metric_func, all_predictions, all_ground_truths, n_boot=1000):
    """Calculate bootstrap confidence intervals for a detection metric."""
    stats_list = []
    n_images = len(all_predictions)

    for _ in tqdm(range(n_boot), desc="Bootstrap iterations", leave=False):
        # Bootstrap over images
        indices = np.random.choice(n_images, n_images, replace=True)
        boot_preds = [all_predictions[i] for i in indices]
        boot_gts = [all_ground_truths[i] for i in indices]

        try:
            stats_list.append(metric_func(boot_preds, boot_gts))
        except:
            continue

    if len(stats_list) == 0:
        return [np.nan, np.nan]
    return np.percentile(stats_list, [2.5, 97.5])


def calculate_detection_confidence_intervals(all_predictions, all_ground_truths, n_bootstrap=500):
    """Calculate confidence intervals for detection metrics."""

    def get_ap50(preds, gts):
        prec, rec, _ = calculate_precision_recall_curve(preds, gts, 0.5)
        return calculate_ap(prec, rec)

    def get_ap75(preds, gts):
        prec, rec, _ = calculate_precision_recall_curve(preds, gts, 0.75)
        return calculate_ap(prec, rec)

    def get_froc_score(preds, gts):
        sens, fp, _ = calculate_froc_curve(preds, gts)
        return calculate_froc_score(sens, fp)

    metric_functions = {
        "AP@50": get_ap50,
        "AP@75": get_ap75,
        "FROC Score": get_froc_score
    }

    return {
        name: bootstrap_detection_metric(func, all_predictions, all_ground_truths, n_boot=n_bootstrap)
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


def plot_precision_recall_curve_detection(all_predictions, all_ground_truths, iou_thresholds, colors, dpi, plots_dir):
    """Generate precision-recall curves at different IoU thresholds."""
    plt.figure(figsize=(10, 8), dpi=dpi)

    for i, iou_thresh in enumerate(iou_thresholds):
        precisions, recalls, _ = calculate_precision_recall_curve(
            all_predictions, all_ground_truths, iou_thresh
        )
        ap = calculate_ap(precisions, recalls)

        color = colors['metrics_colors'][i % len(colors['metrics_colors'])]
        plt.plot(recalls, precisions, color=color, linewidth=2,
                 label=f'IoU={iou_thresh:.2f} (AP={ap:.3f})')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves at Different IoU Thresholds', fontsize=14)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1.02])
    plt.ylim([0, 1.02])

    plt.savefig(os.path.join(plots_dir, "precision_recall.png"), dpi=dpi, bbox_inches='tight')
    return plt.gcf()


def plot_froc_curve(all_predictions, all_ground_truths, colors, dpi, plots_dir):
    """Generate FROC curve (sensitivity vs FP per image)."""
    sensitivities, fp_per_image, _ = calculate_froc_curve(all_predictions, all_ground_truths)
    froc_score = calculate_froc_score(sensitivities, fp_per_image)

    plt.figure(figsize=(10, 8), dpi=dpi)

    plt.plot(fp_per_image, sensitivities, color=colors['froc'], linewidth=2,
             label=f'FROC (Score={froc_score:.3f})')

    # Mark standard FP rates
    fp_rates = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    for fp_rate in fp_rates:
        if fp_rate <= max(fp_per_image):
            plt.axvline(x=fp_rate, color='gray', linestyle=':', alpha=0.5)

    plt.xlabel('Average False Positives per Image', fontsize=12)
    plt.ylabel('Sensitivity (True Positive Rate)', fontsize=12)
    plt.title('Free-Response ROC (FROC) Curve', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, max(8, max(fp_per_image) * 1.1)])
    plt.ylim([0, 1.02])

    plt.savefig(os.path.join(plots_dir, "froc_curve.png"), dpi=dpi, bbox_inches='tight')
    return plt.gcf()


def plot_confidence_distribution(all_predictions, all_ground_truths, colors, dpi, plots_dir):
    """Generate histogram of prediction confidence scores."""
    # Separate TP and FP scores
    tp_scores = []
    fp_scores = []

    for preds, gts in zip(all_predictions, all_ground_truths):
        tp, fp, fn, matched_gt, matched_scores = match_detections_to_ground_truth(preds, gts, iou_threshold=0.5)

        matched_set = set(matched_gt)
        for pred in preds:
            score = pred.get('score', 1.0)
            # Check if this prediction was matched
            is_matched = False
            for gt_idx, gt in enumerate(gts):
                if gt_idx in matched_set:
                    pred_box = pred.get('box', pred.get('bbox'))
                    if calculate_iou(pred_box, gt) >= 0.5:
                        is_matched = True
                        matched_set.discard(gt_idx)
                        break

            if is_matched:
                tp_scores.append(score)
            else:
                fp_scores.append(score)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=dpi)

    # Combined histogram
    axes[0].hist(tp_scores, bins=20, alpha=0.7, label='True Positives', color=colors['true_positive'])
    axes[0].hist(fp_scores, bins=20, alpha=0.7, label='False Positives', color=colors['false_positive'])
    axes[0].set_xlabel('Confidence Score', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Confidence Score Distribution', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot comparison
    data = [tp_scores, fp_scores] if tp_scores and fp_scores else [[0], [0]]
    bp = axes[1].boxplot(data, tick_labels=['True Positives', 'False Positives'], patch_artist=True)
    bp['boxes'][0].set_facecolor(colors['true_positive'])
    bp['boxes'][1].set_facecolor(colors['false_positive'])
    axes[1].set_ylabel('Confidence Score', fontsize=12)
    axes[1].set_title('Confidence Score Comparison', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confidence_distribution.png"), dpi=dpi, bbox_inches='tight')
    return fig


def plot_detections_per_image(all_predictions, all_ground_truths, colors, dpi, plots_dir):
    """Generate histogram of detections per image."""
    n_preds = [len(preds) for preds in all_predictions]
    n_gts = [len(gts) for gts in all_ground_truths]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=dpi)

    # Predictions per image
    axes[0].hist(n_preds, bins=20, alpha=0.7, color=colors['precision_recall'], edgecolor='black')
    axes[0].axvline(x=np.mean(n_preds), color='red', linestyle='--',
                    label=f'Mean: {np.mean(n_preds):.1f}')
    axes[0].set_xlabel('Number of Predictions', fontsize=12)
    axes[0].set_ylabel('Number of Images', fontsize=12)
    axes[0].set_title('Predictions per Image', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # GT vs Predictions scatter
    axes[1].scatter(n_gts, n_preds, alpha=0.5, color=colors['confidence'])
    max_val = max(max(n_gts), max(n_preds)) + 1
    axes[1].plot([0, max_val], [0, max_val], '--', color=colors['reference_line'], label='y=x')
    axes[1].set_xlabel('Ground Truth Count', fontsize=12)
    axes[1].set_ylabel('Prediction Count', fontsize=12)
    axes[1].set_title('Ground Truth vs Predictions per Image', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "detections_per_image.png"), dpi=dpi, bbox_inches='tight')
    return fig


def plot_metrics_summary_detection(metrics_summary, conf_intervals, dpi, plots_dir):
    """Generate metrics summary table plot for detection."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    ax.axis("off")

    table_data = []
    for k, v in metrics_summary.items():
        ci = conf_intervals.get(k, [np.nan, np.nan])
        if np.isnan(v):
            table_data.append([k, "N/A", "N/A"])
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

    ax.set_title("Object Detection Performance Metrics", fontweight="bold", fontsize=14, pad=20)

    plt.savefig(os.path.join(plots_dir, "metrics_summary.png"), dpi=dpi, bbox_inches='tight')
    return fig


def plot_iou_distribution(all_predictions, all_ground_truths, colors, dpi, plots_dir):
    """Generate histogram of IoU values for matched detections."""
    ious = []

    for preds, gts in zip(all_predictions, all_ground_truths):
        if len(preds) == 0 or len(gts) == 0:
            continue

        for pred in preds:
            pred_box = pred.get('box', pred.get('bbox'))
            max_iou = 0
            for gt in gts:
                iou = calculate_iou(pred_box, gt)
                max_iou = max(max_iou, iou)
            ious.append(max_iou)

    if not ious:
        fig = plt.figure(figsize=(10, 6), dpi=dpi)
        plt.text(0.5, 0.5, "No IoU values to display", ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.savefig(os.path.join(plots_dir, "iou_distribution.png"), dpi=dpi, bbox_inches='tight')
        return fig

    plt.figure(figsize=(10, 6), dpi=dpi)

    plt.hist(ious, bins=20, color=colors['precision_recall'], alpha=0.7, edgecolor='black')
    plt.axvline(x=0.5, color='red', linestyle='--', label='IoU=0.5 threshold')
    plt.axvline(x=np.mean(ious), color='green', linestyle='-', label=f'Mean IoU: {np.mean(ious):.3f}')

    plt.xlabel('IoU', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of IoU Values (Prediction to Best GT Match)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])

    plt.savefig(os.path.join(plots_dir, "iou_distribution.png"), dpi=dpi, bbox_inches='tight')
    return plt.gcf()
