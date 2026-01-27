import gradio as gr
import pandas as pd
import numpy as np
import os
import shutil
import json
from omnibin import (
    generate_binary_classification_report, ColorScheme,
    generate_regression_report, RegressionColorScheme,
    generate_segmentation_report, SegmentationColorScheme,
    generate_detection_report, DetectionColorScheme
)

# Define directories
RESULTS_DIR = "/tmp/results"
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "data", "examples")

# Map string color schemes to enum values
CLASSIFICATION_COLOR_SCHEME_MAP = {
    "DEFAULT": ColorScheme.DEFAULT,
    "MONOCHROME": ColorScheme.MONOCHROME,
    "VIBRANT": ColorScheme.VIBRANT
}

REGRESSION_COLOR_SCHEME_MAP = {
    "DEFAULT": RegressionColorScheme.DEFAULT,
    "MONOCHROME": RegressionColorScheme.MONOCHROME,
    "VIBRANT": RegressionColorScheme.VIBRANT
}

SEGMENTATION_COLOR_SCHEME_MAP = {
    "DEFAULT": SegmentationColorScheme.DEFAULT,
    "MONOCHROME": SegmentationColorScheme.MONOCHROME,
    "VIBRANT": SegmentationColorScheme.VIBRANT
}

DETECTION_COLOR_SCHEME_MAP = {
    "DEFAULT": DetectionColorScheme.DEFAULT,
    "MONOCHROME": DetectionColorScheme.MONOCHROME,
    "VIBRANT": DetectionColorScheme.VIBRANT
}


def clean_results_dir():
    """Clean up results directory"""
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def process_classification_csv(csv_file, n_bootstrap=1000, dpi=72, color_scheme="DEFAULT"):
    """Process binary classification data"""
    color_scheme_enum = CLASSIFICATION_COLOR_SCHEME_MAP[color_scheme]

    df = pd.read_csv(csv_file.name)

    required_columns = ['y_true', 'y_pred']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV file must contain 'y_true' and 'y_pred' columns")

    clean_results_dir()

    report_path = generate_binary_classification_report(
        y_true=df['y_true'].values,
        y_scores=df['y_pred'].values,
        output_path=os.path.join(RESULTS_DIR, "classification_report.pdf"),
        n_bootstrap=n_bootstrap,
        random_seed=42,
        dpi=dpi,
        color_scheme=color_scheme_enum
    )

    plots_dir = os.path.join(RESULTS_DIR, "plots")
    plot_paths = {
        "ROC and PR Curves": os.path.join(plots_dir, "roc_pr.png"),
        "Metrics vs Threshold": os.path.join(plots_dir, "metrics_threshold.png"),
        "Confusion Matrix": os.path.join(plots_dir, "confusion_matrix.png"),
        "Calibration Plot": os.path.join(plots_dir, "calibration.png"),
        "Prediction Distribution": os.path.join(plots_dir, "prediction_distribution.png"),
        "Metrics Summary": os.path.join(plots_dir, "metrics_summary.png")
    }

    return report_path, *plot_paths.values()


def process_regression_csv(csv_file, n_bootstrap=1000, dpi=72, color_scheme="DEFAULT"):
    """Process regression data"""
    color_scheme_enum = REGRESSION_COLOR_SCHEME_MAP[color_scheme]

    df = pd.read_csv(csv_file.name)

    required_columns = ['y_true', 'y_pred']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV file must contain 'y_true' and 'y_pred' columns")

    clean_results_dir()

    report_path = generate_regression_report(
        y_true=df['y_true'].values,
        y_pred=df['y_pred'].values,
        output_path=os.path.join(RESULTS_DIR, "regression_report.pdf"),
        n_bootstrap=n_bootstrap,
        random_seed=42,
        dpi=dpi,
        color_scheme=color_scheme_enum
    )

    plots_dir = os.path.join(RESULTS_DIR, "plots")
    plot_paths = [
        os.path.join(plots_dir, "scatter_regression.png"),
        os.path.join(plots_dir, "residuals.png"),
        os.path.join(plots_dir, "qq_plot.png"),
        os.path.join(plots_dir, "bland_altman.png"),
        os.path.join(plots_dir, "error_distribution.png"),
        os.path.join(plots_dir, "metrics_summary.png")
    ]

    return report_path, *plot_paths


def process_segmentation_files(gt_file, pred_file, n_bootstrap=500, dpi=72, color_scheme="DEFAULT"):
    """Process segmentation data from numpy files"""
    color_scheme_enum = SEGMENTATION_COLOR_SCHEME_MAP[color_scheme]

    y_true = np.load(gt_file.name)
    y_pred = np.load(pred_file.name)

    clean_results_dir()

    report_path = generate_segmentation_report(
        y_true=y_true,
        y_pred=y_pred,
        output_path=os.path.join(RESULTS_DIR, "segmentation_report.pdf"),
        n_bootstrap=n_bootstrap,
        random_seed=42,
        dpi=dpi,
        color_scheme=color_scheme_enum
    )

    plots_dir = os.path.join(RESULTS_DIR, "plots")
    plot_paths = [
        os.path.join(plots_dir, "segmentation_comparison.png"),
        os.path.join(plots_dir, "confusion_matrix.png"),
        os.path.join(plots_dir, "metrics_bar_chart.png"),
        os.path.join(plots_dir, "surface_distance.png"),
        os.path.join(plots_dir, "metrics_summary.png")
    ]

    return report_path, *plot_paths


def process_detection_json(json_file, n_bootstrap=500, dpi=72, color_scheme="DEFAULT"):
    """Process detection data from JSON file"""
    color_scheme_enum = DETECTION_COLOR_SCHEME_MAP[color_scheme]

    with open(json_file.name, 'r') as f:
        data = json.load(f)

    predictions = data.get('predictions', [])
    ground_truths = data.get('ground_truths', [])

    clean_results_dir()

    report_path = generate_detection_report(
        predictions=predictions,
        ground_truths=ground_truths,
        output_path=os.path.join(RESULTS_DIR, "detection_report.pdf"),
        n_bootstrap=n_bootstrap,
        random_seed=42,
        dpi=dpi,
        color_scheme=color_scheme_enum
    )

    plots_dir = os.path.join(RESULTS_DIR, "plots")
    plot_paths = [
        os.path.join(plots_dir, "precision_recall.png"),
        os.path.join(plots_dir, "froc_curve.png"),
        os.path.join(plots_dir, "iou_distribution.png"),
        os.path.join(plots_dir, "confidence_distribution.png"),
        os.path.join(plots_dir, "metrics_summary.png")
    ]

    return report_path, *plot_paths


# Create tabs for different report types
with gr.Blocks(title="Omnibin - ML Metrics Report Generator") as app:
    gr.Markdown("# Omnibin - Comprehensive ML Metrics Report Generator")
    gr.Markdown("Generate detailed evaluation reports for classification, regression, segmentation, and detection tasks.")

    with gr.Tabs():
        # Binary Classification Tab
        with gr.TabItem("Binary Classification"):
            gr.Markdown("### Binary Classification Report")
            gr.Markdown("Upload a CSV with 'y_true' (0/1) and 'y_pred' (0-1 probability) columns.")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Download example file to try:**")
                    gr.File(
                        value=os.path.join(EXAMPLES_DIR, "classification_example.csv"),
                        label="Example: classification_example.csv",
                        interactive=False
                    )
                    gr.Markdown("---")
                    class_csv = gr.File(label="Upload CSV")
                    class_bootstrap = gr.Number(label="Bootstrap Iterations", value=1000, minimum=100, maximum=10000)
                    class_dpi = gr.Number(label="DPI", value=72, minimum=50, maximum=300)
                    class_color = gr.Dropdown(label="Color Scheme", choices=["DEFAULT", "MONOCHROME", "VIBRANT"], value="DEFAULT")
                    class_btn = gr.Button("Generate Report", variant="primary")

                with gr.Column():
                    class_pdf = gr.File(label="Report PDF")
                    class_roc = gr.Image(label="ROC/PR Curves")
                    class_thresh = gr.Image(label="Metrics vs Threshold")
                    class_cm = gr.Image(label="Confusion Matrix")
                    class_cal = gr.Image(label="Calibration")
                    class_dist = gr.Image(label="Prediction Distribution")
                    class_summary = gr.Image(label="Metrics Summary")

            class_btn.click(
                process_classification_csv,
                inputs=[class_csv, class_bootstrap, class_dpi, class_color],
                outputs=[class_pdf, class_roc, class_thresh, class_cm, class_cal, class_dist, class_summary]
            )

        # Regression Tab
        with gr.TabItem("Regression"):
            gr.Markdown("### Regression Report")
            gr.Markdown("Upload a CSV with 'y_true' and 'y_pred' columns (continuous values).")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Download example file to try:**")
                    gr.File(
                        value=os.path.join(EXAMPLES_DIR, "regression_example.csv"),
                        label="Example: regression_example.csv",
                        interactive=False
                    )
                    gr.Markdown("---")
                    reg_csv = gr.File(label="Upload CSV")
                    reg_bootstrap = gr.Number(label="Bootstrap Iterations", value=1000, minimum=100, maximum=10000)
                    reg_dpi = gr.Number(label="DPI", value=72, minimum=50, maximum=300)
                    reg_color = gr.Dropdown(label="Color Scheme", choices=["DEFAULT", "MONOCHROME", "VIBRANT"], value="DEFAULT")
                    reg_btn = gr.Button("Generate Report", variant="primary")

                with gr.Column():
                    reg_pdf = gr.File(label="Report PDF")
                    reg_scatter = gr.Image(label="Scatter Plot")
                    reg_residuals = gr.Image(label="Residuals")
                    reg_qq = gr.Image(label="Q-Q Plot")
                    reg_bland = gr.Image(label="Bland-Altman")
                    reg_error = gr.Image(label="Error Distribution")
                    reg_summary = gr.Image(label="Metrics Summary")

            reg_btn.click(
                process_regression_csv,
                inputs=[reg_csv, reg_bootstrap, reg_dpi, reg_color],
                outputs=[reg_pdf, reg_scatter, reg_residuals, reg_qq, reg_bland, reg_error, reg_summary]
            )

        # Segmentation Tab
        with gr.TabItem("Segmentation"):
            gr.Markdown("### Segmentation Report")
            gr.Markdown("Upload two NumPy (.npy) files: ground truth mask and prediction mask.")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Download example files to try:**")
                    with gr.Row():
                        gr.File(
                            value=os.path.join(EXAMPLES_DIR, "segmentation_2d_ground_truth.npy"),
                            label="Example: Ground Truth (2D)",
                            interactive=False
                        )
                        gr.File(
                            value=os.path.join(EXAMPLES_DIR, "segmentation_2d_prediction.npy"),
                            label="Example: Prediction (2D)",
                            interactive=False
                        )
                    gr.Markdown("---")
                    seg_gt = gr.File(label="Ground Truth Mask (.npy)")
                    seg_pred = gr.File(label="Prediction Mask (.npy)")
                    seg_bootstrap = gr.Number(label="Bootstrap Iterations", value=500, minimum=50, maximum=5000)
                    seg_dpi = gr.Number(label="DPI", value=72, minimum=50, maximum=300)
                    seg_color = gr.Dropdown(label="Color Scheme", choices=["DEFAULT", "MONOCHROME", "VIBRANT"], value="DEFAULT")
                    seg_btn = gr.Button("Generate Report", variant="primary")

                with gr.Column():
                    seg_pdf = gr.File(label="Report PDF")
                    seg_compare = gr.Image(label="Segmentation Comparison")
                    seg_cm = gr.Image(label="Confusion Matrix")
                    seg_bar = gr.Image(label="Metrics Bar Chart")
                    seg_surface = gr.Image(label="Surface Distance")
                    seg_summary = gr.Image(label="Metrics Summary")

            seg_btn.click(
                process_segmentation_files,
                inputs=[seg_gt, seg_pred, seg_bootstrap, seg_dpi, seg_color],
                outputs=[seg_pdf, seg_compare, seg_cm, seg_bar, seg_surface, seg_summary]
            )

        # Detection Tab
        with gr.TabItem("Detection"):
            gr.Markdown("### Object Detection Report")
            gr.Markdown("""
            Upload a JSON file with the following structure:
            ```json
            {
                "predictions": [
                    [{"box": [x1,y1,x2,y2], "score": 0.9}, ...],
                    ...
                ],
                "ground_truths": [
                    [[x1,y1,x2,y2], ...],
                    ...
                ]
            }
            ```
            """)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Download example file to try:**")
                    gr.File(
                        value=os.path.join(EXAMPLES_DIR, "detection_example.json"),
                        label="Example: detection_example.json",
                        interactive=False
                    )
                    gr.Markdown("---")
                    det_json = gr.File(label="Upload JSON")
                    det_bootstrap = gr.Number(label="Bootstrap Iterations", value=500, minimum=50, maximum=5000)
                    det_dpi = gr.Number(label="DPI", value=72, minimum=50, maximum=300)
                    det_color = gr.Dropdown(label="Color Scheme", choices=["DEFAULT", "MONOCHROME", "VIBRANT"], value="DEFAULT")
                    det_btn = gr.Button("Generate Report", variant="primary")

                with gr.Column():
                    det_pdf = gr.File(label="Report PDF")
                    det_pr = gr.Image(label="Precision-Recall Curve")
                    det_froc = gr.Image(label="FROC Curve")
                    det_iou = gr.Image(label="IoU Distribution")
                    det_conf = gr.Image(label="Confidence Distribution")
                    det_summary = gr.Image(label="Metrics Summary")

            det_btn.click(
                process_detection_json,
                inputs=[det_json, det_bootstrap, det_dpi, det_color],
                outputs=[det_pdf, det_pr, det_froc, det_iou, det_conf, det_summary]
            )

    gr.Markdown("---")
    gr.Markdown("**Omnibin v0.2.0** - Comprehensive ML evaluation metrics with healthcare focus")

if __name__ == "__main__":
    app.launch()
