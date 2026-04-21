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
    generate_detection_report, DetectionColorScheme,
    generate_text_generation_report, TextGenColorScheme,
    LLMConfig, SUPPORTED_PROVIDERS,
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

TEXT_GEN_COLOR_SCHEME_MAP = {
    "DEFAULT": TextGenColorScheme.DEFAULT,
    "MONOCHROME": TextGenColorScheme.MONOCHROME,
    "VIBRANT": TextGenColorScheme.VIBRANT
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


def process_text_generation_csv(
    csv_file,
    metrics_selection,
    provider,
    model_name,
    api_key,
    n_bootstrap=500,
    dpi=72,
    color_scheme="DEFAULT",
):
    """Process text generation (radiology reports) data."""
    color_scheme_enum = TEXT_GEN_COLOR_SCHEME_MAP[color_scheme]

    df = pd.read_csv(csv_file.name)
    required_columns = ["reference", "candidate"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV file must contain 'reference' and 'candidate' columns")

    # Normalize metric selection (Gradio returns list of strings).
    selected = [m.lower() for m in (metrics_selection or [])]

    # Build LLM config only if a judge metric needing one is selected.
    llm_config = None
    judge_needing_llm = {"green", "radfact", "crimson"}
    if selected and any(m in judge_needing_llm for m in selected):
        if not provider:
            raise ValueError(
                "Provider must be selected when using GREEN / RadFact / CRIMSON."
            )
        if not api_key:
            raise ValueError(
                f"API key is required for provider '{provider}'."
            )
        llm_config = LLMConfig(
            provider=provider,
            model=model_name or None,
            api_key=api_key,
        )

    clean_results_dir()

    report = generate_text_generation_report(
        references=df["reference"].astype(str).tolist(),
        candidates=df["candidate"].astype(str).tolist(),
        output_path=os.path.join(RESULTS_DIR, "text_generation_report.pdf"),
        metrics=selected,
        llm_config=llm_config,
        n_bootstrap=n_bootstrap,
        random_seed=42,
        dpi=dpi,
        color_scheme=color_scheme_enum,
    )

    plots_dir = os.path.join(RESULTS_DIR, "plots")
    summary_plot = os.path.join(plots_dir, "text_gen_metrics_summary.png")
    distribution_plot = os.path.join(plots_dir, "text_gen_per_sample_distribution.png")
    correlation_plot = os.path.join(plots_dir, "text_gen_metric_correlation.png")
    heatmap_plot = os.path.join(plots_dir, "text_gen_per_sample_heatmap.png")

    # Format score tables for display.
    agg_rows = [[k, f"{v:.4f}"] for k, v in report.aggregate_scores.items()]
    sub_rows = [[k, f"{v:.4f}"] for k, v in report.submetrics.items()]
    skip_text = ""
    if report.metrics_skipped:
        lines = [f"- **{k}**: {v}" for k, v in report.metrics_skipped.items()]
        skip_text = "### Skipped metrics\n" + "\n".join(lines)

    return (
        report.output_path,
        summary_plot if os.path.exists(summary_plot) else None,
        distribution_plot if os.path.exists(distribution_plot) else None,
        correlation_plot if os.path.exists(correlation_plot) else None,
        heatmap_plot if os.path.exists(heatmap_plot) else None,
        agg_rows,
        sub_rows,
        skip_text,
    )


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

        # Text Generation Tab (radiology reports)
        with gr.TabItem("Text Generation"):
            gr.Markdown("### Text Generation Report (Radiology Reports)")
            gr.Markdown(
                "Upload a CSV with `reference` and `candidate` columns. "
                "Lexical metrics (BLEU / ROUGE / METEOR / BERTScore) run locally. "
                "LLM-judge metrics (GREEN, RadFact, CRIMSON) require an API key.\n\n"
                "**Paper-default judge models:** GREEN → `StanfordAIMI/GREEN-RadLlama2-7b` "
                "(Ostmeier et al., 2024); RadFact → Llama-3-70B-Instruct "
                "(Bannur et al., 2024); CRIMSON → `MedGemmaCRIMSON-4B` "
                "(Baharoon et al., 2026)."
            )
            gr.Markdown(
                "> ⚠️ **DISCLAIMER — API-mode judges are not paper defaults.**  \n"
                "> The hosted Space runs GREEN and RadFact in **API mode**: we use "
                "each paper's verbatim prompt (GREEN) or system messages (RadFact) "
                "but route them to a generic provider you select (OpenAI / Anthropic "
                "/ Google / OpenRouter / Groq). **Scores can differ from published "
                "numbers**, because the original papers fine-tuned their own judge "
                "models.  \n"
                "> - **GREEN API mode**: verbatim prompt + parser, only the judge "
                "model changes.  \n"
                "> - **RadFact API mode**: verbatim two-stage system prompts but "
                "with zero-shot JSON output (the upstream `radfact` package teaches "
                "output format with 10-shot YAML). Only logical precision/recall/F1; "
                "no grounding or spatial scores.  \n"
                "> For exact paper reproduction: `pip install omnibin[green]` "
                "(needs GPU) or `pip install omnibin[radfact]` (needs a separate "
                "env due to pydantic 1.x / Gradio conflict)."
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Download example file to try:**")
                    gr.File(
                        value=os.path.join(EXAMPLES_DIR, "text_generation_example.csv"),
                        label="Example: text_generation_example.csv (10 fabricated CXR pairs)",
                        interactive=False
                    )
                    gr.Markdown("---")
                    tg_csv = gr.File(label="Upload CSV")
                    tg_metrics = gr.CheckboxGroup(
                        label="Metrics",
                        choices=[
                            "bleu", "rouge", "meteor", "bertscore",
                            "green", "radfact", "crimson",
                        ],
                        value=["bleu", "rouge", "meteor", "bertscore"],
                        info=(
                            "Pick which metrics to compute. GREEN / RadFact / CRIMSON "
                            "require an API key — they run in API mode with your "
                            "chosen provider (see disclaimer above)."
                        ),
                    )
                    tg_provider = gr.Dropdown(
                        label="LLM Provider (for GREEN / RadFact / CRIMSON)",
                        choices=list(SUPPORTED_PROVIDERS),
                        value="openai",
                    )
                    tg_model = gr.Textbox(
                        label="Model Name (optional — defaults per provider)",
                        placeholder="e.g. gpt-4o, claude-sonnet-4-6, gemini-2.5-pro",
                    )
                    tg_api_key = gr.Textbox(
                        label="API Key",
                        type="password",
                        placeholder="sk-...",
                    )
                    tg_bootstrap = gr.Number(label="Bootstrap Iterations", value=500, minimum=50, maximum=5000)
                    tg_dpi = gr.Number(label="DPI", value=72, minimum=50, maximum=300)
                    tg_color = gr.Dropdown(label="Color Scheme", choices=["DEFAULT", "MONOCHROME", "VIBRANT"], value="DEFAULT")
                    tg_btn = gr.Button("Generate Report", variant="primary")

                with gr.Column():
                    tg_pdf = gr.File(label="Report PDF")
                    tg_summary = gr.Image(label="Metrics Summary")
                    tg_dist = gr.Image(label="Per-sample Distribution")
                    tg_corr = gr.Image(label="Metric Correlation")
                    tg_heatmap = gr.Image(label="Per-sample Heatmap")
                    tg_agg_table = gr.Dataframe(
                        headers=["Metric", "Aggregate"],
                        label="Aggregate Scores",
                        interactive=False,
                    )
                    tg_sub_table = gr.Dataframe(
                        headers=["Submetric", "Value"],
                        label="Submetric Breakdown",
                        interactive=False,
                    )
                    tg_skipped = gr.Markdown()

            tg_btn.click(
                process_text_generation_csv,
                inputs=[
                    tg_csv, tg_metrics, tg_provider, tg_model, tg_api_key,
                    tg_bootstrap, tg_dpi, tg_color,
                ],
                outputs=[
                    tg_pdf, tg_summary, tg_dist, tg_corr, tg_heatmap,
                    tg_agg_table, tg_sub_table, tg_skipped,
                ],
            )

    gr.Markdown("---")
    gr.Markdown("**Omnibin v0.3.0** - Comprehensive ML evaluation metrics with healthcare focus")

if __name__ == "__main__":
    app.launch()
