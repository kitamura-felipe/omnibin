import gradio as gr
import pandas as pd
import os
import shutil
from omnibin import generate_binary_classification_report, ColorScheme

# Define results directory
RESULTS_DIR = "/tmp/results"

# Map string color schemes to enum values
COLOR_SCHEME_MAP = {
    "DEFAULT": ColorScheme.DEFAULT,
    "MONOCHROME": ColorScheme.MONOCHROME,
    "VIBRANT": ColorScheme.VIBRANT
}

def process_csv(csv_file, n_bootstrap=1000, dpi=72, color_scheme="DEFAULT"):
    # Convert string color scheme to enum
    color_scheme_enum = COLOR_SCHEME_MAP[color_scheme]
    
    # Read the CSV file
    df = pd.read_csv(csv_file.name)
    
    # Check if required columns exist
    required_columns = ['y_true', 'y_pred']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV file must contain 'y_true' and 'y_pred' columns")
    
    # Clean up results directory if it exists
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    
    # Create fresh results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Generate the report
    report_path = generate_binary_classification_report(
        y_true=df['y_true'].values,
        y_scores=df['y_pred'].values,
        output_path=os.path.join(RESULTS_DIR, "classification_report.pdf"),
        n_bootstrap=n_bootstrap,
        random_seed=42,
        dpi=dpi,
        color_scheme=color_scheme_enum
    )
    
    # Get paths to individual plots
    plots_dir = os.path.join(RESULTS_DIR, "plots")
    plot_paths = {
        "ROC and PR Curves": os.path.join(plots_dir, "roc_pr.png"),
        "Metrics vs Threshold": os.path.join(plots_dir, "metrics_threshold.png"),
        "Confusion Matrix": os.path.join(plots_dir, "confusion_matrix.png"),
        "Calibration Plot": os.path.join(plots_dir, "calibration.png"),
        "Prediction Distribution": os.path.join(plots_dir, "prediction_distribution.png"),
        "Metrics Summary": os.path.join(plots_dir, "metrics_summary.png")
    }
    
    # Return both the PDF and the plot images
    return report_path, *plot_paths.values()

# Create the Gradio interface
iface = gr.Interface(
    fn=process_csv,
    inputs=[
        gr.File(label="Upload CSV file with 'y_true' and 'y_pred' columns"),
        gr.Number(label="Number of Bootstrap Iterations", value=1000, minimum=100, maximum=10000),
        gr.Number(label="DPI", value=72, minimum=50, maximum=300),
        gr.Dropdown(label="Color Scheme", choices=["DEFAULT", "MONOCHROME", "VIBRANT"], value="DEFAULT")
    ],
    outputs=[
        gr.File(label="Classification Report PDF"),
        gr.Image(label="ROC and PR Curves"),
        gr.Image(label="Metrics vs Threshold"),
        gr.Image(label="Confusion Matrix"),
        gr.Image(label="Calibration Plot"),
        gr.Image(label="Prediction Distribution"),
        gr.Image(label="Metrics Summary")
    ],
    title="Binary Classification Report Generator",
    description="Upload a CSV file containing 'y_true' and 'y_pred' columns to generate a binary classification report.\n\n"
                "'y_true': reference standard (0s or 1s).\n\n"
                "'y_pred': model prediction (continuous value between 0 and 1).\n\n"
                "This application takes approximately 35 seconds to generate the report.\n",

    examples=[["scores.csv", 1000, 72, "DEFAULT"]],
    cache_examples=False
)

if __name__ == "__main__":
    iface.launch()
