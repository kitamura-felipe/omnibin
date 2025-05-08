import pandas as pd
import os
from omnibin import generate_binary_classification_report

# Define paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load your data
data = pd.read_csv(os.path.join(DATA_DIR, "scores.csv"))
y_true = data['y_true'].values
y_scores = data['y_pred'].values

# Generate comprehensive classification report
report_path = generate_binary_classification_report(
    y_true=y_true,
    y_scores=y_scores,
    output_path=os.path.join(RESULTS_DIR, "classification_report.pdf"),
    n_bootstrap=1000,
    random_seed=42,  # Set a fixed random seed for reproducibility
    dpi=300
)

print(f"Report generated and saved to: {report_path}")