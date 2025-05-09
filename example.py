import pandas as pd
import numpy as np
import os
from omnibin import generate_binary_classification_report, ColorScheme

# Define paths
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Generate random data
data = pd.DataFrame({
    'y_true': (y:=np.random.choice([0,1],1000,p:=[.9,.1])),
    'y_pred': np.where(
        y,
        np.random.beta(3,1.5,1000)*.9+.1,  # Positive cases: less skewed towards 1.0
        np.random.beta(1.5,3,1000)*.9+.1   # Negative cases: less skewed towards 0.1
    )
})

y_true = data['y_true'].values
y_scores = data['y_pred'].values

# Generate comprehensive classification report
report_path = generate_binary_classification_report(
    y_true=y_true,
    y_scores=y_scores,
    output_path=os.path.join(RESULTS_DIR, "classification_report.pdf"),
    n_bootstrap=1000,
    random_seed=42,  # Set a fixed random seed for reproducibility
    dpi=72,
    color_scheme=ColorScheme.DEFAULT
)

print(f"Report generated and saved to: {report_path}")