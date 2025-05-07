import pandas as pd
from omnibin import generate_binary_classification_report

# Load your data
data = pd.read_csv("scores.csv")
y_true = data['y_true'].values
y_scores = data['y_pred'].values

# Generate comprehensive classification report
report_path = generate_binary_classification_report(
    y_true=y_true,
    y_scores=y_scores,
    output_path="classification_report.pdf",
    n_bootstrap=1000
)

print(f"Report generated and saved to: {report_path}")