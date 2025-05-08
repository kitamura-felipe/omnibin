import pandas as pd
import numpy as np
import os
from omnibin import generate_binary_classification_report

data = pd.DataFrame({'y_true': (y:=np.random.choice([0,1],1000,p:=[.9,.1])),'y_pred': np.where(y, np.random.beta(4,1.5,1000)*.8+.1, np.random.beta(1.5,4,1000)*.8+.1)})

y_true = data['y_true'].values
y_scores = data['y_pred'].values

# Generate comprehensive classification report
report_path = generate_binary_classification_report(
    y_true=y_true,
    y_scores=y_scores,
    output_path=os.path.join(RESULTS_DIR, "classification_report.pdf"),
    n_bootstrap=1000,
    random_seed=42,  # Set a fixed random seed for reproducibility
    dpi=72
)

print(f"Report generated and saved to: {report_path}")