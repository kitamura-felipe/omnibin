# Omnibin

A Python package for generating comprehensive binary classification reports with visualizations and confidence intervals.

## Installation

```bash
pip install omnibin
```

## Usage

```python
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
```

## Input Format

The input data should be provided as:
- `y_true`: Array of true binary labels (0 or 1)
- `y_pred`: Array of predicted probabilities or scores

## Features

- Generates a comprehensive PDF report with:
  - ROC curve with confidence bands
  - Precision-Recall curve with confidence bands
  - Metrics vs. threshold plots
  - Confusion matrix at optimal threshold
  - Calibration plot
  - Summary table with confidence intervals
- Calculates optimal threshold using Youden's J statistic
- Provides confidence intervals using bootstrapping
- Supports both probability and score-based predictions

## Metrics Included

- Accuracy
- Sensitivity (Recall)
- Specificity
- Positive Predictive Value (Precision)
- Matthews Correlation Coefficient
- F1 Score
- AUC-ROC
- AUC-PR

All metrics include 95% confidence intervals calculated through bootstrapping.

## Output

The package generates a PDF report containing:
1. ROC and Precision-Recall curves with confidence bands
2. Metrics plotted across different thresholds
3. Confusion matrix at the optimal threshold
4. Calibration plot
5. Summary table with all metrics and their confidence intervals 