[![Tests](https://github.com/kitamura-felipe/omnibin/actions/workflows/test.yml/badge.svg)](https://github.com/kitamura-felipe/omnibin/actions/workflows/test.yml)
[![Deploy](https://github.com/kitamura-felipe/omnibin/actions/workflows/deploy.yml/badge.svg)](https://github.com/kitamura-felipe/omnibin/actions/workflows/deploy.yml)


# Omnibin

A comprehensive Python package for generating detailed machine learning evaluation reports with visualizations, confidence intervals, and statistical analysis. Supports **binary classification**, **regression**, **segmentation**, **object detection**, and **text generation (radiology reports)** tasks with a focus on healthcare applications.

## Try it Online

You can try Omnibin directly in your browser through our [Hugging Face Space](https://felipekitamura-omnibin.hf.space).

## Installation

```bash
pip install omnibin
```

## Features

### Report Types

| Report Type | Use Case | Key Metrics |
|------------|----------|-------------|
| **Binary Classification** | Disease diagnosis, risk prediction | AUC-ROC, AUC-PR, Sensitivity, Specificity, PPV, NPV |
| **Regression** | Continuous value prediction (e.g., tumor size, survival time) | MAE, RMSE, R², Bland-Altman analysis |
| **Segmentation** | Medical image segmentation (tumors, organs) | Dice Score, IoU, Hausdorff Distance, Surface Distance |
| **Detection** | Lesion/nodule detection | mAP, FROC Score, Precision-Recall at various IoU |
| **Text Generation** | Radiology report generation | BLEU, ROUGE, METEOR, BERTScore, GREEN, RadFact, CRIMSON |

### Common Features

- Comprehensive PDF reports with multiple visualizations
- 95% confidence intervals via bootstrapping
- Multiple color schemes (Default, Monochrome, Vibrant)
- Reproducible results with random seed control
- Healthcare-optimized metrics (Bland-Altman, FROC, etc.)

---

## Usage

### Binary Classification

```python
import numpy as np
from omnibin import generate_binary_classification_report, ColorScheme

# Your data
y_true = np.array([0, 1, 1, 0, 1, ...])  # Binary labels
y_scores = np.array([0.2, 0.8, 0.9, 0.1, 0.7, ...])  # Predicted probabilities

report_path = generate_binary_classification_report(
    y_true=y_true,
    y_scores=y_scores,
    output_path="classification_report.pdf",
    n_bootstrap=1000,
    random_seed=42,
    dpi=300,
    color_scheme=ColorScheme.DEFAULT
)
```

**Metrics Included:**
- Accuracy, Sensitivity (Recall), Specificity
- Positive/Negative Predictive Value (PPV/NPV)
- Matthews Correlation Coefficient (MCC)
- F1 Score, AUC-ROC, AUC-PR

**Visualizations:**
- ROC and Precision-Recall curves with confidence bands
- Metrics vs. threshold plots
- Confusion matrix at optimal threshold
- Calibration plot
- Prediction distribution

---

### Regression

```python
import numpy as np
from omnibin import generate_regression_report, RegressionColorScheme

# Your data
y_true = np.array([10.5, 20.3, 15.2, ...])  # Actual values
y_pred = np.array([11.2, 19.8, 14.9, ...])  # Predicted values

report_path = generate_regression_report(
    y_true=y_true,
    y_pred=y_pred,
    output_path="regression_report.pdf",
    n_bootstrap=1000,
    random_seed=42,
    dpi=300,
    color_scheme=RegressionColorScheme.DEFAULT
)
```

**Metrics Included:**
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE), Root MSE (RMSE)
- R² and Adjusted R²
- Mean Absolute Percentage Error (MAPE)
- Explained Variance
- Normalized RMSE, CV-RMSE

**Visualizations:**
- Scatter plot with regression line
- Residual plots (vs. predicted, histogram)
- Q-Q plot for normality assessment
- **Bland-Altman plot** (essential for medical agreement studies)
- Error distribution histograms
- Prediction intervals

---

### Segmentation

```python
import numpy as np
from omnibin import generate_segmentation_report, SegmentationColorScheme

# Your 2D or 3D segmentation masks
y_true = np.load("ground_truth_mask.npy")  # Binary mask
y_pred = np.load("predicted_mask.npy")     # Binary mask

report_path = generate_segmentation_report(
    y_true=y_true,
    y_pred=y_pred,
    output_path="segmentation_report.pdf",
    n_bootstrap=500,
    random_seed=42,
    dpi=300,
    color_scheme=SegmentationColorScheme.DEFAULT
)
```

**Metrics Included:**
- **Dice Score** (Sørensen–Dice coefficient)
- **IoU / Jaccard Index**
- Pixel Accuracy
- Sensitivity and Specificity
- Precision (PPV)
- Volumetric Similarity
- **Hausdorff Distance** (and 95th percentile)
- **Average Surface Distance**

**Visualizations:**
- Side-by-side comparison (GT vs. Prediction vs. Overlay)
- Pixel-wise confusion matrix
- Metrics bar chart
- Surface distance histograms
- Per-slice Dice distribution (for 3D data)

**Multi-class Segmentation:**
```python
from omnibin import generate_multiclass_segmentation_report

report_path = generate_multiclass_segmentation_report(
    y_true=y_true,
    y_pred=y_pred,
    output_path="multiclass_report.pdf",
    class_names={1: "Tumor", 2: "Edema", 3: "Necrosis"}
)
```

---

### Object Detection

```python
from omnibin import generate_detection_report, DetectionColorScheme

# Predictions: list of predictions per image
predictions = [
    [{"box": [10, 10, 50, 50], "score": 0.9}, {"box": [60, 60, 100, 100], "score": 0.7}],  # Image 1
    [{"box": [20, 20, 80, 80], "score": 0.85}],  # Image 2
    []  # Image 3 (no detections)
]

# Ground truths: list of boxes per image
ground_truths = [
    [[12, 12, 48, 48], [65, 65, 95, 95]],  # Image 1
    [[25, 25, 75, 75]],                     # Image 2
    [[30, 30, 70, 70]]                      # Image 3
]

report_path = generate_detection_report(
    predictions=predictions,
    ground_truths=ground_truths,
    output_path="detection_report.pdf",
    n_bootstrap=500,
    iou_thresholds=[0.5, 0.75]
)
```

**Metrics Included:**
- Average Precision at various IoU (AP@50, AP@75)
- Mean Average Precision (mAP)
- **FROC Score** (Free-Response ROC - medical imaging standard)
- Precision, Recall, F1 at IoU=0.5
- Detection Rate
- Average False Positives per Image

**Visualizations:**
- Precision-Recall curves at different IoU thresholds
- **FROC curve** (Sensitivity vs. FP/Image)
- IoU distribution histogram
- Confidence score distributions (TP vs. FP)
- Detections per image analysis

**Lesion Detection (Healthcare Focus):**
```python
from omnibin import generate_lesion_detection_report

# Optimized for medical imaging with FROC as primary metric
report_path = generate_lesion_detection_report(
    predictions=predictions,
    ground_truths=ground_truths,
    output_path="lesion_detection_report.pdf",
    use_distance=False,  # Use IoU-based matching
    distance_threshold=None
)
```

---

### Text Generation (Radiology Reports)

Lexical metrics (BLEU, ROUGE, METEOR, BERTScore) plus three LLM-judge metrics
wrapping their original reference implementations:

| Metric | Original paper | Paper's judge model | Install |
|--------|----------------|---------------------|---------|
| **GREEN** | Ostmeier et al., EMNLP Findings 2024 | `StanfordAIMI/GREEN-RadLlama2-7b` (local GPU) | `pip install omnibin[green]` |
| **RadFact** | Bannur et al., MAIRA-2, 2024 | Llama-3-70B-Instruct | `pip install omnibin[radfact]` |
| **CRIMSON** | Baharoon et al., 2026 | `MedGemmaCRIMSON-4B` (local GPU) | `pip install omnibin[crimson]` |

Each metric has two modes:

- **Paper-default mode** — uses the upstream package verbatim (GREEN's
  local Llama-2 fine-tune, RadFact's hydra pipeline, CRIMSON's MedGemma).
  Requires the extras install; GREEN/CRIMSON need a GPU.
- **API mode** — replays each paper's pipeline against any of 5 providers
  (**OpenAI, Anthropic, Google Gemini, OpenRouter, Groq**):
  - **GREEN** — verbatim prompt + parser (Apache-2.0); only the judge
    model swaps.
  - **RadFact** — verbatim two-stage system prompts but with zero-shot
    JSON output instead of the upstream 10-shot YAML. Logical P/R/F1 only,
    no grounding / spatial.
  - **CRIMSON** — verbatim prompt + JSON parser + scoring formula (MIT,
    vendored in `omnibin/judge_metrics/_crimson_vendor/`); only the judge
    model swaps.

  **Scores in API mode may differ from published paper numbers** — reserve
  for screening / demo use. For exact paper reproduction use the extras
  install.

```python
from omnibin import (
    generate_text_generation_report,
    TextGenColorScheme,
    LLMConfig,
)

references = ["Findings: The lungs are clear...", ...]
candidates = ["Findings: Heart size is normal...", ...]

# Lexical only — no API keys needed
report = generate_text_generation_report(
    references, candidates,
    output_path="text_generation_report.pdf",
    metrics=["bleu", "rouge", "meteor", "bertscore"],
    n_bootstrap=1000,
    color_scheme=TextGenColorScheme.DEFAULT,
)

# API mode — GREEN + RadFact + CRIMSON via Anthropic Claude (no GPU needed)
llm_config = LLMConfig(
    provider="anthropic",
    model="claude-sonnet-4-6",
    # api_key read from $ANTHROPIC_API_KEY by default
)
report = generate_text_generation_report(
    references, candidates,
    metrics=["bleu", "rouge", "bertscore", "green", "radfact", "crimson"],
    llm_config=llm_config,
)
print(report.aggregate_scores)
print(report.submetrics)          # includes each paper's sub-scores
print(report.confidence_intervals)
```

**Metrics Included:**
- BLEU (sacrebleu, corpus + sentence)
- ROUGE-1 / ROUGE-2 / ROUGE-L F
- METEOR
- BERTScore P / R / F1
- GREEN (mean score, per-error-category breakdown) *— via `green-score`*
- RadFact (logical / grounding / spatial precision & recall) *— via `radfact`*
- CRIMSON (score, false findings, missing findings, attribute errors) *— via `crimson-score`*

**Visualizations:**
- Aggregate bar chart with 95% CI error bars
- Per-sample violin / strip distribution
- Pearson correlation heatmap between metrics
- Per-pair × metric heatmap

**Providers** (`LLMConfig.provider`): `"openai"`, `"anthropic"`, `"google"`,
`"openrouter"`, `"groq"`. API keys are read from the provider's standard
env var (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`,
`OPENROUTER_API_KEY`, `GROQ_API_KEY`) unless passed explicitly.

> **Note on the hosted Space**: GREEN and RadFact run in **API mode** with
> a disclaimer shown to the user. The upstream packages are not installed
> on the Space (GREEN needs GPU + pinned torch; RadFact pins pydantic 1.x).
> For exact paper reproduction, use `pip install omnibin[green]`
> (needs GPU) or `pip install omnibin[radfact]` (separate env from Gradio).

---

## Color Schemes

All report types support three color schemes:

| Scheme | Description |
|--------|-------------|
| `DEFAULT` | Professional blue-based palette |
| `MONOCHROME` | Grayscale for publications |
| `VIBRANT` | High-contrast colorful palette |

---

## Input Formats

| Report Type | Input Format |
|------------|--------------|
| Classification | `y_true`: 0/1 array, `y_scores`: 0-1 probability array |
| Regression | `y_true`: continuous array, `y_pred`: continuous array |
| Segmentation | 2D or 3D NumPy arrays (binary masks) |
| Detection | Lists of dicts with `box` and `score` keys |
| Text Generation | Lists of reference/candidate report strings (or CSV with `reference`/`candidate` columns) |

---

## Example Outputs

### Binary Classification
![ROC and PR Curves](results/plots/roc_pr.png)
![Metrics Summary](results/plots/metrics_summary.png)

### Regression
- Scatter plot with regression line
- Bland-Altman agreement plot
- Q-Q plot for residual normality

### Segmentation
- Ground Truth / Prediction / Overlay comparison
- Dice score distribution across slices

### Detection
- FROC curve (standard for medical imaging)
- Precision-Recall at multiple IoU thresholds

---

## Requirements

- Python >= 3.11
- Core: NumPy, Pandas, Scikit-learn, Matplotlib, SciPy, Seaborn
- Optional extras:
  - `omnibin[text]` → sacrebleu, rouge-score, nltk, bert-score
  - `omnibin[llm-judge]` → litellm (provider router)
  - `omnibin[green]` → green-score (Stanford-AIMI/GREEN, local GPU)
  - `omnibin[crimson]` → crimson-score (rajpurkarlab/CRIMSON)
  - `omnibin[radfact]` → radfact from git (microsoft/radfact — separate env)
  - `omnibin[all-text]` → everything except radfact

## Acknowledgments

The text generation metrics wrap the original reference implementations of:
- [Stanford-AIMI/GREEN](https://github.com/Stanford-AIMI/GREEN) — Ostmeier et al., "GREEN: Generative Radiology Report Evaluation and Error Notation", EMNLP Findings 2024
- [microsoft/radfact](https://github.com/microsoft/radfact) — Bannur et al., "MAIRA-2: Grounded Radiology Report Generation", 2024
- [rajpurkarlab/CRIMSON](https://github.com/rajpurkarlab/CRIMSON) — Baharoon et al., "CRIMSON: A Clinically-Grounded LLM-Based Metric for Generative Radiology Report Evaluation", 2026

Please cite the original papers when using these metrics.

---

## License

MIT License - see LICENSE file for details.
