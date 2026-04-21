# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-04-21

### Added

#### Text Generation Metrics (radiology reports)
- `generate_text_generation_report()` for paired reference/candidate reports
- `TextGenerationReport` dataclass with aggregate scores, per-sample scores,
  submetrics, and bootstrap 95% CIs
- Lexical metrics (all using each paper's original reference implementation):
  - BLEU (sacrebleu — corpus and sentence-level)
  - ROUGE-1 / ROUGE-2 / ROUGE-L (rouge-score)
  - METEOR (nltk.translate.meteor_score)
  - BERTScore P / R / F1 (bert-score)
- LLM-judge metrics, each with two modes:
  - **Paper-default mode** — wraps the original upstream package verbatim:
    - **GREEN** (Ostmeier et al., 2024) — `pip install omnibin[green]`
    - **RadFact** (Bannur et al., MAIRA-2, 2024) — `pip install omnibin[radfact]`
    - **CRIMSON** (Baharoon et al., 2026) — `pip install omnibin[crimson]`
  - **API mode** — replays each paper's pipeline with a swappable judge:
    - GREEN API mode: verbatim prompt + parser (Apache-2.0, ported from
      `green_score/utils.py` and `green_score/green.py`), only the model
      call routes through litellm
    - RadFact API mode: verbatim two-stage system messages (MIT, from
      microsoft/RadFact) with zero-shot JSON output instead of 10-shot
      YAML; logical P/R/F1 only, no spatial/grounding
    - All three metrics now runnable via any of 5 providers (OpenAI,
      Anthropic, Google, OpenRouter, Groq) with a disclaimer in the UI
- `LLMConfig` + `LLMProvider` router supporting 5 providers:
  OpenAI, Anthropic, Google (Gemini), OpenRouter, Groq (via litellm)
- Per-paper default judge models surfaced in docstrings and Gradio UI
- Visualizations:
  - Aggregate bar chart with 95% CI error bars
  - Per-sample violin / strip distribution
  - Pearson correlation heatmap between metrics
  - Per-pair × metric heatmap
  - Submetric breakdown table
- `TextGenColorScheme` enum (DEFAULT / MONOCHROME / VIBRANT)
- 10 fabricated CXR report pairs added as example data (clearly marked
  FABRICATED; covers paraphrase, laterality flip, severity mis-call,
  hallucination, missed finding)
- New Gradio tab with metric checkboxes, provider dropdown, API-key field
- Tests covering input validation, report generation, bootstrap CI,
  provider configuration, and color schemes (13 new tests)

### Changed
- Bumped version to 0.3.0
- `pyproject.toml`: added `text`, `llm-judge`, `green`, `radfact`,
  `crimson`, and `all-text` optional-dependency extras
- `requirements.txt` (Gradio Space): added lexical metric deps, litellm,
  and crimson-score (GREEN and RadFact deliberately omitted due to GPU
  / pydantic-1.x-vs-Gradio-5 conflicts)

## [0.2.0] - 2025-01-27

### Added

#### Regression Metrics
- `generate_regression_report()` for continuous value prediction tasks
- Metrics: MAE, MSE, RMSE, R², Adjusted R², MAPE, Explained Variance, NRMSE, CV-RMSE
- Visualizations:
  - Scatter plot with regression line
  - Residual plots (vs. predicted, histogram)
  - Q-Q plot for normality assessment
  - Bland-Altman plot (essential for medical agreement studies)
  - Error distribution histograms
  - Prediction intervals
- Bootstrap confidence intervals for all metrics
- `RegressionColorScheme` enum for customizable styling

#### Segmentation Metrics
- `generate_segmentation_report()` for 2D and 3D binary segmentation
- `generate_multiclass_segmentation_report()` for multi-class segmentation
- Metrics:
  - Dice Score (Sørensen–Dice coefficient)
  - IoU / Jaccard Index
  - Pixel Accuracy
  - Sensitivity and Specificity
  - Precision (PPV)
  - Volumetric Similarity
  - Hausdorff Distance (standard and 95th percentile)
  - Average Surface Distance
- Visualizations:
  - Side-by-side comparison (GT vs. Prediction vs. Overlay)
  - Pixel-wise confusion matrix
  - Metrics bar chart
  - Surface distance histograms
  - Per-slice Dice distribution (for 3D data)
- `SegmentationColorScheme` enum for customizable styling

#### Detection Metrics
- `generate_detection_report()` for object detection tasks
- `generate_lesion_detection_report()` optimized for medical imaging
- Metrics:
  - Average Precision at various IoU thresholds (AP@50, AP@75)
  - Mean Average Precision (mAP)
  - FROC Score (Free-Response ROC - medical imaging standard)
  - Precision, Recall, F1 at IoU=0.5
  - Detection Rate
  - Average False Positives per Image
  - Sensitivity at specific FP rates (for FROC analysis)
- Visualizations:
  - Precision-Recall curves at different IoU thresholds
  - FROC curve (Sensitivity vs. FP/Image)
  - IoU distribution histogram
  - Confidence score distributions (TP vs. FP)
  - Detections per image analysis
- Support for distance-based matching (alternative to IoU)
- `DetectionColorScheme` enum for customizable styling

#### General
- Updated Gradio app with tabs for all report types
- Comprehensive test suites for all new modules
- Updated documentation with examples for all report types
- Healthcare-focused classifiers in package metadata

### Changed
- Bumped version to 0.2.0
- Updated README with comprehensive documentation
- Added `tqdm` to dependencies for progress bars

## [0.1.0] - 2024-03-19

### Added
- Initial release
- Basic binary classification metrics and visualizations
- Comprehensive reporting functionality
- Confidence interval calculations
- Example usage and documentation
