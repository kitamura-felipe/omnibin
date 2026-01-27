# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
