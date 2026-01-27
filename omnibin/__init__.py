# Binary Classification (original)
from .metrics import generate_binary_classification_report
from .utils import ColorScheme

# Regression
from .regression_metrics import generate_regression_report
from .regression_utils import RegressionColorScheme

# Segmentation
from .segmentation_metrics import (
    generate_segmentation_report,
    generate_multiclass_segmentation_report
)
from .segmentation_utils import SegmentationColorScheme

# Detection
from .detection_metrics import (
    generate_detection_report,
    generate_lesion_detection_report
)
from .detection_utils import DetectionColorScheme

__version__ = "0.2.0"

__all__ = [
    # Binary Classification
    "generate_binary_classification_report",
    "ColorScheme",

    # Regression
    "generate_regression_report",
    "RegressionColorScheme",

    # Segmentation
    "generate_segmentation_report",
    "generate_multiclass_segmentation_report",
    "SegmentationColorScheme",

    # Detection
    "generate_detection_report",
    "generate_lesion_detection_report",
    "DetectionColorScheme"
]
