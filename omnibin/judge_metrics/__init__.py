"""
Thin wrappers around three LLM-judge radiology report metrics:

- GREEN  — Stanford-AIMI/GREEN            (Ostmeier et al., EMNLP Findings 2024)
- RadFact — microsoft/radfact              (Bannur et al., MAIRA-2, 2024)
- CRIMSON — rajpurkarlab/CRIMSON           (Baharoon et al., 2026)

Each wrapper delegates to the original reference implementation and
raises a clear installation message if the upstream package isn't
available. We never re-implement a metric's prompting or scoring logic.
"""
from .green import compute_green, GREENResult, GREEN_PAPER_DEFAULT_MODEL
from .radfact import compute_radfact, RadFactResult, RADFACT_PAPER_DEFAULT_MODEL
from .crimson import compute_crimson, CRIMSONResult, CRIMSON_PAPER_DEFAULT_MODEL

__all__ = [
    "compute_green",
    "GREENResult",
    "GREEN_PAPER_DEFAULT_MODEL",
    "compute_radfact",
    "RadFactResult",
    "RADFACT_PAPER_DEFAULT_MODEL",
    "compute_crimson",
    "CRIMSONResult",
    "CRIMSON_PAPER_DEFAULT_MODEL",
]
