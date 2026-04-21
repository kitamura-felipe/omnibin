"""
Vendored CRIMSON implementation files, used when the official
`crimson-score` PyPI package cannot be installed (its pandas>=3.0.1
pin conflicts with Gradio 5.x's pandas<3.0).

Upstream: https://github.com/rajpurkarlab/CRIMSON  (MIT, see LICENSE).

Files here:
- prompt_parts.py — exact copy of CRIMSON/prompt_parts.py
- utils.py        — exact copy of CRIMSON/utils.py
- score.py        — CRIMSONScore._calculate_crimson lifted to module-level
- LICENSE         — upstream MIT license

Do not edit the vendored files. Refresh them by re-copying from upstream
if prompts change.
"""
from .prompt_parts import build_prompt
from .utils import parse_json_response, clean_report_text
from .score import calculate_crimson

__all__ = ["build_prompt", "parse_json_response", "clean_report_text", "calculate_crimson"]
