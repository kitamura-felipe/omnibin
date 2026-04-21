"""
CRIMSON — Clinically-grounded LLM-based metric for radiology reports.
reference: rajpurkarlab/CRIMSON — https://github.com/rajpurkarlab/CRIMSON
paper: Baharoon et al., "CRIMSON", 2026.

Two modes:
  - **Paper default** (no llm_config given): the upstream ``crimson-score``
    package, which can target either its fine-tuned MedGemmaCRIMSON-4B
    (``api="huggingface"``) or any OpenAI-compatible endpoint
    (``api="openai"``). Install:
        pip install omnibin[crimson]
    Note: as of crimson-score 0.2.0 the package pins ``pandas>=3.0.1``,
    which conflicts with Gradio 5.x — install in a separate env from the
    Gradio app. (Tracked upstream: rajpurkarlab/CRIMSON#3.)
  - **API mode** (llm_config given): replays CRIMSON's verbatim prompt +
    parser + score-calculation via our 5-provider router, bypassing the
    upstream package entirely. See ``_crimson_api.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..llm_provider import LLMConfig


CRIMSON_PAPER_DEFAULT_MODEL = "huggingface:MedGemmaCRIMSON"


@dataclass
class CRIMSONResult:
    per_sample: list[float]
    aggregate: float
    submetrics: dict[str, float] = field(default_factory=dict)
    raw_per_sample: list[dict] = field(default_factory=list)
    num_llm_failures: int = 0
    api_mode: bool = False


def compute_crimson(
    references: list[str],
    candidates: list[str],
    llm_config: Optional[LLMConfig] = None,
) -> CRIMSONResult:
    """
    Compute CRIMSON scores.

    If ``llm_config`` is given, runs API mode (vendored verbatim prompt +
    parser + score) against the selected provider. Otherwise delegates to
    the official ``crimson-score`` package.
    """
    if llm_config is not None:
        from ._crimson_api import compute_crimson_via_api
        res = compute_crimson_via_api(references, candidates, llm_config)
        return CRIMSONResult(
            per_sample=res["per_sample"],
            aggregate=res["aggregate"],
            submetrics=res["submetrics"],
            raw_per_sample=res["raw_per_sample"],
            num_llm_failures=res["num_llm_failures"],
            api_mode=True,
        )

    try:
        from CRIMSON import CRIMSONScore
    except ImportError as e:
        raise ImportError(
            "CRIMSON requires the official package. Install with:\n"
            "    pip install omnibin[crimson]\n"
            "or:\n"
            "    pip install crimson-score\n"
            "Alternative: pass `llm_config=LLMConfig(provider=..., ...)` to run in API mode."
        ) from e

    scorer = CRIMSONScore(api="huggingface", model_name=None)

    scores: list[float] = []
    false_findings: list[int] = []
    missing_findings: list[int] = []
    attribute_errors: list[int] = []
    raw: list[dict] = []

    for ref, cand in zip(references, candidates):
        result = scorer.evaluate(
            reference_findings=ref,
            predicted_findings=cand,
        )
        raw.append(result)
        scores.append(float(result.get("crimson_score", float("nan"))))
        errs = result.get("error_counts", {}) or {}
        false_findings.append(int(errs.get("false_findings", 0) or 0))
        missing_findings.append(int(errs.get("missing_findings", 0) or 0))
        attribute_errors.append(int(errs.get("attribute_errors", 0) or 0))

    import numpy as np
    aggregate = float(np.nanmean(scores)) if scores else float("nan")
    submetrics = {
        "CRIMSON score": aggregate,
        "CRIMSON false findings (mean)": float(np.mean(false_findings)) if false_findings else 0.0,
        "CRIMSON missing findings (mean)": float(np.mean(missing_findings)) if missing_findings else 0.0,
        "CRIMSON attribute errors (mean)": float(np.mean(attribute_errors)) if attribute_errors else 0.0,
    }

    return CRIMSONResult(
        per_sample=scores,
        aggregate=aggregate,
        submetrics=submetrics,
        raw_per_sample=raw,
        num_llm_failures=0,
        api_mode=False,
    )
