"""
CRIMSON — Clinically-grounded LLM-based metric for radiology reports.
reference: rajpurkarlab/CRIMSON — https://github.com/rajpurkarlab/CRIMSON
paper: Baharoon et al., "CRIMSON", 2026.

The upstream package exposes two backends:
  - api="huggingface"  → their fine-tuned MedGemmaCRIMSON 4B (paper default)
  - api="openai"       → any OpenAI-compatible endpoint via model_name + base_url

For Anthropic/Gemini/Groq/OpenRouter we route through `api="openai"` and set
the OpenAI-compatible base URL for each provider.

Install:
    pip install omnibin[crimson]    # adds crimson-score
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..llm_provider import LLMConfig


# Paper default is the fine-tuned MedGemmaCRIMSON served locally via HF.
CRIMSON_PAPER_DEFAULT_MODEL = "huggingface:MedGemmaCRIMSON"


@dataclass
class CRIMSONResult:
    per_sample: list[float]
    aggregate: float
    submetrics: dict[str, float] = field(default_factory=dict)
    raw_per_sample: list[dict] = field(default_factory=list)


def compute_crimson(
    references: list[str],
    candidates: list[str],
    llm_config: Optional[LLMConfig] = None,
) -> CRIMSONResult:
    """
    Compute CRIMSON scores using the official crimson-score package.

    If `llm_config` is None, CRIMSON's default HuggingFace backend
    (MedGemmaCRIMSON-4B) is used — matches the paper but needs a GPU.
    If `llm_config` is provided, the OpenAI-compatible backend is used
    with the selected provider's base URL + API key.

    Returns
    -------
    CRIMSONResult
        - per_sample: list of crimson_score floats (range -1..1)
        - aggregate: mean crimson_score
        - submetrics: mean of each error category (false findings, missing
          findings, attribute errors)
        - raw_per_sample: the full upstream dicts for debugging
    """
    try:
        from CRIMSON import CRIMSONScore
    except ImportError as e:
        raise ImportError(
            "CRIMSON requires the official package. Install with:\n"
            "    pip install omnibin[crimson]\n"
            "or:\n"
            "    pip install crimson-score"
        ) from e

    if llm_config is None:
        scorer = CRIMSONScore(api="huggingface", model_name=None)
    else:
        api_key = llm_config.require_api_key()
        import os
        os.environ["OPENAI_API_KEY"] = api_key
        if llm_config.base_url:
            os.environ["OPENAI_BASE_URL"] = llm_config.base_url
            os.environ["OPENAI_API_BASE"] = llm_config.base_url
        scorer = CRIMSONScore(api="openai", model_name=llm_config.model)

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
    )
