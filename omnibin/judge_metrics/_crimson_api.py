"""
CRIMSON API-mode: replay CRIMSON's scoring pipeline with a non-default judge.

Uses CRIMSON's verbatim prompt + parser + score calculation (MIT, vendored
in ``_crimson_vendor/``). Only the model-call step is different — instead
of the upstream OpenAI/HF backends, we route through ``LLMProvider``
(litellm) so any of the 5 supported providers can drive it.

DEVIATIONS FROM THE ORIGINAL ``crimson-score`` PACKAGE
------------------------------------------------------
1. The upstream OpenAI path sets ``response_format={"type": "json_object"}``.
   Not every provider (e.g. some Anthropic/Google models) exposes this
   parameter via their OpenAI-compatible endpoint. We request JSON via the
   prompt itself (CRIMSON's prompt already says "Return ONLY valid JSON")
   and rely on the robust ``parse_json_response`` progressive-fix pipeline.
2. The upstream OpenAI path passes ``seed=42``. We pass it through litellm
   when the provider supports it, otherwise it is silently ignored.

Everything else — the 13-section prompt, JSON schema, parsing regexes,
weight constants, and the crimson_score formula — is byte-for-byte upstream.

When the official ``crimson-score`` package installs cleanly (e.g. once
upstream relaxes its pandas pin: see rajpurkarlab/CRIMSON#3), users
should prefer the paper-default path via ``omnibin[crimson]``.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..llm_provider import LLMConfig, LLMProvider
from ._crimson_vendor import build_prompt, parse_json_response, calculate_crimson


CRIMSON_SYSTEM_MESSAGE = (
    "You are an expert radiology evaluator that assesses the accuracy of "
    "radiology reports."
)


@dataclass
class _OneSampleResult:
    crimson_score: float
    raw: dict
    failed: bool


def _evaluate_one(
    reference: str,
    candidate: str,
    provider: LLMProvider,
) -> _OneSampleResult:
    prompt = build_prompt(
        reference_findings=reference,
        predicted_findings=candidate,
        patient_context=None,
        include_significance_examples=True,
        include_attribute_guidelines=True,
        include_context_guidelines=True,
    )

    messages = [
        {"role": "system", "content": CRIMSON_SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]

    try:
        raw = provider.chat(
            messages,
            temperature=0.0,
            max_tokens=8192,
            seed=42,
        )
    except Exception:
        return _OneSampleResult(crimson_score=float("nan"), raw={}, failed=True)

    try:
        evaluation = parse_json_response(raw)
    except Exception:
        return _OneSampleResult(crimson_score=float("nan"), raw={"response": raw}, failed=True)

    try:
        result = calculate_crimson(evaluation)
    except Exception:
        return _OneSampleResult(crimson_score=float("nan"), raw={"response": raw}, failed=True)

    return _OneSampleResult(
        crimson_score=float(result["crimson_score"]),
        raw=result,
        failed=False,
    )


def compute_crimson_via_api(
    references: list[str],
    candidates: list[str],
    llm_config: LLMConfig,
) -> dict:
    """
    Returns a dict with:
      - per_sample: list of crimson_score floats (range (-1, 1])
      - aggregate: mean crimson_score across successful samples
      - submetrics: per-error-category mean counts, plus LLM failure count
      - raw_per_sample: full upstream result dicts for debugging
      - num_llm_failures: int
    """
    provider = LLMProvider(llm_config)

    scores: list[float] = []
    raw_results: list[dict] = []
    counters = {
        "false_findings": [],
        "missing_findings": [],
        "attribute_errors": [],
        "location_errors": [],
        "severity_errors": [],
        "descriptor_errors": [],
        "measurement_errors": [],
        "certainty_errors": [],
        "unspecific_errors": [],
        "overinterpretation_errors": [],
        "temporal_errors": [],
    }
    num_failures = 0

    for ref, cand in zip(references, candidates):
        result = _evaluate_one(ref, cand, provider)
        raw_results.append(result.raw)
        if result.failed:
            num_failures += 1
            continue
        scores.append(result.crimson_score)
        err_counts = result.raw.get("error_counts", {})
        for key in counters:
            counters[key].append(int(err_counts.get(key, 0)))

    import numpy as np

    aggregate = float(np.mean(scores)) if scores else float("nan")
    submetrics: dict[str, float] = {"CRIMSON score (API mode)": aggregate}

    label_map = {
        "false_findings": "CRIMSON false findings (mean)",
        "missing_findings": "CRIMSON missing findings (mean)",
        "attribute_errors": "CRIMSON attribute errors (mean)",
        "location_errors": "CRIMSON location errors (mean)",
        "severity_errors": "CRIMSON severity errors (mean)",
        "descriptor_errors": "CRIMSON descriptor errors (mean)",
        "measurement_errors": "CRIMSON measurement errors (mean)",
        "certainty_errors": "CRIMSON certainty errors (mean)",
        "unspecific_errors": "CRIMSON unspecific errors (mean)",
        "overinterpretation_errors": "CRIMSON overinterpretation errors (mean)",
        "temporal_errors": "CRIMSON temporal errors (mean)",
    }
    for k, label in label_map.items():
        if counters[k]:
            submetrics[label] = float(np.mean(counters[k]))

    submetrics["CRIMSON LLM failures"] = float(num_failures)

    return {
        "per_sample": scores,
        "aggregate": aggregate,
        "submetrics": submetrics,
        "raw_per_sample": raw_results,
        "num_llm_failures": num_failures,
    }
