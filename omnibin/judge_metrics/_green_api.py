"""
GREEN API-mode: replay GREEN's pipeline with a non-default judge model.

Uses GREEN's exact prompt (see _prompts.green_build_prompt) and GREEN's
exact response-parsing regexes (ported verbatim from green_score/green.py
and green_score/utils.py, Apache-2.0). Only the model inference step is
different — instead of a local HuggingFace GREEN-RadLlama2-7b, we send the
same prompt to any provider via litellm.

The scalar score is unchanged:
    green = matched / (matched + Σ significant-error subcounts)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..llm_provider import LLMConfig, LLMProvider
from ._prompts import (
    GREEN_ERROR_SUBCATEGORIES,
    green_build_prompt,
)


GREEN_CATEGORIES = (
    "Clinically Significant Errors",
    "Clinically Insignificant Errors",
    "Matched Findings",
)


def _clean_response(response: str) -> str:
    """Verbatim port of green_score.utils.clean_responses."""
    if "[Explanation]:" in response:
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1]
        if (
            "[Explanation]:\n    <Explanation>\n" in response
            or "[Explanation]:\n<Explanation>" in response
        ):
            response = response.split("[Explanation]:")[1]
        else:
            response = response.split("[Explanation]:")[-1]
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1]
    return response.replace("</s>", "").replace("<unk>", "")


def _parse_error_counts(text: str, category: str) -> tuple[Optional[int], list[int]]:
    """
    Verbatim port of GREEN.parse_error_counts.

    Returns (top_count, subcategory_counts).
    - For error categories, top_count == sum(subcategory_counts).
    - For "Matched Findings", returns (int, []).
    """
    pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
    category_text = re.search(pattern, text, re.DOTALL)

    sub_counts = [0] * 6
    total_count: Optional[int] = 0

    if category_text is None or category_text.group(1).startswith("No"):
        if category == "Matched Findings":
            return 0, []
        return 0, sub_counts

    block = category_text.group(1)

    if category == "Matched Findings":
        nums = re.findall(r"^\b\d+\b(?=\.)", block)
        total = int(nums[0]) if nums else 0
        return total, []

    matches = sorted(re.findall(r"\([a-f]\) .*", block))
    if len(matches) == 0:
        matches = sorted(re.findall(r"\([1-6]\) .*", block))
    if len(matches) == 0:
        # Malformed output — surface as None so callers can skip the sample.
        return None, sub_counts

    for i, line in enumerate(matches[:6]):
        count = re.findall(r"(?<=: )\b\d+\b(?=\.)", line)
        if len(count) == 1:
            sub_counts[i] = int(count[0])

    total_count = sum(sub_counts)
    return total_count, sub_counts


@dataclass
class _OneSampleResult:
    score: Optional[float]
    matched: int
    significant_errors: list[int]  # 6 subcategory counts
    raw_response: str


def _score_one_sample(reference: str, candidate: str, provider: LLMProvider) -> _OneSampleResult:
    prompt = green_build_prompt(reference, candidate)
    raw = provider.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024,
    )
    response = _clean_response(raw)

    matched, _ = _parse_error_counts(response, "Matched Findings")
    sig_total, sig_sub = _parse_error_counts(response, "Clinically Significant Errors")

    if matched is None or sig_total is None:
        return _OneSampleResult(None, matched or 0, sig_sub, raw)

    if matched == 0 and sig_total == 0:
        # Upstream defines: matched==0 → score=0.
        return _OneSampleResult(0.0, 0, sig_sub, raw)

    if matched == 0:
        return _OneSampleResult(0.0, 0, sig_sub, raw)

    score = matched / (matched + sig_total)
    return _OneSampleResult(float(score), matched, sig_sub, raw)


def compute_green_via_api(
    references: list[str],
    candidates: list[str],
    llm_config: LLMConfig,
) -> dict:
    """
    Run GREEN's pipeline with a non-default judge model.

    Returns a dict with the same shape the main wrapper expects:
      - per_sample: list of floats (score in [0, 1])
      - aggregate: mean score
      - std: std of per-sample scores
      - submetrics: per-subcategory mean error counts
      - num_llm_failures: int
    """
    provider = LLMProvider(llm_config)

    per_sample: list[float] = []
    per_sub_counts = [[] for _ in range(6)]
    per_matched: list[int] = []
    num_failures = 0

    for ref, cand in zip(references, candidates):
        try:
            result = _score_one_sample(ref, cand, provider)
        except Exception:
            num_failures += 1
            continue

        if result.score is None:
            num_failures += 1
            continue

        per_sample.append(result.score)
        per_matched.append(result.matched)
        for i in range(6):
            per_sub_counts[i].append(result.significant_errors[i])

    import numpy as np

    agg = float(np.mean(per_sample)) if per_sample else float("nan")
    std = float(np.std(per_sample)) if per_sample else float("nan")

    submetrics: dict[str, float] = {"GREEN (mean)": agg, "GREEN (std)": std}
    for i, label in enumerate(GREEN_ERROR_SUBCATEGORIES):
        if per_sub_counts[i]:
            submetrics[f"GREEN sig-err ({chr(ord('a') + i)}) {label}"] = float(
                np.mean(per_sub_counts[i])
            )
    if per_matched:
        submetrics["GREEN matched findings (mean)"] = float(np.mean(per_matched))
    submetrics["GREEN LLM failures"] = float(num_failures)

    return {
        "per_sample": per_sample,
        "aggregate": agg,
        "std": std,
        "submetrics": submetrics,
        "num_llm_failures": num_failures,
    }
