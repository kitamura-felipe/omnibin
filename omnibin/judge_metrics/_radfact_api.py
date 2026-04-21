"""
RadFact API-mode: replay RadFact's two-stage pipeline with a non-default judge.

Uses RadFact's exact system messages (MIT, from microsoft/RadFact) for both
stages — phrase decomposition and single-phrase entailment verification.

DEVIATIONS FROM THE ORIGINAL PIPELINE
------------------------------------
1. The upstream `radfact` package teaches output format with 10 few-shot
   YAML examples per stage. API mode instead asks the LLM to emit JSON
   directly, following the schema in the system prompt. Modern chat models
   follow structured-output instructions reliably without few-shots, but
   scores may drift from published RadFact numbers.
2. Only logical precision / recall / F1 are produced. Spatial and grounding
   scores require bounding boxes and are reported as N/A.
3. No retries / phrase-rewrite correction — malformed LLM outputs are
   counted as failures and skipped.

These deviations are surfaced in the Gradio UI disclaimer and in the
report's submetrics table.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from ..llm_provider import LLMConfig, LLMProvider
from ._prompts import (
    RADFACT_DECOMPOSITION_OUTPUT_INSTRUCTION,
    RADFACT_DECOMPOSITION_SYSTEM,
    RADFACT_ENTAILMENT_OUTPUT_INSTRUCTION,
    RADFACT_ENTAILMENT_SYSTEM,
)


def _extract_json(raw: str) -> Optional[dict]:
    """Pull a JSON object out of an LLM response, tolerant of code fences."""
    if raw is None:
        return None
    s = raw.strip()
    # Strip markdown code fences if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    # Find the outermost JSON object.
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


@dataclass
class _DecompositionResult:
    phrases: list[str]
    failed: bool


def _decompose(report: str, provider: LLMProvider) -> _DecompositionResult:
    messages = [
        {
            "role": "system",
            "content": RADFACT_DECOMPOSITION_SYSTEM
            + "\n\n"
            + RADFACT_DECOMPOSITION_OUTPUT_INSTRUCTION,
        },
        {"role": "user", "content": f"Report:\n{report}"},
    ]
    raw = provider.chat(messages, temperature=0.0, max_tokens=1024)
    data = _extract_json(raw)
    if not data or not isinstance(data.get("phrases"), list):
        return _DecompositionResult(phrases=[], failed=True)
    phrases = [p.strip() for p in data["phrases"] if isinstance(p, str) and p.strip()]
    return _DecompositionResult(phrases=phrases, failed=False)


@dataclass
class _EntailmentResult:
    entailed: bool
    evidence: list[str]
    failed: bool


def _entailment(
    hypothesis: str,
    reference_phrases: list[str],
    provider: LLMProvider,
) -> _EntailmentResult:
    premise_text = "\n".join(f"- {p}" for p in reference_phrases) or "(no phrases)"
    user_content = (
        f"Reference phrases:\n{premise_text}\n\n"
        f"Hypothesis to verify: {hypothesis}"
    )
    messages = [
        {
            "role": "system",
            "content": RADFACT_ENTAILMENT_SYSTEM
            + "\n\n"
            + RADFACT_ENTAILMENT_OUTPUT_INSTRUCTION,
        },
        {"role": "user", "content": user_content},
    ]
    raw = provider.chat(messages, temperature=0.0, max_tokens=512)
    data = _extract_json(raw)
    if not data or "status" not in data:
        return _EntailmentResult(entailed=False, evidence=[], failed=True)
    status = str(data.get("status", "")).lower().strip()
    evidence = [e for e in (data.get("evidence") or []) if isinstance(e, str)]
    entailed = status == "entailment"
    return _EntailmentResult(entailed=entailed, evidence=evidence, failed=False)


def compute_radfact_via_api(
    references: list[str],
    candidates: list[str],
    llm_config: LLMConfig,
) -> dict:
    """
    Returns a dict with:
      - per_sample: list of per-pair logical_f1 values
      - aggregate: mean logical_f1 across pairs
      - submetrics: logical_precision / logical_recall / logical_f1 means,
        plus LLM failure counts
      - num_llm_failures: total failed LLM calls across all pairs
    """
    provider = LLMProvider(llm_config)

    per_sample_f1: list[float] = []
    per_sample_p: list[float] = []
    per_sample_r: list[float] = []
    failures = 0

    for ref, cand in zip(references, candidates):
        ref_dec = _decompose(ref, provider)
        cand_dec = _decompose(cand, provider)
        failures += int(ref_dec.failed) + int(cand_dec.failed)
        if ref_dec.failed or cand_dec.failed:
            continue

        # logical_precision: fraction of candidate phrases entailed by reference
        if not cand_dec.phrases:
            precision = 0.0
        else:
            entailed_c = 0
            for phrase in cand_dec.phrases:
                r = _entailment(phrase, ref_dec.phrases, provider)
                if r.failed:
                    failures += 1
                    continue
                if r.entailed:
                    entailed_c += 1
            precision = entailed_c / len(cand_dec.phrases)

        # logical_recall: fraction of reference phrases entailed by candidate
        if not ref_dec.phrases:
            recall = 0.0
        else:
            entailed_r = 0
            for phrase in ref_dec.phrases:
                r = _entailment(phrase, cand_dec.phrases, provider)
                if r.failed:
                    failures += 1
                    continue
                if r.entailed:
                    entailed_r += 1
            recall = entailed_r / len(ref_dec.phrases)

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_sample_p.append(precision)
        per_sample_r.append(recall)
        per_sample_f1.append(f1)

    import numpy as np

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    submetrics = {
        "RadFact logical P (API mode)": _mean(per_sample_p),
        "RadFact logical R (API mode)": _mean(per_sample_r),
        "RadFact logical F1 (API mode)": _mean(per_sample_f1),
        "RadFact LLM failures": float(failures),
        "RadFact spatial / grounding": float("nan"),  # not available without boxes
    }

    return {
        "per_sample": per_sample_f1,
        "aggregate": _mean(per_sample_f1),
        "submetrics": submetrics,
        "num_llm_failures": failures,
    }
