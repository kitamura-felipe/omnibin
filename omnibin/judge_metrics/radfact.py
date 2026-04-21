"""
RadFact — grounded radiology report evaluation.
reference: microsoft/RadFact — https://github.com/microsoft/RadFact
paper: Bannur et al., "MAIRA-2: Grounded Radiology Report Generation", 2024.

Two modes:
  - **Paper default** (no llm_config given and upstream installed):
    official `radfact` package with its hydra + langchain pipeline.
    Paper's judge is Llama-3-70B-Instruct. Install:
        pip install omnibin[radfact]
    Note: radfact pins pydantic 1.x — install in a separate env from Gradio.
  - **API mode** (llm_config given): replays RadFact's two-stage
    decomposition + entailment pipeline using the verbatim system messages
    from microsoft/RadFact but with zero-shot JSON output instead of the
    10-shot YAML format. Scores may differ from paper numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..llm_provider import LLMConfig


RADFACT_PAPER_DEFAULT_MODEL = "llama-3-70b-instruct"


@dataclass
class RadFactResult:
    per_sample: list[float]
    aggregate: float
    submetrics: dict[str, float] = field(default_factory=dict)
    num_llm_failures: int = 0
    api_mode: bool = False


def _f1(p: Optional[float], r: Optional[float]) -> float:
    if p is None or r is None or (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)


def compute_radfact(
    references: list[str],
    candidates: list[str],
    llm_config: Optional[LLMConfig] = None,
    is_narrative_text: bool = True,
) -> RadFactResult:
    """
    Compute RadFact metrics.

    If ``llm_config`` is given, runs API mode (verbatim system messages,
    zero-shot JSON, logical precision/recall only — no spatial/grounding).
    Otherwise delegates to the official microsoft/RadFact package.
    """
    if llm_config is not None:
        from ._radfact_api import compute_radfact_via_api
        res = compute_radfact_via_api(references, candidates, llm_config)
        return RadFactResult(
            per_sample=res["per_sample"],
            aggregate=res["aggregate"],
            submetrics=res["submetrics"],
            num_llm_failures=res["num_llm_failures"],
            api_mode=True,
        )

    try:
        from radfact.metric.radfact import RadFactMetric
    except ImportError as e:
        raise ImportError(
            "RadFact requires the official package. Install with:\n"
            "    pip install omnibin[radfact]\n"
            "or:\n"
            "    pip install git+https://github.com/microsoft/radfact.git\n"
            "Note: radfact pins pydantic 1.x which conflicts with Gradio 5.x — "
            "use a separate environment from the app.\n"
            "Alternative: pass `llm_config=LLMConfig(provider=..., ...)` to run in API mode."
        ) from e

    metric = RadFactMetric()
    metric_inputs = [
        {"id": i, "reference": r, "candidate": c, "is_narrative_text": is_narrative_text}
        for i, (r, c) in enumerate(zip(references, candidates))
    ]
    raw = metric.compute_metric_score(metric_inputs)

    def _get(obj, key, default=None):
        if hasattr(obj, key):
            return getattr(obj, key)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    aggregates = _get(raw, "metrics", raw) or {}
    per_sample_blocks = _get(raw, "per_sample_metrics", []) or []

    logical_p = float(_get(aggregates, "logical_precision", 0.0) or 0.0)
    logical_r = float(_get(aggregates, "logical_recall", 0.0) or 0.0)
    grounding_p = float(_get(aggregates, "grounding_precision", 0.0) or 0.0)
    grounding_r = float(_get(aggregates, "grounding_recall", 0.0) or 0.0)
    spatial_p = float(_get(aggregates, "spatial_precision", 0.0) or 0.0)
    spatial_r = float(_get(aggregates, "spatial_recall", 0.0) or 0.0)
    num_failures = int(_get(aggregates, "num_llm_failures", 0) or 0)

    per_sample: list[float] = []
    for block in per_sample_blocks:
        p = _get(block, "logical_precision")
        r = _get(block, "logical_recall")
        per_sample.append(_f1(p, r))

    logical_f1 = _f1(logical_p, logical_r)

    submetrics = {
        "RadFact logical P": logical_p,
        "RadFact logical R": logical_r,
        "RadFact logical F1": logical_f1,
        "RadFact grounding P": grounding_p,
        "RadFact grounding R": grounding_r,
        "RadFact spatial P": spatial_p,
        "RadFact spatial R": spatial_r,
    }

    return RadFactResult(
        per_sample=per_sample,
        aggregate=logical_f1,
        submetrics=submetrics,
        num_llm_failures=num_failures,
        api_mode=False,
    )
