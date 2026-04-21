"""
GREEN (Generative Radiology Report Evaluation and Error Notation)
reference: Stanford-AIMI/GREEN — https://github.com/Stanford-AIMI/GREEN

Two modes:
  - **Paper default** (no llm_config given): local HuggingFace model via
    the official `green-score` package. Defaults to the paper's
    StanfordAIMI/GREEN-RadLlama2-7b. Requires a GPU. Install:
        pip install omnibin[green]
  - **API mode** (llm_config given): replays GREEN's exact prompt and
    parser against any supported provider (OpenAI, Anthropic, Google,
    OpenRouter, Groq). Useful on GPU-less environments like the Gradio
    Space. Scores may differ from the paper because the judge model is
    not the Stanford-AIMI fine-tune.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..llm_provider import LLMConfig


GREEN_PAPER_DEFAULT_MODEL = "StanfordAIMI/GREEN-radllama2-7b"


@dataclass
class GREENResult:
    per_sample: list[float]
    aggregate: float
    std: float
    summary: str
    submetrics: dict[str, float] = field(default_factory=dict)
    api_mode: bool = False
    num_llm_failures: int = 0


def compute_green(
    references: list[str],
    candidates: list[str],
    llm_config: Optional[LLMConfig] = None,
    model_name: str = GREEN_PAPER_DEFAULT_MODEL,
    output_dir: str = ".",
) -> GREENResult:
    """
    Compute GREEN scores.

    Parameters
    ----------
    references, candidates : list of str
    llm_config : LLMConfig, optional
        If given, uses API mode (verbatim GREEN prompt + parser, non-default
        judge model). If None, uses the official green-score package.
    model_name : str
        HF model id for the local path (ignored in API mode).
    output_dir : str
        Directory for green-score's intermediate outputs (local path only).
    """
    if llm_config is not None:
        from ._green_api import compute_green_via_api
        res = compute_green_via_api(references, candidates, llm_config)
        return GREENResult(
            per_sample=res["per_sample"],
            aggregate=res["aggregate"],
            std=res["std"],
            summary=(
                f"GREEN (API mode) via {llm_config.provider}/{llm_config.model}. "
                "Uses verbatim Stanford-AIMI/GREEN prompt and parser; judge "
                "model differs from paper's fine-tuned RadLlama2-7B."
            ),
            submetrics=res["submetrics"],
            api_mode=True,
            num_llm_failures=res["num_llm_failures"],
        )

    # Paper default: official green-score package (local HF model, GPU).
    try:
        from green_score import GREEN
    except ImportError as e:
        raise ImportError(
            "GREEN requires the official package. Install with:\n"
            "    pip install omnibin[green]\n"
            "or:\n"
            "    pip install green-score\n"
            "Note: GREEN's default model runs a 7B HF model locally and needs a GPU.\n"
            "Alternative: pass `llm_config=LLMConfig(provider=..., ...)` to run in API mode."
        ) from e

    scorer = GREEN(model_name, output_dir=output_dir)
    mean, std, per_sample, summary, df = scorer(references, candidates)

    submetrics: dict[str, float] = {"GREEN (mean)": float(mean), "GREEN (std)": float(std)}
    try:
        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            if col.lower() not in ("index", "sample_id"):
                submetrics[col] = float(df[col].mean())
    except Exception:
        pass

    return GREENResult(
        per_sample=[float(x) for x in per_sample],
        aggregate=float(mean),
        std=float(std),
        summary=str(summary),
        submetrics=submetrics,
        api_mode=False,
        num_llm_failures=0,
    )
