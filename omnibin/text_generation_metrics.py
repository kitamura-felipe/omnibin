"""
Text generation metrics for radiology reports.

Combines four lexical metrics (BLEU, ROUGE, METEOR, BERTScore) with three
LLM-judge metrics (GREEN, RadFact, CRIMSON), each wrapped around its
original reference implementation.

Public entry point: ``generate_text_generation_report``.

LLM-judge metrics are optional and each requires an extras install:
    pip install omnibin[green]      # GREEN (local GPU, paper default)
    pip install omnibin[radfact]    # RadFact (separate env from Gradio)
    pip install omnibin[crimson]    # CRIMSON
    pip install omnibin[llm-judge]  # litellm provider router

Paper defaults (original judge model per paper):
    - GREEN:   StanfordAIMI/GREEN-RadLlama2-7b  (Ostmeier et al., 2024)
    - RadFact: Llama-3-70B-Instruct             (Bannur et al., 2024)
    - CRIMSON: MedGemmaCRIMSON 4B               (Baharoon et al., 2026)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .llm_provider import LLMConfig
from .text_generation_utils import (
    ALL_METRICS,
    LEXICAL_COMPUTE_FNS,
    LEXICAL_METRICS,
    JUDGE_METRICS,
    TextGenColorScheme,
    bootstrap_mean_ci,
    create_output_directories,
    plot_metric_correlation,
    plot_metric_summary_bars,
    plot_per_sample_distribution,
    plot_per_sample_heatmap,
    plot_submetrics_table,
    validate_text_inputs,
)


@dataclass
class TextGenerationReport:
    """Structured result returned alongside the PDF path."""
    output_path: str
    aggregate_scores: dict[str, float]
    submetrics: dict[str, float]
    per_sample_scores: dict[str, list[float]]
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    metrics_run: list[str] = field(default_factory=list)
    metrics_skipped: dict[str, str] = field(default_factory=dict)


def _resolve_metrics(requested: Optional[list[str]]) -> list[str]:
    if requested is None:
        return list(LEXICAL_METRICS)
    out = []
    for m in requested:
        ml = m.lower()
        if ml not in ALL_METRICS:
            raise ValueError(
                f"Unknown metric '{m}'. Supported: {', '.join(ALL_METRICS)}"
            )
        out.append(ml)
    return out


def generate_text_generation_report(
    references,
    candidates,
    output_path: str = "text_generation_report.pdf",
    metrics: Optional[list[str]] = None,
    llm_config: Optional[LLMConfig] = None,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
    dpi: int = 300,
    color_scheme: TextGenColorScheme = TextGenColorScheme.DEFAULT,
    green_model: Optional[str] = None,
    radfact_narrative: bool = True,
) -> TextGenerationReport:
    """
    Generate a comprehensive text generation report for radiology reports.

    Parameters
    ----------
    references : list of str
        Ground-truth reports.
    candidates : list of str
        Generated reports (same length as ``references``).
    output_path : str
        Path for the output PDF report.
    metrics : list of str, optional
        Subset of ``{"bleu", "rouge", "meteor", "bertscore", "green",
        "radfact", "crimson"}``. Defaults to the lexical metrics.
    llm_config : LLMConfig, optional
        Required for "radfact" or "crimson" when you want to use a specific
        provider. If omitted, each judge metric falls back to its paper
        default (GREEN → local HF model, CRIMSON → local HF model,
        RadFact → its own hydra-configured endpoint).
    n_bootstrap : int
        Bootstrap iterations for 95% CIs on the mean of each metric.
    random_seed : int
    dpi : int
    color_scheme : TextGenColorScheme
    green_model : str, optional
        Override the HF model GREEN loads. Defaults to the paper's
        StanfordAIMI/GREEN-RadLlama2-7b.
    radfact_narrative : bool
        Whether the inputs are narrative text (True) or already decomposed
        into phrase lists (False). Passed through to RadFact.

    Returns
    -------
    TextGenerationReport
    """
    refs, cands = validate_text_inputs(references, candidates)
    wanted = _resolve_metrics(metrics)

    if random_seed is not None:
        np.random.seed(random_seed)
    plt.rcParams["figure.dpi"] = dpi

    colors = color_scheme.value

    aggregate_scores: dict[str, float] = {}
    submetrics: dict[str, float] = {}
    per_sample_scores: dict[str, list[float]] = {}
    confidence_intervals: dict[str, tuple[float, float]] = {}
    metrics_skipped: dict[str, str] = {}

    # --- Lexical metrics -------------------------------------------------
    for metric_name in [m for m in wanted if m in LEXICAL_METRICS]:
        try:
            result = LEXICAL_COMPUTE_FNS[metric_name](refs, cands)
            label = metric_name.upper()
            aggregate_scores[label] = float(result["aggregate"])
            submetrics.update(result.get("submetrics", {}))
            per_sample_scores[label] = [float(s) for s in result["per_sample"]]
            confidence_intervals[label] = bootstrap_mean_ci(
                per_sample_scores[label], n_boot=n_bootstrap,
            )
        except ImportError as e:
            metrics_skipped[metric_name] = str(e)
        except Exception as e:
            metrics_skipped[metric_name] = f"{type(e).__name__}: {e}"

    # --- LLM-judge metrics ----------------------------------------------
    if "green" in wanted:
        try:
            from .judge_metrics.green import compute_green, GREEN_PAPER_DEFAULT_MODEL
            res = compute_green(
                refs, cands,
                llm_config=llm_config,
                model_name=green_model or GREEN_PAPER_DEFAULT_MODEL,
            )
            label = "GREEN (API mode)" if res.api_mode else "GREEN"
            aggregate_scores[label] = res.aggregate
            submetrics.update(res.submetrics)
            per_sample_scores[label] = res.per_sample
            if res.per_sample:
                confidence_intervals[label] = bootstrap_mean_ci(
                    res.per_sample, n_boot=n_bootstrap,
                )
        except Exception as e:
            metrics_skipped["green"] = f"{type(e).__name__}: {e}"

    if "radfact" in wanted:
        try:
            from .judge_metrics.radfact import compute_radfact
            res = compute_radfact(
                refs, cands, llm_config=llm_config,
                is_narrative_text=radfact_narrative,
            )
            label = "RadFact (API mode, logical F1)" if res.api_mode else "RadFact (logical F1)"
            aggregate_scores[label] = res.aggregate
            submetrics.update(res.submetrics)
            if res.per_sample:
                per_sample_scores[label] = res.per_sample
                confidence_intervals[label] = bootstrap_mean_ci(
                    res.per_sample, n_boot=n_bootstrap,
                )
        except Exception as e:
            metrics_skipped["radfact"] = f"{type(e).__name__}: {e}"

    if "crimson" in wanted:
        try:
            from .judge_metrics.crimson import compute_crimson
            res = compute_crimson(refs, cands, llm_config=llm_config)
            label = "CRIMSON (API mode)" if res.api_mode else "CRIMSON"
            aggregate_scores[label] = res.aggregate
            submetrics.update(res.submetrics)
            if res.per_sample:
                per_sample_scores[label] = res.per_sample
                confidence_intervals[label] = bootstrap_mean_ci(
                    res.per_sample, n_boot=n_bootstrap,
                )
        except Exception as e:
            metrics_skipped["crimson"] = f"{type(e).__name__}: {e}"

    # --- PDF -------------------------------------------------------------
    plots_dir = create_output_directories(output_path)

    with PdfPages(output_path) as pdf:
        figs = [
            plot_metric_summary_bars(
                aggregate_scores, confidence_intervals, colors, dpi, plots_dir,
            ),
            plot_per_sample_distribution(per_sample_scores, colors, dpi, plots_dir),
            plot_submetrics_table(submetrics, dpi, plots_dir),
        ]
        corr_fig = plot_metric_correlation(per_sample_scores, colors, dpi, plots_dir)
        if corr_fig is not None:
            figs.append(corr_fig)
        heatmap_fig = plot_per_sample_heatmap(per_sample_scores, colors, dpi, plots_dir)
        if heatmap_fig is not None:
            figs.append(heatmap_fig)

        if metrics_skipped:
            skip_fig = _plot_skipped_table(metrics_skipped, dpi, plots_dir)
            figs.append(skip_fig)

        for f in figs:
            pdf.savefig(f, dpi=dpi)
            plt.close(f)

    return TextGenerationReport(
        output_path=output_path,
        aggregate_scores=aggregate_scores,
        submetrics=submetrics,
        per_sample_scores=per_sample_scores,
        confidence_intervals=confidence_intervals,
        metrics_run=[m for m in wanted if m not in metrics_skipped],
        metrics_skipped=metrics_skipped,
    )


def _plot_skipped_table(skipped: dict[str, str], dpi: int, plots_dir: str) -> plt.Figure:
    import os
    rows = [[name, reason[:120]] for name, reason in skipped.items()]
    fig, ax = plt.subplots(figsize=(10, max(2, 0.4 * len(rows) + 1)), dpi=dpi)
    ax.axis("off")
    ax.set_title("Metrics skipped (install required packages to enable)", pad=12)
    ax.table(
        cellText=rows,
        colLabels=["Metric", "Reason"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "text_gen_skipped.png"), dpi=dpi, bbox_inches="tight")
    return fig
