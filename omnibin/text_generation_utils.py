"""
Utilities for text generation metrics (radiology report evaluation).

Lexical metrics (BLEU / ROUGE / METEOR / BERTScore) wrap their standard
reference implementations:
  - BLEU: sacrebleu (Post, 2018)
  - ROUGE: google-research/rouge (rouge-score on PyPI)
  - METEOR: nltk.translate.meteor_score
  - BERTScore: Tiiiger/bert_score (bert-score on PyPI)

Each metric is imported lazily so users only pay for what they install.
"""
from __future__ import annotations

import os
from enum import Enum
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# The four lexical metrics we support out of the box. Each entry is the
# name shown in reports / the public API.
LEXICAL_METRICS = ("bleu", "rouge", "meteor", "bertscore")

# The three LLM-judge metrics, wired to their original implementations.
JUDGE_METRICS = ("green", "radfact", "crimson")

ALL_METRICS = LEXICAL_METRICS + JUDGE_METRICS


class TextGenColorScheme(Enum):
    DEFAULT = {
        "bars": "tab:blue",
        "accent": "tab:red",
        "distribution": "tab:blue",
        "reference_line": "gray",
        "heatmap_cmap": "Blues",
        "metrics_colors": [
            "tab:blue", "tab:red", "tab:green",
            "tab:purple", "tab:orange", "tab:brown",
            "tab:pink",
        ],
    }
    MONOCHROME = {
        "bars": "#404040",
        "accent": "#000000",
        "distribution": "#404040",
        "reference_line": "#808080",
        "heatmap_cmap": "Greys",
        "metrics_colors": [
            "#000000", "#303030", "#505050",
            "#707070", "#909090", "#B0B0B0", "#D0D0D0",
        ],
    }
    VIBRANT = {
        "bars": "#4ECDC4",
        "accent": "#FF6B6B",
        "distribution": "#45B7D1",
        "reference_line": "#95A5A6",
        "heatmap_cmap": "Greens",
        "metrics_colors": [
            "#FF6B6B", "#4ECDC4", "#45B7D1",
            "#96CEB4", "#FFEEAD", "#D4A5A5", "#A0CFEC",
        ],
    }


def validate_text_inputs(references: Iterable[str], candidates: Iterable[str]) -> tuple[list[str], list[str]]:
    refs = list(references)
    cands = list(candidates)
    if len(refs) != len(cands):
        raise ValueError(
            f"Length mismatch: {len(refs)} references vs {len(cands)} candidates."
        )
    if len(refs) == 0:
        raise ValueError("At least one reference/candidate pair is required.")
    for i, (r, c) in enumerate(zip(refs, cands)):
        if not isinstance(r, str) or not isinstance(c, str):
            raise TypeError(f"Pair {i}: references and candidates must be strings.")
    return refs, cands


def create_output_directories(output_path: str) -> str:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots") if output_dir else "plots"
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


# ---------------------------------------------------------------------------
# Lexical metrics — each returns a list of per-sample scores plus an
# aggregate. Raises ImportError with a clear message if its dep is missing.
# ---------------------------------------------------------------------------

def compute_bleu(references: list[str], candidates: list[str]) -> dict:
    """Corpus- and sentence-level BLEU via sacrebleu."""
    try:
        import sacrebleu
    except ImportError as e:
        raise ImportError(
            "sacrebleu is required for BLEU. Install with: pip install sacrebleu"
        ) from e

    per_sample = [
        sacrebleu.sentence_bleu(cand, [ref]).score / 100.0
        for ref, cand in zip(references, candidates)
    ]
    corpus = sacrebleu.corpus_bleu(candidates, [references]).score / 100.0
    return {
        "per_sample": per_sample,
        "aggregate": corpus,
        "submetrics": {"BLEU (corpus)": corpus, "BLEU (mean sentence)": float(np.mean(per_sample))},
    }


def compute_rouge(references: list[str], candidates: list[str]) -> dict:
    """ROUGE-1, ROUGE-2, ROUGE-L F-measures via rouge-score."""
    try:
        from rouge_score import rouge_scorer
    except ImportError as e:
        raise ImportError(
            "rouge-score is required for ROUGE. Install with: pip install rouge-score"
        ) from e

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for ref, cand in zip(references, candidates):
        s = scorer.score(ref, cand)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)

    per_sample = rl  # headline per-sample score is ROUGE-L
    return {
        "per_sample": per_sample,
        "aggregate": float(np.mean(rl)),
        "submetrics": {
            "ROUGE-1 F": float(np.mean(r1)),
            "ROUGE-2 F": float(np.mean(r2)),
            "ROUGE-L F": float(np.mean(rl)),
        },
    }


def _ensure_nltk_corpus(name: str, path: str) -> None:
    import nltk
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name, quiet=True)


def compute_meteor(references: list[str], candidates: list[str]) -> dict:
    """Sentence-level METEOR via nltk.translate.meteor_score."""
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
    except ImportError as e:
        raise ImportError(
            "nltk is required for METEOR. Install with: pip install nltk"
        ) from e

    _ensure_nltk_corpus("wordnet", "corpora/wordnet")
    _ensure_nltk_corpus("punkt_tab", "tokenizers/punkt_tab")

    per_sample = []
    for ref, cand in zip(references, candidates):
        ref_tok = nltk.word_tokenize(ref.lower())
        cand_tok = nltk.word_tokenize(cand.lower())
        per_sample.append(meteor_score([ref_tok], cand_tok))
    return {
        "per_sample": per_sample,
        "aggregate": float(np.mean(per_sample)),
        "submetrics": {"METEOR": float(np.mean(per_sample))},
    }


def compute_bertscore(
    references: list[str],
    candidates: list[str],
    model_type: str = "distilbert-base-uncased",
    lang: str = "en",
) -> dict:
    """BERTScore F1 via bert-score."""
    try:
        from bert_score import score as bertscore_fn
    except ImportError as e:
        raise ImportError(
            "bert-score is required for BERTScore. Install with: pip install bert-score"
        ) from e

    P, R, F1 = bertscore_fn(
        candidates, references, model_type=model_type, lang=lang, verbose=False,
    )
    per_sample = F1.tolist()
    return {
        "per_sample": per_sample,
        "aggregate": float(np.mean(per_sample)),
        "submetrics": {
            "BERTScore P": float(P.mean()),
            "BERTScore R": float(R.mean()),
            "BERTScore F1": float(F1.mean()),
        },
    }


LEXICAL_COMPUTE_FNS: dict[str, Callable[..., dict]] = {
    "bleu": compute_bleu,
    "rouge": compute_rouge,
    "meteor": compute_meteor,
    "bertscore": compute_bertscore,
}


# ---------------------------------------------------------------------------
# Bootstrap CIs for per-sample lexical scores
# ---------------------------------------------------------------------------

def bootstrap_mean_ci(values: list[float], n_boot: int = 1000) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    arr = np.asarray(values, dtype=float)
    boot_means = []
    for _ in range(n_boot):
        idx = np.random.choice(len(arr), len(arr), replace=True)
        boot_means.append(arr[idx].mean())
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_metric_summary_bars(
    aggregates: dict[str, float],
    confidence_intervals: dict[str, tuple[float, float]],
    colors: dict,
    dpi: int,
    plots_dir: str,
) -> plt.Figure:
    """Bar chart of aggregate score per metric, with 95% CI error bars."""
    names = list(aggregates.keys())
    values = [aggregates[n] for n in names]
    lows = [confidence_intervals.get(n, (v, v))[0] for n, v in zip(names, values)]
    highs = [confidence_intervals.get(n, (v, v))[1] for n, v in zip(names, values)]
    lower_err = [max(0.0, v - lo) for v, lo in zip(values, lows)]
    upper_err = [max(0.0, hi - v) for v, hi in zip(values, highs)]

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(names)), 6), dpi=dpi)
    palette = colors["metrics_colors"]
    bar_colors = [palette[i % len(palette)] for i in range(len(names))]
    ax.bar(names, values, color=bar_colors, yerr=[lower_err, upper_err], capsize=5)
    ax.set_ylabel("Score")
    ax.set_title("Text Generation Metrics — aggregate scores with 95% CI")
    ax.set_ylim(bottom=0)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "text_gen_metrics_summary.png"), dpi=dpi, bbox_inches="tight")
    return fig


def plot_per_sample_distribution(
    per_sample_scores: dict[str, list[float]],
    colors: dict,
    dpi: int,
    plots_dir: str,
) -> plt.Figure:
    """Violin + strip plot of per-sample scores for each metric."""
    rows = []
    for metric, scores in per_sample_scores.items():
        for s in scores:
            rows.append({"metric": metric, "score": s})
    if not rows:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        ax.text(0.5, 0.5, "No per-sample scores available", ha="center", va="center")
        return fig
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(per_sample_scores)), 6), dpi=dpi)
    sns.violinplot(data=df, x="metric", y="score", inner=None, ax=ax, color=colors["distribution"])
    sns.stripplot(data=df, x="metric", y="score", ax=ax, color=colors["accent"], size=3, alpha=0.6)
    ax.set_title("Per-sample score distribution")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "text_gen_per_sample_distribution.png"), dpi=dpi, bbox_inches="tight")
    return fig


def plot_metric_correlation(
    per_sample_scores: dict[str, list[float]],
    colors: dict,
    dpi: int,
    plots_dir: str,
) -> Optional[plt.Figure]:
    """Pearson-correlation heatmap between per-sample scores of each metric."""
    usable = {k: v for k, v in per_sample_scores.items() if len(v) > 1}
    if len(usable) < 2:
        return None
    # all metrics must have same length (same samples)
    n = len(next(iter(usable.values())))
    usable = {k: v for k, v in usable.items() if len(v) == n}
    if len(usable) < 2:
        return None
    df = pd.DataFrame(usable)
    corr = df.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(usable)), max(5, 0.8 * len(usable))), dpi=dpi)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=colors["heatmap_cmap"], ax=ax, vmin=-1, vmax=1)
    ax.set_title("Pearson correlation between metrics (per-sample)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "text_gen_metric_correlation.png"), dpi=dpi, bbox_inches="tight")
    return fig


def plot_per_sample_heatmap(
    per_sample_scores: dict[str, list[float]],
    colors: dict,
    dpi: int,
    plots_dir: str,
) -> Optional[plt.Figure]:
    """Heatmap of (sample × metric) scores to surface per-pair variability."""
    usable = {k: v for k, v in per_sample_scores.items() if len(v) > 0}
    if not usable:
        return None
    n = len(next(iter(usable.values())))
    usable = {k: v for k, v in usable.items() if len(v) == n}
    if not usable:
        return None
    df = pd.DataFrame(usable)
    df.index = [f"pair-{i + 1:02d}" for i in range(len(df))]
    fig, ax = plt.subplots(
        figsize=(max(6, 0.9 * len(usable)), max(4, 0.35 * len(df))), dpi=dpi,
    )
    sns.heatmap(df, annot=True, fmt=".2f", cmap=colors["heatmap_cmap"], ax=ax)
    ax.set_title("Per-sample scores (pair × metric)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "text_gen_per_sample_heatmap.png"), dpi=dpi, bbox_inches="tight")
    return fig


def plot_submetrics_table(
    submetrics: dict[str, float],
    dpi: int,
    plots_dir: str,
) -> plt.Figure:
    """Render a simple table figure listing all submetric values."""
    rows = [[name, f"{value:.4f}"] for name, value in submetrics.items()]
    fig, ax = plt.subplots(figsize=(8, max(2, 0.35 * len(rows) + 1)), dpi=dpi)
    ax.axis("off")
    ax.table(
        cellText=rows,
        colLabels=["Metric", "Score"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    ax.set_title("Submetric breakdown", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "text_gen_submetrics.png"), dpi=dpi, bbox_inches="tight")
    return fig
