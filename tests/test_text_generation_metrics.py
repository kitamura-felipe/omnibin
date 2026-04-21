import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from omnibin.text_generation_metrics import generate_text_generation_report
from omnibin.text_generation_utils import (
    TextGenColorScheme,
    validate_text_inputs,
    bootstrap_mean_ci,
)
from omnibin.llm_provider import LLMConfig, SUPPORTED_PROVIDERS, PROVIDER_DEFAULTS


def _fake_litellm_response(text: str) -> MagicMock:
    """Shape a mock response to look like litellm's completion return."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestTextGenerationInputValidation(unittest.TestCase):
    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            validate_text_inputs(["a", "b"], ["c"])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            validate_text_inputs([], [])

    def test_non_string_raises(self):
        with self.assertRaises(TypeError):
            validate_text_inputs(["a"], [123])

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            generate_text_generation_report(
                ["a"], ["b"], metrics=["not_a_metric"],
            )


class TestTextGenerationReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs = [
            "The lungs are clear without focal consolidation.",
            "Right lower lobe pneumonia is present with small pleural effusion.",
            "Cardiomegaly with pulmonary vascular congestion.",
            "No acute cardiopulmonary abnormality.",
        ]
        cls.cands = [
            "Lungs are clear with no consolidation.",
            "Right lower lobe consolidation compatible with pneumonia; small effusion.",
            "Enlarged cardiac silhouette with congested pulmonary vasculature.",
            "Normal chest radiograph.",
        ]
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_lexical_only_report(self):
        out = os.path.join(self.tmp.name, "lex.pdf")
        report = generate_text_generation_report(
            self.refs, self.cands,
            output_path=out,
            metrics=["bleu", "rouge", "meteor"],  # skip bertscore to keep test fast
            n_bootstrap=50,
            dpi=72,
        )
        self.assertTrue(os.path.exists(out))
        self.assertGreater(os.path.getsize(out), 0)
        # Every requested metric should produce an aggregate and per-sample scores.
        self.assertIn("BLEU", report.aggregate_scores)
        self.assertIn("ROUGE", report.aggregate_scores)
        self.assertIn("METEOR", report.aggregate_scores)
        for label in ("BLEU", "ROUGE", "METEOR"):
            self.assertEqual(len(report.per_sample_scores[label]), len(self.refs))
            lo, hi = report.confidence_intervals[label]
            self.assertLessEqual(lo, report.aggregate_scores[label] + 1e-9)
            self.assertGreaterEqual(hi, report.aggregate_scores[label] - 1e-9)
        self.assertEqual(report.metrics_skipped, {})

    def test_default_metrics_are_lexical(self):
        out = os.path.join(self.tmp.name, "defaults.pdf")
        report = generate_text_generation_report(
            self.refs, self.cands,
            output_path=out,
            n_bootstrap=10,
            dpi=72,
        )
        # Default should not require any API key.
        self.assertEqual(report.metrics_skipped, {})
        self.assertTrue(len(report.aggregate_scores) >= 3)

    def test_bootstrap_ci_handles_constant_input(self):
        lo, hi = bootstrap_mean_ci([0.5, 0.5, 0.5, 0.5], n_boot=50)
        self.assertAlmostEqual(lo, 0.5)
        self.assertAlmostEqual(hi, 0.5)


class TestLLMProvider(unittest.TestCase):
    def test_all_supported_providers_have_defaults(self):
        for provider in SUPPORTED_PROVIDERS:
            defaults = PROVIDER_DEFAULTS[provider]
            self.assertIn("env_key", defaults)
            self.assertIn("base_url", defaults)
            self.assertIn("litellm_prefix", defaults)
            self.assertIn("suggested_model", defaults)

    def test_config_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            LLMConfig(provider="pineapple")

    def test_config_applies_defaults(self):
        c = LLMConfig(provider="openai", api_key="dummy")
        self.assertEqual(c.model, PROVIDER_DEFAULTS["openai"]["suggested_model"])
        self.assertEqual(c.base_url, PROVIDER_DEFAULTS["openai"]["base_url"])
        self.assertTrue(c.litellm_model.startswith("openai/"))

    def test_config_require_api_key_without_env(self):
        # Scrub env vars for providers, then require should fail.
        for p in SUPPORTED_PROVIDERS:
            os.environ.pop(PROVIDER_DEFAULTS[p]["env_key"], None)
        c = LLMConfig(provider="anthropic")
        with self.assertRaises(RuntimeError):
            c.require_api_key()

    def test_litellm_model_no_double_prefix(self):
        c = LLMConfig(provider="openrouter", model="openrouter/openai/gpt-4o", api_key="k")
        self.assertEqual(c.litellm_model, "openrouter/openai/gpt-4o")


class TestGREENAPIMode(unittest.TestCase):
    """Exercise GREEN's API-mode scorer with mocked litellm."""

    GREEN_GOOD_RESPONSE = """<|assistant|>
[Explanation]:
Reports mostly align on cardiopulmonary findings.

[Clinically Significant Errors]:
(a) False report of a finding in the candidate: 0.
(b) Missing a finding present in the reference: 1. missed left pleural effusion
(c) Misidentification of a finding's anatomic location/position: 0.
(d) Misassessment of the severity of a finding: 0.
(e) Mentioning a comparison that isn't in the reference: 0.
(f) Omitting a comparison detailing a change from a prior study: 0.

[Clinically Insignificant Errors]:
(a) False report of a finding in the candidate: 0.
(b) Missing a finding present in the reference: 0.
(c) Misidentification of a finding's anatomic location/position: 0.
(d) Misassessment of the severity of a finding: 0.
(e) Mentioning a comparison that isn't in the reference: 0.
(f) Omitting a comparison detailing a change from a prior study: 0.

[Matched Findings]:
3. clear lungs; normal heart size; no pneumothorax
"""

    def test_parser_produces_expected_score(self):
        # matched=3, sig-error-sum=1 → score = 3/(3+1) = 0.75
        from omnibin.judge_metrics._green_api import _score_one_sample

        provider = MagicMock()
        provider.chat.return_value = self.GREEN_GOOD_RESPONSE
        result = _score_one_sample("ref report", "cand report", provider)
        self.assertAlmostEqual(result.score, 0.75)
        self.assertEqual(result.matched, 3)
        self.assertEqual(sum(result.significant_errors), 1)

    def test_green_api_end_to_end_with_mocked_litellm(self):
        from omnibin.judge_metrics.green import compute_green
        cfg = LLMConfig(provider="openai", api_key="dummy")

        with patch("litellm.completion", return_value=_fake_litellm_response(self.GREEN_GOOD_RESPONSE)):
            result = compute_green(
                ["ref 1", "ref 2"], ["cand 1", "cand 2"],
                llm_config=cfg,
            )
        self.assertTrue(result.api_mode)
        self.assertEqual(len(result.per_sample), 2)
        for s in result.per_sample:
            self.assertAlmostEqual(s, 0.75)
        self.assertEqual(result.num_llm_failures, 0)
        # Submetrics include per-subcategory error means.
        self.assertIn("GREEN (mean)", result.submetrics)


class TestRadFactAPIMode(unittest.TestCase):
    """Exercise RadFact's API-mode scorer with mocked litellm."""

    def test_radfact_api_end_to_end_with_mocked_litellm(self):
        from omnibin.judge_metrics.radfact import compute_radfact

        # Each decomposition call yields 2 phrases; each entailment call says entailed.
        decomposition_response = _fake_litellm_response(
            '{"phrases": ["phrase one", "phrase two"]}'
        )
        entailment_response = _fake_litellm_response(
            '{"status": "entailment", "evidence": ["phrase one"]}'
        )

        def side_effect(**kwargs):
            # Distinguish by the system message — decomposition vs entailment.
            messages = kwargs.get("messages", [])
            system = messages[0]["content"] if messages else ""
            if "extract phrases" in system:
                return decomposition_response
            return entailment_response

        cfg = LLMConfig(provider="anthropic", api_key="dummy")
        with patch("litellm.completion", side_effect=side_effect):
            result = compute_radfact(
                ["reference"], ["candidate"],
                llm_config=cfg,
            )
        self.assertTrue(result.api_mode)
        self.assertEqual(len(result.per_sample), 1)
        # All 2 candidate phrases entailed, all 2 reference phrases entailed
        # → precision=recall=1.0, f1=1.0
        self.assertAlmostEqual(result.per_sample[0], 1.0)
        self.assertEqual(result.num_llm_failures, 0)

    def test_radfact_api_handles_malformed_json(self):
        from omnibin.judge_metrics.radfact import compute_radfact

        bad_response = _fake_litellm_response("not even json lol")

        cfg = LLMConfig(provider="google", api_key="dummy")
        with patch("litellm.completion", return_value=bad_response):
            result = compute_radfact(
                ["reference"], ["candidate"],
                llm_config=cfg,
            )
        # Malformed decomposition => failures, no per_sample score
        self.assertTrue(result.api_mode)
        self.assertGreaterEqual(result.num_llm_failures, 1)
        self.assertEqual(result.per_sample, [])


class TestColorSchemes(unittest.TestCase):
    def test_all_schemes_have_required_keys(self):
        required = {"bars", "accent", "distribution", "reference_line", "heatmap_cmap", "metrics_colors"}
        for scheme in TextGenColorScheme:
            self.assertTrue(required.issubset(scheme.value.keys()))


if __name__ == "__main__":
    unittest.main()
