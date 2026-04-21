"""
Unified LLM provider interface for text generation metrics.

Routes chat completion requests to OpenAI, Anthropic, Google (Gemini),
OpenRouter, or Groq through litellm, which all expose an OpenAI-compatible
interface. Used by CRIMSON, RadFact and any in-house LLM-judge prompts.

litellm is an optional dependency. Install with:
    pip install omnibin[llm-judge]
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional


SUPPORTED_PROVIDERS = ("openai", "anthropic", "google", "openrouter", "groq")


# Per-provider defaults. `env_key` is the environment variable the provider's
# official SDK reads by default. `base_url` is the OpenAI-compatible endpoint
# some downstream libraries (CRIMSON, langchain) need when pointed at a
# non-OpenAI provider.
PROVIDER_DEFAULTS: dict[str, dict[str, Optional[str]]] = {
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "litellm_prefix": "openai",
        "suggested_model": "gpt-4o",
    },
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com/v1",
        "litellm_prefix": "anthropic",
        "suggested_model": "claude-sonnet-4-6",
    },
    "google": {
        "env_key": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "litellm_prefix": "gemini",
        "suggested_model": "gemini-2.5-pro",
    },
    "openrouter": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "litellm_prefix": "openrouter",
        "suggested_model": "openai/gpt-4o",
    },
    "groq": {
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "litellm_prefix": "groq",
        "suggested_model": "llama-3.3-70b-versatile",
    },
}


@dataclass
class LLMConfig:
    """
    Configuration for LLM-judge metrics.

    Parameters
    ----------
    provider : str
        One of "openai", "anthropic", "google", "openrouter", "groq".
    model : str
        Model identifier as the provider expects it (e.g. "gpt-4o",
        "claude-sonnet-4-6", "gemini-2.5-pro"). If omitted a sensible
        per-provider default is used.
    api_key : str, optional
        API key. If not given, the provider's standard environment variable
        is read (see PROVIDER_DEFAULTS[provider]["env_key"]).
    base_url : str, optional
        Override the OpenAI-compatible base URL (useful for self-hosted
        proxies or a litellm router).
    temperature : float
        Sampling temperature. Judge metrics should generally be 0.0.
    max_tokens : int
        Maximum tokens in the response.
    extra : dict
        Additional kwargs forwarded to litellm.completion.
    """
    provider: str
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unknown provider '{self.provider}'. "
                f"Supported: {', '.join(SUPPORTED_PROVIDERS)}"
            )
        defaults = PROVIDER_DEFAULTS[self.provider]
        if self.model is None:
            self.model = defaults["suggested_model"]
        if self.api_key is None:
            self.api_key = os.environ.get(defaults["env_key"])
        if self.base_url is None:
            self.base_url = defaults["base_url"]

    @property
    def litellm_model(self) -> str:
        """Model string in litellm's prefixed form (e.g. 'anthropic/claude-sonnet-4-6')."""
        prefix = PROVIDER_DEFAULTS[self.provider]["litellm_prefix"]
        if self.model.startswith(prefix + "/"):
            return self.model
        return f"{prefix}/{self.model}"

    def require_api_key(self) -> str:
        if not self.api_key:
            env_key = PROVIDER_DEFAULTS[self.provider]["env_key"]
            raise RuntimeError(
                f"No API key for provider '{self.provider}'. "
                f"Set {env_key} or pass api_key explicitly."
            )
        return self.api_key


class LLMProvider:
    """
    Minimal chat wrapper built on litellm.

    All five providers expose OpenAI-compatible chat completion through
    litellm. This class exists so the judge-metric wrappers don't need to
    know which provider is in use.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        try:
            import litellm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "litellm is required for LLM-judge metrics. "
                "Install with: pip install omnibin[llm-judge]"
            ) from e

    def chat(self, messages: list[dict[str, str]], **overrides: Any) -> str:
        """
        Send chat messages and return the assistant's text response.

        Parameters
        ----------
        messages : list of {"role": ..., "content": ...}
            Standard chat messages.
        overrides : dict
            Per-call overrides for temperature / max_tokens / etc.
        """
        import litellm

        api_key = self.config.require_api_key()
        params = {
            "model": self.config.litellm_model,
            "messages": messages,
            "temperature": overrides.pop("temperature", self.config.temperature),
            "max_tokens": overrides.pop("max_tokens", self.config.max_tokens),
            "api_key": api_key,
            **self.config.extra,
            **overrides,
        }
        if self.config.base_url:
            params["api_base"] = self.config.base_url

        response = litellm.completion(**params)
        return response.choices[0].message.content or ""
