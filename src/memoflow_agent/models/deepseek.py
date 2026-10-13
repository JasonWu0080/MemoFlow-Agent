"""DeepSeek provider (OpenAI-compatible Chat Completions API)."""

from __future__ import annotations

from .openai import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek's OpenAI-compatible endpoint.

    Differences from OpenAI handled here: base URL, ``max_tokens`` instead of
    ``max_completion_tokens``, ``reasoning_content`` deltas (hidden, never
    shown as answer text) and ``prompt_cache_hit_tokens`` usage.
    """

    name = "deepseek"
    default_base_url = "https://api.deepseek.com"
    api_key_env = "DEEPSEEK_API_KEY"
    max_tokens_field = "max_tokens"
