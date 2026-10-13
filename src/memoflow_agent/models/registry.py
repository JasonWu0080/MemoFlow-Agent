"""``"provider:model"`` strings → provider instances.

    resolve_model("openai:<model-id>")
    resolve_model("anthropic:<model-id>")
    resolve_model("deepseek:<model-id>")

No model list is hard-coded: whatever id follows the colon is sent to the
provider as is. Additional providers register a factory.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..exceptions import ConfigurationError
from .base import ModelProvider

_FACTORIES: dict[str, Callable[..., ModelProvider]] = {}


def register_provider(prefix: str, factory: Callable[..., ModelProvider]) -> None:
    _FACTORIES[prefix.lower()] = factory


def _builtin() -> None:
    from .anthropic import AnthropicProvider
    from .deepseek import DeepSeekProvider
    from .openai import OpenAIProvider

    _FACTORIES.setdefault("openai", OpenAIProvider)
    _FACTORIES.setdefault("anthropic", AnthropicProvider)
    _FACTORIES.setdefault("deepseek", DeepSeekProvider)


def resolve_model(model: str | ModelProvider, **options: Any) -> ModelProvider:
    if not isinstance(model, str):
        return model
    _builtin()
    prefix, sep, model_id = model.partition(":")
    if not sep or not model_id:
        raise ConfigurationError(f"model must look like 'provider:model-id', got {model!r}")
    factory = _FACTORIES.get(prefix.lower())
    if factory is None:
        raise ConfigurationError(f"unknown model provider {prefix!r}; known: {sorted(_FACTORIES)}")
    return factory(model_id, **options)
