"""Exceptions raised by the kernel.

Every exception a caller may want to branch on lives here, so applications
can import them without importing the modules that raise them.
"""

from __future__ import annotations


class MemoFlowAgentError(Exception):
    """Base class for errors raised by the framework."""


class ConfigurationError(MemoFlowAgentError, ValueError):
    """The agent was configured with something that cannot work.

    Raised at construction time — for example tools on a model whose declared
    capabilities do not include tool calling — rather than half way through a
    run.
    """


class ModelProviderError(MemoFlowAgentError, RuntimeError):
    """A model provider call failed.

    ``code`` is a short, stable identifier: ``rate_limit``, ``upstream``,
    ``timeout``, ``network``, ``auth``, ``bad_request``, ``context_length`` or
    a provider-specific value. Retry policies key off it.
    """

    def __init__(self, message: str = "", *, code: str = "upstream", status_code: int | None = None):
        super().__init__(message or code)
        self.code = code
        self.status_code = status_code


class ContextTooLong(MemoFlowAgentError, RuntimeError):
    """The request does not fit the model's input budget even after compaction."""

    def __init__(self, required: int, budget: int):
        super().__init__(f"context requires {required} tokens, budget {budget}")
        self.required = required
        self.budget = budget


class BudgetExceeded(MemoFlowAgentError, RuntimeError):
    """A run guard tripped (model turns, tool calls, tokens, time, failures)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class CapacityExhausted(MemoFlowAgentError, RuntimeError):
    """The application refused another model call (e.g. no money left).

    Unlike :class:`BudgetExceeded`, this comes from the application, not the
    kernel's counters. The run ends with what it has, labelled partial.
    """

    def __init__(self, reason: str = "capacity"):
        super().__init__(reason)
        self.reason = reason


class LeaseLost(MemoFlowAgentError, RuntimeError):
    """Another worker owns this run now; stop without writing anything."""


class RunNotFound(MemoFlowAgentError, LookupError):
    """No run with that id (for this principal)."""


class RunNotSuspended(MemoFlowAgentError, RuntimeError):
    """A resume arrived for a run that is not waiting — e.g. a double tap."""


class InvalidResume(MemoFlowAgentError, ValueError):
    """A resume does not answer what the run is waiting for, or is malformed."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code


class CheckpointIncompatible(MemoFlowAgentError, RuntimeError):
    """A checkpoint was written by an incompatible schema version."""
