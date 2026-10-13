"""Tool contract, ``@tool`` decorator and the single execution pipeline."""

from .base import (
    RetryPolicy,
    Risk,
    SideEffect,
    StrictInput,
    ToolContext,
    ToolError,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    coerce_result,
)
from .decorator import tool
from .executor import ExecOutcome, ExecutionHooks, MappedError, ToolExecutor

__all__ = [
    "ExecOutcome",
    "ExecutionHooks",
    "MappedError",
    "RetryPolicy",
    "Risk",
    "SideEffect",
    "StrictInput",
    "ToolContext",
    "ToolError",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "coerce_result",
    "tool",
]
