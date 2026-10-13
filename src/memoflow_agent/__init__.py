"""MemoFlow Agent — a lightweight, extensible Python AI agent framework.

Small Core. Reliable Agents.

    from memoflow_agent import Agent, tool

    @tool
    def add(a: int, b: int) -> int:
        "Add two integers."
        return a + b

    agent = Agent(model="openai:<model-id>", tools=[add])
    result = agent.run_sync("What is 2 + 40?")
    print(result.output)
"""

from ._version import __version__
from .agent import Agent, RunResult
from .budget import AgentBudget, BudgetLimits
from .decisions import DecisionType, ToolCall
from .events import Event, EventType
from .exceptions import (
    BudgetExceeded,
    CapacityExhausted,
    ConfigurationError,
    ContextTooLong,
    InvalidResume,
    MemoFlowAgentError,
    ModelProviderError,
    RunNotFound,
    RunNotSuspended,
)
from .loop import AgentLoop, Application, CancellationRegistry, Failure
from .models import ModelCapabilities, ModelProvider, ModelRequest, ModelTurn, resolve_model
from .persistence.memory import InMemoryStore
from .policy import (
    Allow,
    AllowAll,
    CapabilityPolicy,
    Deny,
    GrantDuration,
    InMemoryGrantStore,
    Principal,
    RequireAuthorization,
    RequireConfirmation,
)
from .resume import Resume
from .state import RunState, RunStatus, StopReason, WaitingReason
from .tools import RetryPolicy, Risk, SideEffect, ToolContext, ToolError, ToolResult, ToolSpec, tool

__all__ = [
    "Agent",
    "AgentBudget",
    "AgentLoop",
    "Allow",
    "AllowAll",
    "Application",
    "BudgetExceeded",
    "BudgetLimits",
    "CancellationRegistry",
    "CapabilityPolicy",
    "CapacityExhausted",
    "ConfigurationError",
    "ContextTooLong",
    "DecisionType",
    "Deny",
    "Event",
    "EventType",
    "Failure",
    "GrantDuration",
    "InMemoryGrantStore",
    "InMemoryStore",
    "InvalidResume",
    "MemoFlowAgentError",
    "ModelCapabilities",
    "ModelProvider",
    "ModelProviderError",
    "ModelRequest",
    "ModelTurn",
    "Principal",
    "RequireAuthorization",
    "RequireConfirmation",
    "Resume",
    "RetryPolicy",
    "Risk",
    "RunNotFound",
    "RunNotSuspended",
    "RunResult",
    "RunState",
    "RunStatus",
    "SideEffect",
    "StopReason",
    "ToolCall",
    "ToolContext",
    "ToolError",
    "ToolResult",
    "ToolSpec",
    "WaitingReason",
    "__version__",
    "resolve_model",
    "tool",
]
