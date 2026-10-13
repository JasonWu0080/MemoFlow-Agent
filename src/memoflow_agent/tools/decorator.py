"""``@tool``: turn a typed Python function into a :class:`ToolSpec`.

    @tool(risk="low")
    async def calculate(expression: str) -> dict:
        "Evaluate an arithmetic expression."
        ...

The input schema is generated from the signature with pydantic; unknown
arguments are rejected. A parameter named ``ctx`` (or annotated with
:class:`ToolContext`) receives the tool context and is not part of the schema.
The docstring's first paragraph is the description unless one is given.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import Field, create_model

from .base import RetryPolicy, Risk, SideEffect, StrictInput, ToolContext, ToolSpec


def _context_parameter(fn: Callable, hints: dict[str, Any]) -> str | None:
    for name in inspect.signature(fn).parameters:
        if name == "ctx" or hints.get(name) is ToolContext:
            return name
    return None


def _description(fn: Callable) -> str:
    doc = inspect.getdoc(fn) or ""
    return doc.split("\n\n", 1)[0].strip()


def tool(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    risk: Risk | str = Risk.LOW,
    side_effect: SideEffect | str = SideEffect.READ_ONLY,
    permission: str | None = None,
    requires_approval: bool = False,
    timeout: float = 30.0,
    retry: RetryPolicy | None = None,
    public_fields: tuple[str, ...] | None = None,
    audit_fields: tuple[str, ...] = (),
    interactive_only: bool = False,
) -> Any:
    """Decorator (usable bare or with arguments) producing a :class:`ToolSpec`."""

    def build(function: Callable) -> ToolSpec:
        hints = get_type_hints(function)
        ctx_name = _context_parameter(function, hints)
        fields: dict[str, Any] = {}
        for parameter in inspect.signature(function).parameters.values():
            if parameter.name == ctx_name:
                continue
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                raise TypeError(f"tool {function.__name__}: *args/**kwargs are not allowed")
            annotation = hints.get(parameter.name, Any)
            default = ... if parameter.default is inspect.Parameter.empty else parameter.default
            fields[parameter.name] = (annotation, Field(default) if default is not ... else ...)
        tool_name = name or function.__name__
        input_model = create_model(f"{tool_name}_input", __base__=StrictInput, **fields)
        argument_names = list(fields)

        def bind(ctx: Any, arguments: Any) -> dict[str, Any]:
            kwargs = {key: getattr(arguments, key) for key in argument_names}
            if ctx_name is not None:
                kwargs[ctx_name] = ctx
            return kwargs

        handler: Callable[[Any, Any], Any]
        if inspect.iscoroutinefunction(function):

            async def async_handler(ctx: Any, arguments: Any) -> Any:
                return await function(**bind(ctx, arguments))

            handler = async_handler
        else:

            def sync_handler(ctx: Any, arguments: Any) -> Any:
                return function(**bind(ctx, arguments))

            handler = sync_handler

        handler.__wrapped__ = function  # type: ignore[union-attr]
        side = SideEffect(side_effect)
        return ToolSpec(
            name=tool_name,
            description=description or _description(function) or tool_name,
            input_model=input_model,
            handler=handler,
            risk=Risk(risk),
            side_effect=side,
            timeout_seconds=timeout,
            retry=retry or RetryPolicy(),
            permission=permission,
            requires_approval=requires_approval,
            interactive_only=interactive_only,
            audit_fields=audit_fields,
            public_fields=tuple(argument_names) if public_fields is None else public_fields,
        )

    if fn is not None:
        return build(fn)
    return build
