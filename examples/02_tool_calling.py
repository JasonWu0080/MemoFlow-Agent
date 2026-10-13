"""02 · Native tool calling with a typed Python function."""

import ast
import asyncio
import operator

from _common import default_model
from memoflow_agent import Agent, ToolError, tool

_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
              ast.Pow: operator.pow, ast.USub: operator.neg}


def _evaluate(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ToolError("unsupported_expression", "Only + - * / ** and numbers are supported.")


@tool(risk="low")
def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression such as "17 * 23 + 4"."""
    return _evaluate(ast.parse(expression, mode="eval").body)


async def main(model=None) -> str:
    agent = Agent(model=default_model(model), tools=[calculate])
    result = await agent.run("What is 17 * 23 + 4? Use the calculator.")
    for event in result.events:
        if event.type == "tool.completed":
            print("tool:", event.payload["tool"], event.payload["status"])
    print(result.output)
    return result.output


if __name__ == "__main__":
    asyncio.run(main())
