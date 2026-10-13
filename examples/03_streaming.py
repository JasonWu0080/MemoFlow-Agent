"""03 · Streaming: text deltas and run events as they happen."""

import asyncio

from _common import default_model
from memoflow_agent import Agent


async def main(model=None) -> list[str]:
    agent = Agent(model=default_model(model))
    seen = []
    async for event in agent.stream("Write a haiku about durable execution."):
        seen.append(event.type)
        if event.type == "model.delta":
            print(event.text, end="", flush=True)
        elif event.type in {"run.completed", "run.failed"}:
            print(f"\n[{event.type}]")
    return seen


if __name__ == "__main__":
    asyncio.run(main())
