"""01 · The smallest agent: one model, no tools."""

import asyncio

from _common import default_model
from memoflow_agent import Agent


async def main(model=None) -> str:
    agent = Agent(model=default_model(model))
    result = await agent.run("Explain how AI agents work in two sentences.")
    print(result.output)
    return result.output


if __name__ == "__main__":
    asyncio.run(main())
