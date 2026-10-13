"""04 · The same agent on different providers.

Set any of MEMOFLOW_AGENT_OPENAI_MODEL, MEMOFLOW_AGENT_ANTHROPIC_MODEL,
MEMOFLOW_AGENT_DEEPSEEK_MODEL (model ids) with the matching API key.
"""

import asyncio
import os

from memoflow_agent import Agent, tool


@tool
def word_count(text: str) -> int:
    "Count the words in a text."
    return len(text.split())


def configured_models() -> list[str]:
    models = []
    for provider in ("openai", "anthropic", "deepseek"):
        model_id = os.getenv(f"MEMOFLOW_AGENT_{provider.upper()}_MODEL")
        if model_id:
            models.append(f"{provider}:{model_id}")
    return models


async def main(models=None) -> dict:
    answers = {}
    for model in models if models is not None else configured_models():
        agent = Agent(model=model, tools=[word_count])
        result = await agent.run('How many words are in "small core reliable agents"? Use the tool.')
        label = model if isinstance(model, str) else getattr(model, "name", "model")
        answers[label] = result.output
        print(f"{label}: {result.output}")
    if not answers:
        print("No provider configured; see the module docstring.")
    return answers


if __name__ == "__main__":
    asyncio.run(main())
