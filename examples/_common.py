"""Shared by the examples: which model to use.

Set MEMOFLOW_AGENT_MODEL to "provider:model-id" (openai, anthropic or
deepseek) and the provider's API key (OPENAI_API_KEY, ANTHROPIC_API_KEY,
DEEPSEEK_API_KEY). No model id is hard-coded here on purpose.
"""

import os
import sys


def default_model(model=None):
    if model is not None:
        return model
    name = os.getenv("MEMOFLOW_AGENT_MODEL")
    if not name:
        sys.exit('Set MEMOFLOW_AGENT_MODEL, e.g. MEMOFLOW_AGENT_MODEL="openai:<model-id>", plus the provider API key.')
    return name
