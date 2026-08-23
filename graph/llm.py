"""LangChain chat-model factory with a provider switch.

`init_chat_model` is LangChain's provider-agnostic constructor: give it a model
id and a provider name and you get back a `BaseChatModel` with the same
interface regardless of vendor. That uniform interface is the reason the node
code in `nodes.py` never mentions Anthropic or OpenAI - swapping `LLM_PROVIDER`
in `.env` changes the vendor without touching the graph.
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

# Sensible defaults per provider so `.env` only needs LLM_PROVIDER + an API key.
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o-mini",
}


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """Build the chat model named by the environment.

    temperature=0 by default: this is a grounding demo, and we want the model to
    restate the retrieved facts, not to be creative about them.
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(
            f"Unsupported LLM_PROVIDER={provider!r}. "
            f"Choose one of: {', '.join(DEFAULT_MODELS)}"
        )

    key_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    if not os.getenv(key_var):
        raise RuntimeError(
            f"LLM_PROVIDER={provider} but {key_var} is not set. "
            "Copy .env.example to .env and fill it in."
        )

    model = os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider]
    return init_chat_model(model, model_provider=provider, temperature=temperature)
