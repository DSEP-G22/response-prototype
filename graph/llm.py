"""LangChain chat-model factory with a provider switch.

Two providers are wired up, chosen by `LLM_PROVIDER` in `.env`:

  gemini  - Google's Gemini, reached through its OpenAI-COMPATIBLE endpoint.
            Google exposes a `/v1beta/openai` path that speaks the same
            chat-completions wire format as OpenAI, so `ChatOpenAI` drives it
            unchanged - you only repoint `base_url` and swap the key. That is
            why there is no `langchain-google-*` dependency here.

  ollama  - models served by a local Ollama daemon. Tags ending in `-cloud`
            execute on Ollama's hosted infrastructure (needs `ollama signin`,
            downloads nothing); tags without it use a locally pulled model.
            Either way the endpoint is localhost, so no API key is involved.

Both return a `BaseChatModel`. That shared interface is the whole point: nothing
in `nodes.py` mentions a vendor, so switching providers is an `.env` edit rather
than a code change.

A caveat worth knowing while learning this: structured output (used by the
classifier node) relies on the provider's tool-calling support. Gemini and the
larger Ollama models handle it well; very small local models often do not, and
will fail or return junk. That is a model-capability limit, not a LangChain bug.
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

# Defaults are chosen so a fresh `.env` only needs LLM_PROVIDER plus, for
# Gemini, a key. Both are overridable per provider.
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_OLLAMA_MODEL = "gpt-oss:20b-cloud"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

SUPPORTED = ("gemini", "ollama")


def _build_gemini(temperature: float) -> BaseChatModel:
    """Gemini through its OpenAI-compatible surface."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LLM_PROVIDER=gemini but neither GOOGLE_API_KEY nor GEMINI_API_KEY "
            "is set. Copy .env.example to .env and fill it in."
        )

    return ChatOpenAI(
        model=os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
        # Repointing base_url is the entire trick - same client, same protocol,
        # different host.
        base_url=os.getenv("GEMINI_BASE_URL") or DEFAULT_GEMINI_BASE_URL,
        api_key=api_key,
        temperature=temperature,
    )


def _build_ollama(temperature: float) -> BaseChatModel:
    """A model served by the local Ollama daemon (or its cloud tags)."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL,
        base_url=os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL,
        temperature=temperature,
        # Ollama defaults to a small context window; the grounding prompt
        # carries several JSON tool payloads, so give it room.
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
    )


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """Build the chat model named by the environment.

    temperature=0 by default: this is a grounding demo, and we want the model to
    restate the retrieved facts faithfully, not to be creative about them.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    if provider == "gemini":
        return _build_gemini(temperature)
    if provider == "ollama":
        return _build_ollama(temperature)

    raise ValueError(
        f"Unsupported LLM_PROVIDER={provider!r}. Choose one of: {', '.join(SUPPORTED)}"
    )


def get_structured_llm(schema, temperature: float = 0.0):
    """Return a model constrained to emit `schema`, using the best method per provider.

    Why this wrapper exists - a genuinely useful thing to know when learning
    LangChain: `with_structured_output` supports several strategies, and the
    default is not equally reliable across providers.

      - `json_schema` (ChatOllama's default) asks the model to emit JSON matching
        the schema. Ollama models frequently answer with a bare value instead -
        `connectivity` rather than `{"request_type": "connectivity", ...}` -
        which blows up as `OutputParserException: Invalid json output`.
      - `function_calling` presents the schema as a tool the model must call.
        The tool-call path is much better constrained, and it is what actually
        works here.

    Gemini's default already goes through tool-calling, so it is left alone.
    Keeping this decision in the provider layer means `nodes.py` never has to
    know which vendor it is talking to.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    llm = get_llm(temperature)

    if provider == "ollama":
        return llm.with_structured_output(schema, method="function_calling")
    return llm.with_structured_output(schema)
