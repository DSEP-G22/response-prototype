"""Entry point - a small CLI that runs one support query through the graph.

Usage:
    python app.py                                   # interactive loop
    python app.py --customer CUST-1002 --query "Why is my internet not working?"
    python app.py --scenarios                       # run the validation scenarios

What happens on each run:
    1. `.env` is loaded, which is also what turns LangSmith tracing on.
    2. An MCP client session is opened - this SPAWNS `mcp_server/server.py` as a
       child process and speaks JSON-RPC to it over stdio.
    3. The MCP tools are loaded and adapted into LangChain tools.
    4. The graph is compiled with those tools and invoked.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

# Load .env BEFORE importing anything that reads env vars. LangSmith tracing is
# configured entirely through the environment (LANGSMITH_TRACING / _API_KEY /
# _PROJECT), and the tracer is wired up at import time - so a late load_dotenv
# silently produces untraced runs. This ordering is load-bearing.
load_dotenv(PROJECT_ROOT / ".env")

from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: E402
from langchain_mcp_adapters.tools import load_mcp_tools  # noqa: E402

from graph.build_graph import build_graph  # noqa: E402
from mcp_server.db import DB_PATH, list_customer_ids  # noqa: E402

# How the client reaches the server. `transport: stdio` means "run this command
# and talk to its stdin/stdout" - no ports, no network. Using `sys.executable`
# guarantees the child runs in the same virtualenv as the parent.
MCP_CONNECTION = {
    "telecom": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(PROJECT_ROOT / "mcp_server" / "server.py")],
    }
}

# The two scenarios the prototype must demonstrate, plus the bonus billing one.
SCENARIOS = [
    (
        "CUST-1002",
        "Why is my internet not working?",
        "connectivity -> network check AND payment check; must surface the "
        "suspension as the cause",
    ),
    (
        "CUST-1004",
        "Can I upgrade my plan?",
        "plan -> plan lookup; grounded in current tier and data cap",
    ),
    (
        "CUST-1003",
        "Why was I charged extra this month?",
        "bonus: billing -> payment status AND plan details together",
    ),
]


def _check_env() -> None:
    """Fail early with an actionable message rather than deep in a stack trace."""
    if not DB_PATH.exists():
        sys.exit(f"Mock DB not found at {DB_PATH}.\nRun:  python data/seed.py")

    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    if provider == "gemini":
        # Gemini is the only provider here that needs a credential; Ollama talks
        # to localhost.
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            sys.exit(
                "LLM_PROVIDER=gemini but GOOGLE_API_KEY is not set.\n"
                "Copy .env.example to .env and fill it in."
            )
    elif provider == "ollama":
        # Fail here with a clear message rather than deep inside an httpx
        # ConnectError once the graph is already running.
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            import urllib.request

            urllib.request.urlopen(f"{base_url}/api/tags", timeout=5)
        except Exception:
            sys.exit(
                f"LLM_PROVIDER=ollama but no Ollama daemon is reachable at "
                f"{base_url}.\nStart it with:  ollama serve"
            )
    else:
        sys.exit(
            f"Unsupported LLM_PROVIDER={provider!r}. Use 'gemini' or 'ollama'."
        )


def _print_run(state: dict, note: str | None = None) -> None:
    """Render the final graph state so the grounding is inspectable in-terminal.

    This mirrors what you would click through in LangSmith - classification,
    which tools were chosen, what they returned, and the answer built from them.
    Seeing the tool payload next to the answer is the fastest way to verify the
    response is grounded rather than guessed.
    """
    print("\n" + "=" * 74)
    print(f"QUERY        : {state['query']}")
    print(f"CUSTOMER     : {state['customer_id']}")
    if note:
        print(f"EXPECTED     : {note}")
    print("-" * 74)
    print(f"CLASSIFIED   : {state.get('request_type')}")
    print(f"  reason     : {state.get('classification_reason')}")
    print(f"TOOLS CALLED : {', '.join(state.get('planned_tools', []))}")
    print("-" * 74)
    print("MCP TOOL RESULTS (the only facts the answer may use):")
    print(json.dumps(state.get("tool_results", {}), indent=2))
    print("-" * 74)
    print("GROUNDED ANSWER:")
    print(state.get("answer", "<no answer>"))
    print("=" * 74)


async def run_queries(pairs: list[tuple[str, str, str | None]]) -> None:
    """Open one MCP session and run every query through the graph on it.

    The `async with client.session(...)` block holds a single server subprocess
    open for the whole batch. The alternative - `MultiServerMCPClient.get_tools()`
    - re-spawns the server on every call, which works but makes traces noisier
    and wastes a process launch per tool.
    """
    client = MultiServerMCPClient(MCP_CONNECTION)

    async with client.session("telecom") as session:
        # This is the MCP tool adapter doing its job: it calls `tools/list` on
        # the server, reads each tool's JSON schema, and wraps it as a LangChain
        # `BaseTool` with a matching pydantic args model. From here on the graph
        # treats them as ordinary LangChain tools and never sees the protocol.
        tools = await load_mcp_tools(session)
        print(f"Loaded {len(tools)} MCP tools: {', '.join(t.name for t in tools)}")

        graph = build_graph(tools)

        for customer_id, query, note in pairs:
            # Each `ainvoke` is one LangSmith trace: a root run named after the
            # graph, with a child run per node and a nested run per tool call.
            try:
                final_state = await graph.ainvoke(
                    {"query": query, "customer_id": customer_id}
                )
            except Exception as exc:  # noqa: BLE001 - CLI needs a friendly face
                # The most common failure by far is Gemini's free-tier daily
                # quota (20 requests/day, and each query costs 2 calls). It
                # surfaces as a deeply nested 429 inside a LangGraph
                # ExceptionGroup, so the raw traceback buries the one line that
                # matters. Translate it instead of dumping 40 frames.
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    print(
                        "\n[quota] The Gemini API returned 429 - the free tier "
                        "allows 20 requests/day and each query uses 2.\n"
                        "        Switch to Ollama (no daily cap):  "
                        "set LLM_PROVIDER=ollama in .env"
                    )
                    return
                raise
            _print_run(final_state, note)


async def interactive() -> None:
    """Simple REPL: pick a customer id, then ask questions as that customer."""
    ids = list_customer_ids()
    print("\nSeeded customers:", ", ".join(ids))
    print("Type 'quit' to exit, or 'id CUST-XXXX' to switch customer.\n")

    customer_id = ids[0]
    print(f"Acting as {customer_id}")

    while True:
        try:
            query = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            return
        if query.lower().startswith("id "):
            candidate = query[3:].strip().upper()
            if candidate in ids:
                customer_id = candidate
                print(f"Acting as {customer_id}")
            else:
                print(f"Unknown id. Known: {', '.join(ids)}")
            continue

        await run_queries([(customer_id, query, None)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customer", help="Customer ID, e.g. CUST-1002")
    parser.add_argument("--query", help="The support question to ask")
    parser.add_argument(
        "--scenarios",
        action="store_true",
        help="Run the built-in validation scenarios end-to-end",
    )
    args = parser.parse_args()

    _check_env()

    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        print(f"LangSmith tracing ON -> project "
              f"{os.getenv('LANGSMITH_PROJECT', 'default')!r}")
    else:
        print("LangSmith tracing OFF (set LANGSMITH_TRACING=true in .env)")

    if args.scenarios:
        asyncio.run(run_queries(SCENARIOS))
    elif args.query:
        asyncio.run(
            run_queries([(args.customer or "CUST-1001", args.query, None)])
        )
    else:
        asyncio.run(interactive())


if __name__ == "__main__":
    main()
