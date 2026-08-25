"""Offline end-to-end validation with a fake LLM - NO API KEY REQUIRED.

    python test_offline.py

This swaps `graph.nodes.get_llm` for a stub, so the classifier's decision is
forced rather than predicted. Everything else is the real code path: the MCP
server really is spawned as a subprocess, tools are really discovered over
JSON-RPC, SQLite is really queried, and the real graph does the real routing.

That split is the useful part. The two things you actually want to be sure of -
"did connectivity chain BOTH the network and the payment lookup?" and "did the
retrieved facts really end up in the generation prompt?" - are deterministic and
provable without spending a token. Only the model's prose is left unverified,
and `python app.py --scenarios` covers that once you have a key.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import graph.nodes as nodes
from graph.nodes import Classification

CAPTURED = {"generate_prompts": []}


class FakeStructured:
    """Stands in for llm.with_structured_output(Classification)."""

    def __init__(self, request_type):
        self.request_type = request_type

    async def ainvoke(self, messages):
        return Classification(
            request_type=self.request_type, reason="fake classifier"
        )


class FakeResponse:
    def __init__(self, text):
        self._t = text

    # A property, mirroring langchain-core 1.x AIMessage.text (it was a method
    # in 0.x). Keeping the fake faithful to the real contract is what makes this
    # test able to catch a regression in how nodes.generate reads the response.
    @property
    def text(self):
        return self._t


class FakeLLM:
    def __init__(self, request_type):
        self.request_type = request_type

    def with_structured_output(self, schema):
        return FakeStructured(self.request_type)

    async def ainvoke(self, messages):
        # Capture the grounded prompt so we can assert the tool JSON is in it.
        CAPTURED["generate_prompts"].append(messages[-1].content)
        return FakeResponse("<fake grounded answer>")


def install_fake(request_type):
    # Both entry points are patched: `generate` uses get_llm, while `classify`
    # goes through get_structured_llm (which picks a per-provider strategy).
    nodes.get_llm = lambda temperature=0.0: FakeLLM(request_type)
    nodes.get_structured_llm = (
        lambda schema, temperature=0.0: FakeStructured(request_type)
    )


from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: E402
from langchain_mcp_adapters.tools import load_mcp_tools  # noqa: E402

from graph.build_graph import build_graph  # noqa: E402

CONN = {
    "telecom": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(ROOT / "mcp_server" / "server.py")],
    }
}

CASES = [
    # (forced classification, customer, query, expected tools, expected facts)
    ("connectivity", "CUST-1002", "Why is my internet not working?",
     ["get_customer_by_id", "get_network_access_status", "get_payment_status"],
     [("get_network_access_status", "network_access", "blocked"),
      ("get_payment_status", "service_suspended_for_nonpayment", True),
      ("get_payment_status", "outstanding_balance", 8450.0)]),
    ("plan", "CUST-1004", "Can I upgrade my plan?",
     ["get_customer_by_id", "get_plan_details"],
     [("get_plan_details", "data_cap", "50 GB"),
      ("get_plan_details", "subscription_tier", "basic")]),
    ("billing", "CUST-1003", "Why was I charged extra this month?",
     ["get_customer_by_id", "get_payment_status", "get_plan_details"],
     [("get_payment_status", "payment_status", "overdue"),
      ("get_plan_details", "data_used", "489 GB")]),
    ("other", "CUST-1001", "What is the weather?",
     ["get_customer_by_id"],
     [("get_customer_by_id", "name", "Amara Perera")]),
    ("plan", "CUST-9999", "Can I upgrade?",
     ["get_customer_by_id", "get_plan_details"],
     [("get_customer_by_id", "found", False)]),
]

failures = []


def check(cond, msg):
    if cond:
        print(f"    PASS  {msg}")
    else:
        print(f"    FAIL  {msg}")
        failures.append(msg)


async def main():
    client = MultiServerMCPClient(CONN)
    async with client.session("telecom") as session:
        tools = await load_mcp_tools(session)
        print(f"MCP tools loaded: {[t.name for t in tools]}\n")
        check(len(tools) == 4, "server exposes 4 tools")

        for rtype, cid, query, want_tools, want_facts in CASES:
            print(f"[{rtype}] {cid} :: {query}")
            install_fake(rtype)
            g = build_graph(tools)
            before = len(CAPTURED["generate_prompts"])
            st = await g.ainvoke({"query": query, "customer_id": cid})

            check(st["request_type"] == rtype, f"routed to {rtype}")
            check(st["planned_tools"] == want_tools,
                  f"planned tools == {want_tools}")
            check(set(st["tool_results"]) == set(want_tools),
                  "every planned tool actually returned data")

            for tool, key, val in want_facts:
                got = st["tool_results"].get(tool, {})
                check(isinstance(got, dict) and got.get(key) == val,
                      f"{tool}.{key} == {val!r} (got {got.get(key)!r})")

            # Parsed to real dicts, not raw MCP content blocks / JSON strings.
            check(all(isinstance(v, dict) for v in st["tool_results"].values()),
                  "tool outputs parsed into dicts")

            check(len(CAPTURED["generate_prompts"]) == before + 1,
                  "generate node ran once")
            prompt = CAPTURED["generate_prompts"][-1]
            check("BACKEND DATA" in prompt, "grounding block present in prompt")
            for tool, key, val in want_facts:
                check(json.dumps(val) in prompt or str(val) in prompt,
                      f"fact {key}={val!r} embedded in grounding prompt")
            check(st.get("answer") == "<fake grounded answer>",
                  "answer written to state")
            print()

        # Connectivity must chain BOTH network and payment - the whole point.
        install_fake("connectivity")
        g = build_graph(tools)
        st = await g.ainvoke(
            {"query": "no internet", "customer_id": "CUST-1002"})
        print("KEY ASSERTION - connectivity multi-tool chain:")
        check("get_network_access_status" in st["tool_results"]
              and "get_payment_status" in st["tool_results"],
              "connectivity chained network AND payment lookups")

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILURES:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


asyncio.run(main())
