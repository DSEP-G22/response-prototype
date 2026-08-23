"""The node functions of the support graph.

Each node is `async def node(state) -> dict` - it reads the accumulated state and
returns ONLY the keys it wants to change. LangGraph merges that patch for you.

Every node is also a unit LangSmith traces separately, so a run in the LangSmith
UI reads as: classify -> plan_tools -> call_tools -> generate, each with its own
inputs, outputs and latency. When an answer is wrong you can see immediately
whether the classifier misrouted it or the generator ignored good tool data.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from graph.llm import get_llm
from graph.state import SupportState

# --------------------------------------------------------------------------
# (a) classify
# --------------------------------------------------------------------------


class Classification(BaseModel):
    """Schema the classifier LLM must fill in.

    Passing a Pydantic model to `with_structured_output` makes LangChain enforce
    the shape via the provider's tool-calling API, so we get a validated object
    back instead of prose we would have to regex. `reason` is not used for
    control flow - it exists purely so the LangSmith trace explains the routing
    decision.
    """

    request_type: str = Field(
        description="One of: billing, connectivity, plan, other."
    )
    reason: str = Field(description="One short sentence justifying the choice.")


CLASSIFY_SYSTEM = """You classify inbound telecom customer-support messages.

Choose exactly one request_type:
- billing      : charges, invoices, payments, refunds, unexpected costs.
- connectivity : internet/line/signal not working, slow, dropped, no service.
- plan         : current plan, data cap, upgrades, downgrades, tier changes.
- other        : anything that fits none of the above.

Classify the customer's INTENT, not the words they happen to use. "Why is my
internet off?" is connectivity even if the real cause turns out to be billing -
diagnosing the cause is a later step, not your job."""


async def classify(state: SupportState) -> dict[str, Any]:
    """Route the incoming query into one of four buckets."""
    llm = get_llm().with_structured_output(Classification)
    result: Classification = await llm.ainvoke(
        [
            SystemMessage(content=CLASSIFY_SYSTEM),
            HumanMessage(content=state["query"]),
        ]
    )

    # Defend against a model returning something off-menu; `other` is the safe
    # fallback because it triggers the fewest tool calls.
    request_type = result.request_type.strip().lower()
    if request_type not in {"billing", "connectivity", "plan", "other"}:
        request_type = "other"

    return {"request_type": request_type, "classification_reason": result.reason}


# --------------------------------------------------------------------------
# (b) decide which tools are needed
# --------------------------------------------------------------------------

# The heart of the demo. This mapping is deliberately explicit code rather than
# "let the agent decide", because the connectivity case has a *causal* structure
# the model cannot be trusted to rediscover on every call:
#
#   network says "blocked"  -->  WHY?  -->  billing has the answer
#
# An agent that free-calls tools will often stop after get_network_access_status,
# because that one tool already "answers" the literal question. Encoding the pair
# as a policy makes the chain deterministic and testable.
TOOL_PLAN: dict[str, list[str]] = {
    "connectivity": [
        "get_customer_by_id",
        "get_network_access_status",
        "get_payment_status",  # <- the call a naive implementation skips
    ],
    "billing": [
        "get_customer_by_id",
        "get_payment_status",
        "get_plan_details",  # extra charges are usually a plan-limit story
    ],
    "plan": [
        "get_customer_by_id",
        "get_plan_details",
    ],
    # For `other` we still confirm identity - it lets the model say "Hi Amara,
    # I cannot help with that here" instead of a cold generic refusal.
    "other": [
        "get_customer_by_id",
    ],
}


async def plan_tools(state: SupportState) -> dict[str, Any]:
    """Turn the classification into a concrete list of MCP tool calls."""
    return {"planned_tools": TOOL_PLAN[state["request_type"]]}


# --------------------------------------------------------------------------
# (c) call the MCP tools
# --------------------------------------------------------------------------


def _parse_tool_output(raw: Any) -> Any:
    """Normalise whatever the MCP adapter hands back into plain Python.

    MCP tools return a list of typed content blocks - roughly
    `[{"type": "text", "text": "<json string>"}]` - because the protocol also
    supports image and resource results. For our JSON payloads we want the
    decoded dict, so we unwrap the block and json.loads it. Falling back to the
    raw value keeps this safe if a future adapter version returns dicts directly.
    """
    if isinstance(raw, list):
        texts = [
            b.get("text")
            for b in raw
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        raw = texts[0] if len(texts) == 1 else (texts or raw)
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def make_call_tools_node(tools: list[BaseTool]):
    """Build the tool-calling node, closing over the loaded MCP tools.

    Why a factory: the MCP tools only exist inside a live client session, which
    is opened in `app.py`. Rather than reaching for a global, we inject the tool
    list when the graph is assembled - the node stays a pure function of state
    plus its captured dependency, which keeps it easy to test with fakes.
    """
    by_name = {t.name: t for t in tools}

    async def call_tools(state: SupportState) -> dict[str, Any]:
        customer_id = state["customer_id"]
        results: dict[str, Any] = {}

        # Sequential rather than gathered: the tool count is tiny, and a strictly
        # ordered trace is far easier to read in LangSmith when teaching.
        for name in state["planned_tools"]:
            tool = by_name.get(name)
            if tool is None:
                # Surface the gap in state instead of raising - a missing tool
                # should degrade the answer, not kill the request.
                results[name] = {
                    "error": f"Tool {name!r} not exposed by MCP server."
                }
                continue
            # `ainvoke` on a LangChain tool is what actually crosses the MCP
            # boundary: adapter -> JSON-RPC over stdio -> server.py -> SQLite.
            results[name] = _parse_tool_output(
                await tool.ainvoke({"customer_id": customer_id})
            )

        return {"tool_results": results}

    return call_tools


# --------------------------------------------------------------------------
# (d) grounded answer generation
# --------------------------------------------------------------------------

GENERATE_SYSTEM = """You are a customer-support agent for a telecom company.

GROUNDING RULES - these override everything else:
1. Use ONLY the facts in the BACKEND DATA block below. It is the single source
   of truth about this customer.
2. Never invent account details, amounts, dates, speeds, prices or plan names.
   If a fact is not in the data, say you do not have it and offer to check.
3. If the data shows a cause for a problem, state that cause plainly. Do not
   suggest generic troubleshooting when the backend already explains the issue.
4. If `found` is false, tell the customer the account could not be located.

STYLE: warm, direct, 2-4 sentences. Address the customer by name when known.
Lead with the answer, then the next concrete step. No bullet lists, no preamble.

CONNECTIVITY NOTE: if network access is blocked AND payment is suspended, the
suspension is the cause. Say so, quote the outstanding balance, and explain that
service resumes after payment. Telling such a customer to reboot their router
would be wrong and would waste their time."""


async def generate(state: SupportState) -> dict[str, Any]:
    """Synthesise the customer-facing answer from tool data only.

    The whole grounding strategy is in the prompt construction below: the model
    receives the retrieved facts as a fenced JSON block and is told that block is
    the only admissible evidence. Because the facts arrive *in the prompt* rather
    than from the model's weights, a wrong answer becomes a visible, debuggable
    mismatch between the block and the text - which is exactly what you inspect
    in a LangSmith trace.
    """
    facts = json.dumps(state.get("tool_results", {}), indent=2)

    human = (
        f"CUSTOMER MESSAGE:\n{state['query']}\n\n"
        f"REQUEST TYPE (classified): {state.get('request_type')}\n\n"
        "BACKEND DATA (retrieved via MCP tools - the only facts you may use):\n"
        f"```json\n{facts}\n```\n\n"
        "Write the reply to the customer."
    )

    llm = get_llm()
    response = await llm.ainvoke(
        [
            SystemMessage(content=GENERATE_SYSTEM),
            HumanMessage(content=human),
        ]
    )
    return {"answer": response.text().strip()}


# --------------------------------------------------------------------------
# (e) respond
# --------------------------------------------------------------------------


async def respond(state: SupportState) -> dict[str, Any]:
    """Terminal node.

    It intentionally does no work. Having a single named exit gives every branch
    one place to converge, so the rendered graph shows the fan-out and fan-in
    clearly - and it is the obvious seam for logging, PII redaction or a handoff
    to a human queue later.
    """
    return {}
