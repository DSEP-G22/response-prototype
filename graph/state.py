"""LangGraph state schema.

In LangGraph, "state" is a single typed dict that every node receives and
partially updates. A node returns only the keys it changed; LangGraph merges
that patch into the running state and passes it to the next node. This is why
nodes stay small and independently testable - none of them owns the whole
pipeline, they just contribute one field each.

`total=False` means every key is optional, so a node can legitimately return
`{"classification": ...}` alone without having to reconstruct the rest.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

# The four buckets the classifier can produce. `other` is the escape hatch for
# anything the demo doesn't model - important, because forcing an off-topic
# question into "billing" would trigger irrelevant tool calls.
RequestType = Literal["billing", "connectivity", "plan", "other"]


class SupportState(TypedDict, total=False):
    """State passed between nodes of the support graph."""

    # --- Input ---
    query: str
    customer_id: str

    # --- Set by the classify node ---
    request_type: RequestType
    classification_reason: str  # why the classifier chose that bucket (for traces)

    # --- Set by the plan_tools node ---
    # Names of the MCP tools this request needs. Deciding this as *data* (rather
    # than letting the LLM free-call tools) is what makes the multi-tool chain
    # for connectivity auditable: you can assert on this list in a test.
    planned_tools: list[str]

    # --- Set by the call_tools node ---
    # tool name -> parsed JSON payload returned by the MCP server. This dict is
    # the ONLY factual material the generator node is allowed to use.
    tool_results: dict[str, Any]

    # --- Set by the generate node ---
    answer: str
