"""Assemble the LangGraph state machine.

Shape of the graph:

                        START
                          |
                      [classify]
                          |
            (conditional edge on request_type)
          /           |            |          \\
   billing_plan  connectivity_plan  plan_plan  other_plan
          \\           |            |          /
                    [call_tools]
                          |
                     [generate]
                          |
                     [respond] --> END

The four `*_plan` nodes all run the same `plan_tools` function. They exist as
separate nodes purely so the branch is VISIBLE - in the LangSmith trace and in
the rendered diagram you can see which route a request took, instead of having
to open the node's output to find out. For a teaching artifact, making control
flow legible is worth one extra line of wiring.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from graph.nodes import classify, generate, make_call_tools_node, plan_tools, respond
from graph.state import SupportState

# request_type -> the node that plans tools for it. Passed straight through as
# the `path_map` of the conditional edge below, so adding a fifth request type
# is a one-line change here plus one entry in nodes.TOOL_PLAN.
ROUTE_TO_NODE = {
    "billing": "plan_billing_tools",
    "connectivity": "plan_connectivity_tools",
    "plan": "plan_plan_tools",
    "other": "plan_other_tools",
}


def route_by_request_type(state: SupportState) -> str:
    """The routing function of the conditional edge.

    A conditional edge is LangGraph's `if/elif` - after `classify` finishes,
    LangGraph calls this with the current state and picks the next node from
    what this function returns. Note that it only *reads* state: routing
    decisions are made from facts an earlier node already wrote, never by doing
    new work here.

    Important detail about the return value: because we pass `path_map` as a
    dict, LangGraph looks the return value up in that dict's KEYS. So this must
    return the routing key ("connectivity"), not the destination node name
    ("plan_connectivity_tools") - returning the node name raises a KeyError at
    runtime. Returning a bare node name is only correct when `path_map` is
    omitted or given as a list.
    """
    request_type = state.get("request_type", "other")
    return request_type if request_type in ROUTE_TO_NODE else "other"


def build_graph(tools: list[BaseTool]):
    """Wire and compile the graph.

    `tools` are the live MCP tools loaded from the running server; they are
    injected into the tool-calling node so nothing in this module needs to know
    about MCP transports.
    """
    # The state schema tells LangGraph what keys exist and how to merge node
    # patches. Ours is a plain TypedDict, so merging is a dict update.
    builder = StateGraph(SupportState)

    builder.add_node("classify", classify)

    # Same callable registered under four names - see the module docstring.
    for node_name in ROUTE_TO_NODE.values():
        builder.add_node(node_name, plan_tools)

    builder.add_node("call_tools", make_call_tools_node(tools))
    builder.add_node("generate", generate)
    builder.add_node("respond", respond)

    builder.add_edge(START, "classify")

    # The conditional edge. `path_map` maps each possible router return value
    # to the node that handles it. Supplying it is optional at runtime but worth
    # it here: LangGraph uses it to draw the four branches in the rendered
    # diagram, and it documents the full set of routes in one place.
    builder.add_conditional_edges(
        "classify",
        route_by_request_type,
        path_map=ROUTE_TO_NODE,
    )

    # All four branches converge - the tool-calling, generation and response
    # steps are identical regardless of route; only the *planned tool list*
    # differs, and that already lives in state by this point.
    for node_name in ROUTE_TO_NODE.values():
        builder.add_edge(node_name, "call_tools")

    builder.add_edge("call_tools", "generate")
    builder.add_edge("generate", "respond")
    builder.add_edge("respond", END)

    # `compile()` validates the topology (no unreachable nodes, no dangling
    # edges) and returns a Runnable - which is why the result supports the same
    # `.ainvoke()` / `.astream()` interface as any other LangChain component,
    # and why LangSmith can trace it without extra instrumentation.
    return builder.compile()
