"""MCP server exposing the mock telecom backend as four tools.

MCP (Model Context Protocol) is a transport-level standard: the server declares
tools with JSON schemas, and *any* MCP-aware client can discover and call them
without being compiled against this code. That decoupling is the point - in a
real deployment these tools would be owned by the billing/network teams and this
server would live behind a network boundary, but the client code in `graph/`
wouldn't change at all.

Transport here is stdio: the client launches this file as a subprocess and talks
JSON-RPC over the pipe. Because stdout is the protocol channel, this file must
never `print()` - use stderr if you need to debug.

Run standalone to sanity-check it:  python mcp_server/server.py
(it will sit there waiting for JSON-RPC on stdin, which is correct behaviour)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Same sys.path trick as the seeder: this file is executed directly as a
# subprocess script, not imported as part of a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from mcp_server.db import fetch_customer  # noqa: E402

# FastMCP derives each tool's name, description and input schema from the
# function signature + docstring below. That generated schema is exactly what
# the LLM sees when deciding whether a tool is relevant, so the docstrings here
# are prompt engineering, not just documentation.
mcp = FastMCP("telecom-backend")


def _not_found(customer_id: str) -> dict[str, Any]:
    """Uniform miss payload.

    Every tool returns `found` so the model never has to distinguish "no data"
    from "data that happens to be empty". An explicit false is much harder to
    hallucinate around than a bare `{}`.
    """
    return {"found": False, "customer_id": customer_id,
            "error": "No customer with that ID exists."}


@mcp.tool()
def get_customer_by_id(customer_id: str) -> dict[str, Any]:
    """Look up a customer's identity and account record by their customer ID.

    Use this to confirm who the customer is and whether their account is active.
    Returns name, account status, plan name and subscription tier.
    """
    customer = fetch_customer(customer_id)
    if customer is None:
        return _not_found(customer_id)
    return {
        "found": True,
        "customer_id": customer["customer_id"],
        "name": customer["name"],
        "account_active": customer["account_active"],
        "plan_name": customer["plan_name"],
        "subscription_tier": customer["subscription_tier"],
    }


@mcp.tool()
def get_payment_status(customer_id: str) -> dict[str, Any]:
    """Check a customer's billing state: paid, overdue, or suspended.

    Use this for any billing question, and ALSO for any connectivity problem -
    a suspended account is the most common cause of a line that stopped working.
    Returns payment_status, outstanding_balance and the next renewal date.
    """
    customer = fetch_customer(customer_id)
    if customer is None:
        return _not_found(customer_id)
    return {
        "found": True,
        "customer_id": customer["customer_id"],
        "payment_status": customer["payment_status"],
        "outstanding_balance": customer["outstanding_balance"],
        # A pre-formatted string alongside the raw number. Without it the model
        # copies the bare float into its reply and customers read "8450.0",
        # which looks broken. Formatting is a presentation concern the backend
        # can settle once, rather than something each model has to get right.
        "outstanding_balance_display": f"LKR {customer['outstanding_balance']:,.2f}",
        "renewal_date": customer["renewal_date"],
        # Pre-computing this flag keeps the causal link explicit in the payload
        # rather than asking the model to infer it from two separate fields.
        "service_suspended_for_nonpayment": customer["payment_status"] == "suspended",
    }


@mcp.tool()
def get_network_access_status(customer_id: str) -> dict[str, Any]:
    """Check whether a customer's line currently has network access.

    Returns account_active and the last time the line was seen on the network.
    IMPORTANT: this tool only reports the network symptom. It does not know WHY
    access was cut - always pair it with get_payment_status before explaining an
    outage to a customer.
    """
    customer = fetch_customer(customer_id)
    if customer is None:
        return _not_found(customer_id)
    return {
        "found": True,
        "customer_id": customer["customer_id"],
        "account_active": customer["account_active"],
        "network_access": "active" if customer["account_active"] else "blocked",
        "last_network_access": customer["last_network_access"],
    }


@mcp.tool()
def get_plan_details(customer_id: str) -> dict[str, Any]:
    """Get the customer's current plan, data cap, usage and renewal date.

    Use this for plan/upgrade questions and for billing questions where the
    charge may relate to plan limits. Returns plan_name, subscription_tier,
    data_cap, data_used and renewal_date.
    """
    customer = fetch_customer(customer_id)
    if customer is None:
        return _not_found(customer_id)
    return {
        "found": True,
        "customer_id": customer["customer_id"],
        "plan_name": customer["plan_name"],
        "subscription_tier": customer["subscription_tier"],
        "data_cap": customer["data_cap"],
        "data_used": customer["data_used"],
        "renewal_date": customer["renewal_date"],
    }


if __name__ == "__main__":
    # stdio transport - the LangGraph app spawns this as a child process.
    mcp.run(transport="stdio")
