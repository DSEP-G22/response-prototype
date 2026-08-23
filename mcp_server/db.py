"""SQLite access helpers for the mock telecom backend.

This module is deliberately the *only* place that knows how the mock data is
stored. Both `data/seed.py` (writes) and `mcp_server/server.py` (reads) import
from here, so the MCP tools and the seeder can never drift apart on schema.

Nothing here is telecom-realistic - it is a stand-in for what would really be
four separate internal microservices (CRM, billing, network, catalogue). The
point of the prototype is *how the LLM reaches this data*, not the data itself.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

# The DB lives next to the seed script, in `data/`. Resolving relative to this
# file (rather than the process CWD) matters because the MCP server is launched
# as a subprocess by the LangGraph app and may inherit a different CWD.
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "telecom.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id         TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    payment_status      TEXT NOT NULL,   -- 'paid' | 'overdue' | 'suspended'
    last_network_access TEXT NOT NULL,   -- ISO-8601 timestamp
    account_active      INTEGER NOT NULL,-- SQLite has no bool; 0/1
    plan_name           TEXT NOT NULL,
    data_cap            TEXT NOT NULL,
    data_used           TEXT NOT NULL,
    subscription_tier   TEXT NOT NULL,
    renewal_date        TEXT NOT NULL,   -- ISO-8601 date
    outstanding_balance REAL NOT NULL
);
"""


def connect() -> sqlite3.Connection:
    """Open a connection with row access by column name.

    `row_factory = sqlite3.Row` is what lets us do `dict(row)` below, which in
    turn is what makes the MCP tools return clean structured payloads instead of
    positional tuples the LLM would have to guess the meaning of.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_customer(customer_id: str) -> dict[str, Any] | None:
    """Return one customer as a plain dict, or None if the id is unknown.

    Returning None (rather than raising) keeps the MCP tool layer simple: each
    tool can translate a miss into a structured `{"found": false}` payload, which
    the LLM can then honestly report as "I couldn't find that account" instead of
    inventing one.
    """
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()
    if row is None:
        return None
    record = dict(row)
    # Normalise the SQLite 0/1 back into a real bool so the JSON the MCP tool
    # emits is unambiguous for the model.
    record["account_active"] = bool(record["account_active"])
    return record


def list_customer_ids() -> list[str]:
    """All seeded ids - used by the CLI to show what you can query."""
    with connect() as conn:
        return [r["customer_id"] for r in conn.execute(
            "SELECT customer_id FROM customers ORDER BY customer_id"
        )]
