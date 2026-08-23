"""Create and populate the mock telecom SQLite DB.

Run once before starting the app:  python data/seed.py

The five fixtures below are chosen so the graph's conditional edges actually
have something to branch on. In particular CUST-1002 is the scenario the whole
demo exists to prove: the network is down *because* billing suspended it. A
naive implementation that only calls `get_network_access_status` would answer
"your line is inactive, try rebooting your router" - which is wrong and would
generate a second support ticket. Grounding on two tools fixes that.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow `python data/seed.py` from the project root without installing a package:
# add the project root to sys.path so `mcp_server.db` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_server.db import DB_PATH, SCHEMA, connect  # noqa: E402

NOW = datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


# (customer_id, name, payment_status, last_network_access, account_active,
#  plan_name, data_cap, data_used, subscription_tier, renewal_date, balance)
CUSTOMERS = [
    (
        # Happy path: everything fine. Answers here should be reassuring and
        # should NOT invent a problem just because the customer asked a question.
        "CUST-1001", "Amara Perera", "paid", _iso(NOW - timedelta(minutes=8)), 1,
        "FibreMax 300", "Unlimited", "412 GB", "premium",
        (NOW + timedelta(days=21)).date().isoformat(), 0.0,
    ),
    (
        # The headline scenario: network access stopped 3 days ago AND the
        # account is suspended for non-payment. Connectivity questions must
        # surface the billing cause, not just the symptom.
        "CUST-1002", "Ravi Fernando", "suspended", _iso(NOW - timedelta(days=3)), 0,
        "HomeNet 100", "500 GB", "137 GB", "standard",
        (NOW + timedelta(days=9)).date().isoformat(), 8450.00,
    ),
    (
        # Overdue but not yet cut off - a grace-period case. Connectivity is
        # technically fine; the honest answer is "working now, but pay soon".
        "CUST-1003", "Nadia Silva", "overdue", _iso(NOW - timedelta(minutes=2)), 1,
        "HomeNet 100", "500 GB", "489 GB", "standard",
        (NOW + timedelta(days=4)).date().isoformat(), 3200.00,
    ),
    (
        # Near the data cap on a mid tier - the natural upgrade conversation.
        "CUST-1004", "Dinesh Jayawardena", "paid", _iso(NOW - timedelta(minutes=45)), 1,
        "MobileGo 50", "50 GB", "47 GB", "basic",
        (NOW + timedelta(days=12)).date().isoformat(), 0.0,
    ),
    (
        # Top tier, already at the ceiling of the catalogue - the correct answer
        # to "can I upgrade?" is "you're already on the highest plan".
        "CUST-1005", "Priya Kumar", "paid", _iso(NOW - timedelta(hours=2)), 1,
        "FibreMax 1000", "Unlimited", "1.2 TB", "platinum",
        (NOW + timedelta(days=27)).date().isoformat(), 0.0,
    ),
]


def seed() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)
        # Idempotent: wipe and re-insert so re-running the seeder is always safe
        # and always produces timestamps relative to "now".
        conn.execute("DELETE FROM customers")
        conn.executemany(
            "INSERT INTO customers VALUES (?,?,?,?,?,?,?,?,?,?,?)", CUSTOMERS
        )
    print(f"Seeded {len(CUSTOMERS)} customers into {DB_PATH}")
    for c in CUSTOMERS:
        print(f"  {c[0]}  {c[1]:<22} payment={c[2]:<9} active={bool(c[4])}  plan={c[5]}")


if __name__ == "__main__":
    seed()
