# Telecom MCP-Grounded Response Prototype

A small, deliberately disposable prototype for learning **LangChain**, **LangGraph**
and **LangSmith** together, using **MCP** as the tool transport.

The idea it demonstrates: before the LLM answers a support question, a graph
decides which backend facts are needed, fetches them through MCP tools, and hands
the model *only those facts* to answer from. The model retrieves; it does not
guess.

## The scenario this exists to prove

A customer asks **"Why is my internet not working?"**

A naive implementation calls one tool — `get_network_access_status` — sees
`network_access: "blocked"`, and replies *"your line appears inactive, please
reboot your router."* That answer is wrong, wastes the customer's time, and
generates a second ticket.

The real cause lives in a different system: the account was **suspended for
non-payment**. So connectivity questions here always chain **two** lookups —
network access **and** payment status — and the graph makes that pairing explicit
rather than hoping the model rediscovers it on every call.

## Architecture

```
              START
                │
           ┌────▼─────┐
           │ classify │  LLM → billing | connectivity | plan | other
           └────┬─────┘
                │  conditional edge on request_type
     ┌──────────┼──────────┬──────────┐
     ▼          ▼          ▼          ▼
  billing  connectivity   plan      other      ← each picks its tool list
     └──────────┴──────────┴──────────┘
                │
         ┌──────▼──────┐
         │ call_tools  │  ── JSON-RPC/stdio ──▶  mcp_server/server.py ──▶ SQLite
         └──────┬──────┘
                │
          ┌─────▼─────┐
          │ generate  │  LLM sees ONLY the fetched facts
          └─────┬─────┘
                │
           ┌────▼────┐
           │ respond │──▶ END
           └─────────┘
```

| Layer | Where | What it does |
|---|---|---|
| Mock backend | `data/`, `mcp_server/db.py` | SQLite with 5 seeded customers |
| MCP server | `mcp_server/server.py` | 4 tools over stdio, via the official `mcp` SDK |
| Orchestration | `graph/` | LangGraph state machine + conditional edges |
| LLM + adapters | `graph/llm.py`, `langchain-mcp-adapters` | Swappable provider; MCP tools → LangChain tools |
| Observability | env vars | LangSmith traces every node and tool call |

### Which tools each request type triggers

| Request type | Tools called | Why |
|---|---|---|
| `connectivity` | customer + **network access** + **payment status** | the suspension case above |
| `billing` | customer + payment status + plan details | extra charges are usually a plan-limit story |
| `plan` | customer + plan details | upgrade advice needs the current tier and cap |
| `other` | customer | enough to greet by name and decline gracefully |

## Setup

Requires **Python 3.11+** (built and verified on 3.13).

```bash
cd response-prototype

python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux

pip install -r requirements.txt
```

Configure the environment:

```bash
copy .env.example .env           # Windows
# cp .env.example .env           # macOS / Linux
```

Then edit `.env`:

```ini
LLM_PROVIDER=anthropic           # or: openai
ANTHROPIC_API_KEY=sk-ant-...     # or OPENAI_API_KEY if you switched provider

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=telecom-response-prototype
```

Seed the mock database (safe to re-run; it wipes and re-seeds):

```bash
python data/seed.py
```

## Run

**The two required scenarios, plus the bonus billing one:**

```bash
python app.py --scenarios
```

**A single query:**

```bash
python app.py --customer CUST-1002 --query "Why is my internet not working?"
python app.py --customer CUST-1004 --query "Can I upgrade my plan?"
```

**Interactive:**

```bash
python app.py
# query> Why is my internet not working?
# query> id CUST-1004          ← switch customer
# query> quit
```

Each run prints the classification, the tools that were called, the raw MCP
payloads, and the final answer — so you can see the grounding for yourself
without opening LangSmith.

**No API key handy?** This runs the whole pipeline with a stubbed model:

```bash
python test_offline.py
```

It spawns the real MCP server, discovers real tools, queries real SQLite and
exercises the real graph — asserting that connectivity chains both lookups and
that retrieved facts land in the generation prompt. Only the model's prose is
faked.

## Seeded customers

| ID | Name | Payment | Network | Plan | Illustrates |
|---|---|---|---|---|---|
| CUST-1001 | Amara Perera | paid | active | FibreMax 300 | happy path — don't invent a problem |
| CUST-1002 | Ravi Fernando | **suspended** | **blocked** | HomeNet 100 | **the headline scenario** |
| CUST-1003 | Nadia Silva | overdue | active | HomeNet 100 | grace period; also near cap |
| CUST-1004 | Dinesh Jayawardena | paid | active | MobileGo 50 | 47/50 GB — upgrade conversation |
| CUST-1005 | Priya Kumar | paid | active | FibreMax 1000 | already top tier — "you can't upgrade" |

Try `CUST-9999` to see the not-found path: every tool returns `found: false`, and
the model is instructed to say so rather than improvise an account.

## What to look at in LangSmith

With `LANGSMITH_TRACING=true`, each `graph.ainvoke()` becomes one trace tree.
Tracing needs no code — LangChain instruments itself when the env vars are
present, which is why `app.py` calls `load_dotenv()` *before* importing anything
else.

Worth clicking through:

- **`classify`** — the structured output, including the `reason` field, which
  exists purely to make the routing decision legible in the trace.
- **`call_tools`** — one child run per MCP tool, with arguments and returned
  payload. On a connectivity query you should see **three**; if you ever see two,
  the chain has regressed.
- **`generate`** — the fully assembled prompt. Every fact in the answer should be
  traceable to the `BACKEND DATA` block; anything else is a hallucination and is
  now visible as one.
- **Latency per node** — the tool calls are local SQLite and near-instant, so the
  wall time is essentially two LLM round-trips.

## Layout

```
response-prototype/
├── app.py                  # CLI entry point; opens the MCP session
├── test_offline.py         # full-pipeline validation, no API key needed
├── requirements.txt        # pinned to verified versions
├── .env.example
├── data/
│   ├── seed.py             # creates + populates the mock DB
│   └── telecom.db          # generated (gitignored)
├── mcp_server/
│   ├── db.py               # SQLite helpers, shared with the seeder
│   └── server.py           # MCP server exposing the 4 tools
└── graph/
    ├── state.py            # LangGraph state schema
    ├── llm.py              # provider-swappable chat model
    ├── nodes.py            # classify / plan_tools / call_tools / generate
    └── build_graph.py      # assembly + conditional edges
```

## Notes and gotchas

- **Versions matter.** This targets LangChain/LangGraph **1.x**. Most tutorials
  online are 0.x and use different shapes. `requirements.txt` is pinned to what
  was actually verified.
- **`path_map` keys are router return values.** With a dict `path_map`, the
  routing function must return the *key* (`"connectivity"`), not the destination
  node name — returning the node name raises `KeyError` at runtime. Cost me a
  test run; see the comment in `build_graph.py`.
- **MCP tool results are content blocks**, roughly
  `[{"type": "text", "text": "<json>"}]`, not plain dicts — the protocol also
  carries images and resources. `_parse_tool_output` in `nodes.py` unwraps them.
- **Never `print()` in `server.py`.** stdout *is* the JSON-RPC channel; use
  stderr for debugging.
- **The tool plan is code, not agent choice.** `TOOL_PLAN` in `nodes.py` is a
  plain dict on purpose — it makes the connectivity chain deterministic and
  assertable. A free-calling agent would be more flexible and less reliable, and
  reliability is the lesson here.
- **LangSmith env vars**: use `LANGSMITH_*`. The old `LANGCHAIN_*` spellings
  still work but are deprecated.

## Scope

Learning artifact, not production. No auth, no retries, no rate limiting, no
migrations, no deployment story. Self-contained in this folder, offline except
for LLM API calls and LangSmith trace upload.
