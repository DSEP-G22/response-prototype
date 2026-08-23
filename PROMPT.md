# Prompt: Telecom MCP-Grounded Response Prototype (LangChain + LangGraph + LangSmith)

> Paste this whole doc into a fresh Claude Code session opened at `d:\DSEP22\response-prototype\` (or point it there). No prior context needed.

## Role & Objective

Act as senior AI engineer. Build minimal but real, working prototype demonstrating **MCP-grounded response generation** for dummy telecom company. Purpose: learning exercise for LangChain, LangGraph, LangSmith — not production. This is throwaway/disposable, decoupled from any other project in this repo. Favor clarity and small scope over completeness.

Core idea: before LLM answers customer support query, it must call MCP tools to fetch real (mock) backend facts — payment status, network/session access, plan/subscription, customer identity — and ground its answer on those facts, not guess.

## Scope Boundary

Everything lives inside `d:\DSEP22\response-prototype\`. Do not touch or reference other folders in this repo (`MVP/`, `ingestion-pipeline-prototype/`, `TriageModel/`, etc.) — this is standalone. No auth, no scaling, no deployment concerns. Optimize for readable code with inline comments explaining *why* — this is a teaching artifact for someone learning these three frameworks.

## Architecture to Build

### 1. Mock data layer (`data/`)
- SQLite DB (preferred) or JSON-seeded store, fully offline, no external services.
- `data/seed.py` — script that creates/populates DB with fake customers. Fields per customer:
  - `customer_id`, `name`, `payment_status` (`paid` / `overdue` / `suspended`), `last_network_access` (timestamp), `account_active` (bool), `plan_name`, `data_cap`, `subscription_tier`, `renewal_date`.
- Seed at least 4-5 customers covering different states (e.g. one paid+active, one overdue+suspended-network, one active but near data cap, etc.) so sample scenarios below actually exercise branching logic.

### 2. MCP server (`mcp_server/`)
- Python, official `mcp` SDK.
- Exposes tools, each reads mock DB:
  - `get_customer_by_id`
  - `get_payment_status`
  - `get_network_access_status`
  - `get_plan_details`
- Keep tool schemas simple (customer_id in, structured dict out).

### 3. LangGraph orchestration (`graph/`)
State machine, nodes:
- (a) **parse/classify** incoming request → determine request type (billing / connectivity / plan question / other).
- (b) **decide tools needed** based on classification.
- (c) **call MCP tool(s)** — via LangChain MCP tool adapters.
- (d) **ground response generation** — LLM synthesizes final answer using only fetched tool data (no hallucinated facts).
- (e) **return final answer**.

Conditional edges branch on request type (billing vs connectivity vs plan). Connectivity questions specifically must chain network-access check **and** payment-status check (suspended-for-nonpayment is the classic case a lazy single-tool-call implementation misses) — make this multi-tool chaining visible/explicit in the graph, since it's the whole point of the demo.

### 4. LangChain
- LLM wrapper (Anthropic or OpenAI, pick one, make provider swappable via `.env`).
- MCP tool adapters bound into graph nodes as LangChain tools.

### 5. LangSmith
- Tracing enabled via env vars (`LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_TRACING=true` or current SDK's equivalent — use latest LangSmith env var conventions, not deprecated `LANGCHAIN_*` names, unless current docs say those are still required).
- Every graph run inspectable: tool calls, intermediate state, latency, per-node breakdown.

## Repo Scaffold

```
response-prototype/
  mcp_server/
    server.py           # MCP server exposing the 4 tools
    db.py                # DB access helpers shared with seed.py
  graph/
    state.py             # LangGraph state schema
    nodes.py              # node functions (classify, plan tools, call tools, generate, respond)
    build_graph.py        # graph assembly + conditional edges
  data/
    seed.py               # creates + populates mock DB
    telecom.db             # generated (gitignored)
  app.py                 # entry point — simple CLI loop (or minimal FastAPI endpoint) to submit a query and print traced result
  requirements.txt
  .env.example            # ANTHROPIC_API_KEY (or OPENAI_API_KEY), LANGSMITH_API_KEY, LANGSMITH_PROJECT, LANGSMITH_TRACING
  README.md                # setup + run steps
  .gitignore
```

## Sample Scenarios to Validate

Implement and manually test at least these two end-to-end:
1. **"Why is my internet not working?"** → classified connectivity → must trigger network-access lookup **and** payment-status lookup before answering → correctly surfaces suspended-for-nonpayment case when applicable.
2. **"Can I upgrade my plan?"** → classified plan question → triggers plan/subscription lookup → answers grounded in current tier/data cap.

Bonus if time allows: a billing question ("why was I charged extra?") exercising payment-status + plan lookup together.

## Constraints

- Keep it minimal — learning-focused, not feature-complete.
- No auth, no scaling, no persistence beyond local SQLite/JSON file.
- Prefer readable code; add inline comments explaining LangGraph/LangChain/LangSmith concepts as they're used (e.g. why a conditional edge, what a MCP tool adapter does, what LangSmith trace captures) — comments should teach, not just describe.
- Use latest LangChain / LangGraph SDK conventions (current `StateGraph` API, current MCP adapter package, current LangSmith env vars) — check installed package versions/docs rather than assuming older API shapes.
- Fully offline except LLM API calls and LangSmith trace upload.

## Explicit Ask

Scaffold full directory structure above. Seed mock data. Implement MCP server + LangGraph orchestration + LangChain tool binding. Wire LangSmith tracing. Write README with setup steps (env vars, install, seed, run) and run instructions for both sample scenarios. Confirm the two sample scenarios work end-to-end before calling it done.
