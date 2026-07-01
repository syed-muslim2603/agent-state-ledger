# Agent State Ledger

> **Production-grade virtual memory controller for multi-agent LLM environments.**
> Prevents context-window bloat and state divergence through progressive disclosure filtering, transactional rollback, and Docker-isolated execution sandboxes.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Architecture Overview](#2-architecture-overview)
3. [Subsystem Deep-Dives](#3-subsystem-deep-dives)
   - 3.1 [State Router (FastAPI / FastMCP)](#31-state-router)
   - 3.2 [Progressive Disclosure Engine](#32-progressive-disclosure-engine)
   - 3.3 [Snapshot Engine](#33-snapshot-engine)
   - 3.4 [Docker Sandbox Wrapper](#34-docker-sandbox-wrapper)
4. [JSON-RPC Schema Handling](#4-json-rpc-schema-handling)
5. [Context Optimisation Metrics](#5-context-optimisation-metrics)
6. [Configuration Reference](#6-configuration-reference)
7. [Step-by-Step Setup](#7-step-by-step-setup)
8. [Running Tests](#8-running-tests)
9. [Docker Deployment](#9-docker-deployment)
10. [Monitoring Dashboard](#10-monitoring-dashboard)
11. [Production Hardening Checklist](#11-production-hardening-checklist)
12. [Project Layout](#12-project-layout)

---

## 1. Problem Statement

Modern multi-agent LLM pipelines suffer from two compounding failure modes:

| Failure Mode | Symptom | Root Cause |
|---|---|---|
| **Context-window bloat** | Agents hallucinate or refuse to continue | Raw tool outputs (search results, code diffs, DB dumps) injected verbatim saturate the context window |
| **State divergence** | Agents loop, contradict themselves, or produce stale outputs | No transactional memory — a crashed agent restarts from scratch or from a corrupted mid-run state |

Agent State Ledger solves both problems in a single, drop-in service layer:

- **Progressive Disclosure** — Tool outputs are intercepted and compressed into budget-safe tiers before reaching the LLM context.
- **Transactional Snapshots** — Every agent state transition is checkpointed to durable storage. If deviation from declared intent is detected, the engine rolls back atomically.
- **Execution Isolation** — Untrusted agent loops run in ephemeral Docker containers with cgroup limits, no network, and a strict seccomp syscall allowlist.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Agent Orchestrator                          │
│  (LangChain / CrewAI / custom agent loop)                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │  POST /v1/tool-call  (raw tool output)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        State Router                                  │
│  FastAPI + FastMCP  •  port 8000                                    │
│                                                                     │
│  ┌──────────────────┐   ┌───────────────────────────────────────┐  │
│  │ Middleware Stack  │   │  Route Handlers                       │  │
│  │  • Trace context  │   │  POST /v1/sessions    (create)        │  │
│  │  • Prometheus     │   │  GET  /v1/sessions    (list/get)      │  │
│  │  • Payload limit  │   │  DELETE /v1/sessions  (terminate)     │  │
│  │  • Rate limiter   │   │  POST /v1/tool-call   (intercept)     │  │
│  └──────────────────┘   │  POST /v1/rollback    (manual)        │  │
│                          │  POST /v1/rpc         (JSON-RPC/MCP)  │  │
│                          │  GET  /v1/health                      │  │
│                          └───────────────────────────────────────┘  │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Progressive Disclosure Engine  (context_filter.py)           │  │
│  │  MINIMAL → CONDENSED → STANDARD → EXPANDED → FULL            │  │
│  │  Token budget enforcement  •  Compression ratio reporting     │  │
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ checkpoint / rollback
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Snapshot Engine                                 │
│  asyncio task  •  aiosqlite  •  WAL journal                         │
│  • Monotonic generation counter per session                         │
│  • xxHash-64 state fingerprinting                                   │
│  • zlib snapshot compression                                        │
│  • Intent-deviation auto-rollback                                   │
│  • Background maintenance: pruning + retention sweeps               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ read-only
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Snapshot Monitor (Rich TUI)                        │
│  asl-monitor  •  port-less  •  reads SQLite in RO mode              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     Docker Sandbox (on-demand)                       │
│  --network none  •  --read-only  •  --cap-drop ALL                  │
│  seccomp allowlist  •  --user 65534  •  cgroup CPU + memory limits  │
│  Entrypoint: /sandbox/entrypoint.py → sandbox_tools.TOOL_REGISTRY   │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow — Tool-Call Interception

```
Agent loop
  │
  │  Raw ToolCallOutput (potentially 50k+ tokens)
  ▼
POST /v1/tool-call
  │
  ├─► Session validation (active? tool in allowlist? step budget?)
  ├─► Budget computation  (global budget − tokens_consumed)
  ├─► Progressive Disclosure (select lowest tier that fits budget)
  ├─► Session counters updated (step_count++, tokens_consumed += filtered_tokens)
  ├─► Loop detection (recent call-ID cycle check)
  ├─► Deviation score recomputed
  ├─► If score ≥ threshold → auto_rollback()
  └─► maybe_checkpoint() → SnapshotEngine
  │
  ▼
FilteredToolCallOutput (budget-safe, compression_ratio < 1.0)
  │
  ▼
LLM context injection
```

---

## 3. Subsystem Deep-Dives

### 3.1 State Router

**File:** `src/agent_state_ledger/router/`

The State Router is a FastAPI application with four middleware layers applied in outermost-first order:

| Layer | Class | Purpose |
|---|---|---|
| GZip | `GZipMiddleware` | Compress large responses (≥ 4 KB) |
| Trace | `TraceContextMiddleware` | Propagate `X-ASL-Trace-ID`; bind to structlog context |
| Metrics | `RequestMetricsMiddleware` | Histogram + counter per (method, route, status) |
| Payload Guard | `PayloadSizeLimitMiddleware` | Reject bodies > `ASL_ROUTER_MAX_PAYLOAD_BYTES` |
| Rate Limit | `AgentRateLimitMiddleware` | Fixed-window 300 req/min per `X-ASL-Agent-ID` |

**Session lifecycle state machine:**

```
INITIALIZING ──► ACTIVE ──► COMPLETED
                   │
                   ├──► SUSPENDED
                   ├──► ROLLED_BACK
                   └──► FAILED
```

Sessions are held in-memory in `SessionStore` (fast authoritative reads) and durably checkpointed to SQLite via the Snapshot Engine. The two stores are kept consistent through a per-session `asyncio.Lock`.

### 3.2 Progressive Disclosure Engine

**File:** `src/agent_state_ledger/router/context_filter.py`

#### Disclosure Tiers

| Tier | Name | Transformation | Target Reduction |
|---|---|---|---|
| 1 | `MINIMAL` | Scalars only; lists → `[N items]`; dicts → `{N keys}` | 75–85% |
| 2 | `CONDENSED` | All top-level keys; lists capped at 3; nested dicts → placeholder | 40–60% |
| 3 | `STANDARD` | Full structure; lists capped at 10; strings truncated at 512 chars | 20–40% |
| 4 | `EXPANDED` | Full structure; lists capped at 50; base64 blobs elided | 5–15% |
| 5 | `FULL` | Verbatim; no transformation applied | 0% |

#### Budget Selection Algorithm

```
given: raw_payload, budget_remaining, requested_level

1. Estimate raw_tokens(raw_payload)
2. If raw_tokens ≤ budget_remaining:
       → apply requested_level transformation (may still compress)
       → return result
3. Else iterate [MINIMAL, CONDENSED, STANDARD, EXPANDED, FULL]:
       candidate = transform(level, raw_payload)
       if estimate_tokens(candidate) ≤ budget_remaining:
           → return candidate at this level
4. Fallback: return FULL (deliver raw; log budget overage)
```

#### Token Counting Methods

| Method | Algorithm | Use Case |
|---|---|---|
| `char_approx` | `ceil(len(text) / 4)` | Default; fast; ± 10% accurate for English |
| `whitespace` | `len(text.split())` | Fastest; less accurate for code/JSON |
| `tiktoken` | `cl100k_base` encoder | Exact GPT-4 / Claude count; requires `tiktoken` |

Set via `ASL_CONTEXT_TOKEN_COUNT_METHOD`.

### 3.3 Snapshot Engine

**File:** `src/agent_state_ledger/snapshot/engine.py`

#### Persistence Schema

**`snapshots` table:**

```sql
CREATE TABLE snapshots (
    snapshot_id           TEXT    NOT NULL,
    session_id            TEXT    NOT NULL,
    agent_id              TEXT    NOT NULL,
    generation            INTEGER NOT NULL,   -- monotonically increasing per session
    session_state_json    BLOB    NOT NULL,   -- zlib-compressed JSON or raw JSON
    state_hash            TEXT    NOT NULL,   -- xxHash-64 hex digest
    intent_deviation_score REAL   NOT NULL,
    is_compressed         INTEGER NOT NULL,
    created_at            TEXT    NOT NULL,
    PRIMARY KEY (session_id, generation)
);
```

**`rollback_audit` table (append-only):**

```sql
CREATE TABLE rollback_audit (
    rollback_id                 TEXT NOT NULL PRIMARY KEY,
    session_id                  TEXT NOT NULL,
    from_generation             INTEGER NOT NULL,
    to_generation               INTEGER NOT NULL,
    reason                      TEXT NOT NULL,   -- RollbackReason enum value
    initiator                   TEXT NOT NULL,   -- "system" | "operator" | "agent"
    deviation_score_at_rollback REAL NOT NULL,
    notes                       TEXT,
    executed_at                 TEXT NOT NULL
);
```

#### Checkpointing

- **Eager checkpoint**: `engine.checkpoint(session)` — always writes immediately. Called on session creation, manual rollback, and session termination.
- **Debounced checkpoint**: `engine.maybe_checkpoint(session)` — writes only when `now - last_checkpoint ≥ checkpoint_interval_seconds`. Otherwise marks session as dirty. Called on every tool-call completion.
- **Background flush**: Every `checkpoint_interval_seconds` the maintenance task flushes all dirty sessions and purges snapshots older than `retention_days`.

#### Intent-Deviation Score

The deviation score is a weighted composite:

```
score = (step_ratio × 0.5) + (token_ratio × 0.3) + (carry_over × 0.2)

where:
  step_ratio   = step_count / max_steps
  token_ratio  = tokens_consumed / context_budget
  carry_over   = previous intent_deviation_score
```

Score ∈ [0.0, 1.0]. When `score ≥ ASL_SNAPSHOT_INTENT_DEVIATION_THRESHOLD` (default `0.35`), `auto_rollback()` fires, finding the highest generation whose stored deviation score is below the threshold.

#### Loop Detection

Two mechanisms run in parallel:

1. **Call-ID cycle check** (router layer): If the last 20 call IDs exactly match the 20 before them, the deviation score is penalised by +0.2.
2. **State hash ring** (engine layer): A rolling window of 50 xxHash-64 digests per session. If the same hash appears more than once, the session is flagged as looping.

### 3.4 Docker Sandbox Wrapper

**File:** `src/agent_state_ledger/sandbox/executor.py`

#### Security Flags

```
docker run
  --rm                               # ephemeral container
  --network none                     # zero network access
  --read-only                        # immutable root filesystem
  --memory 512m                      # hard cgroup memory limit
  --cpus 1.0                         # CPU quota
  --cap-drop ALL                     # all Linux capabilities dropped
  --security-opt no-new-privileges   # block setuid privilege escalation
  --security-opt seccomp=<profile>   # strict allowlist of ~70 syscalls
  --user 65534:65534                 # nobody:nogroup — non-root
  --env ASL_AGENT_PAYLOAD=<b64>      # payload via env var (no shell injection)
```

#### Payload Protocol

The agent payload is base64-encoded to avoid shell-injection attacks when passed via `--env`:

```python
payload = {"tool_name": "compute_hash", "arguments": {"data": "hello", "algorithm": "sha256"}}
b64 = base64.b64encode(json.dumps(payload).encode()).decode()
# → e.g. "eyJ0b29sX25hbWUiOiAiY29tcH..."
```

The container's `entrypoint.py` reads `ASL_AGENT_PAYLOAD`, base64-decodes it, validates the schema, dispatches to `TOOL_REGISTRY[tool_name](**arguments)`, and writes the JSON result to stdout.

#### Timeout Enforcement

```python
try:
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)
except asyncio.TimeoutError:
    await run("docker kill <container_name>")
    raise SandboxTimeoutError(...)
```

The named container is force-killed on timeout so no zombie processes remain.

---

## 4. JSON-RPC Schema Handling

The router exposes a `POST /v1/rpc` endpoint compatible with the **Model Context Protocol (MCP)** JSON-RPC 2.0 transport.

### Request Format

```json
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "method": "tool/call",
  "params": {
    "call_id":   "01HWXY4K9RQMBN3P",
    "tool_name": "search_web",
    "agent_id":  "research-agent-v2",
    "session_id":"01HWXY4K9SB3MN4Q",
    "result": {
      "results": [
        {"title": "FastAPI docs", "url": "https://fastapi.tiangolo.com", "snippet": "..."}
      ]
    },
    "requested_disclosure_level": 3
  }
}
```

### Successful Response

```json
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "result": {
    "call_id": "01HWXY4K9RQMBN3P",
    "tool_name": "search_web",
    "agent_id": "research-agent-v2",
    "session_id": "01HWXY4K9SB3MN4Q",
    "disclosure_level_applied": 3,
    "filtered_result": {
      "results": [
        {"title": "FastAPI docs", "url": "https://fastapi.tiangolo.com", "snippet": "..."},
        "… 0 more"
      ]
    },
    "raw_token_estimate": 312,
    "filtered_token_estimate": 89,
    "compression_ratio": 0.285,
    "budget_remaining": 7103,
    "error": null,
    "timestamp": "2025-01-15T14:22:31.042Z"
  }
}
```

### Error Response (RFC 7807 extended)

```json
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "error": {
    "code": -32000,
    "message": "Session not found",
    "data": "Session '01HWXY4K9SB3MN4Q' not found. It may have been deleted or never created."
  }
}
```

### Standard JSON-RPC Error Codes

| Code | Meaning |
|---|---|
| `-32700` | Parse error — malformed JSON body |
| `-32600` | Invalid Request — `jsonrpc` ≠ `"2.0"` |
| `-32601` | Method not found — unknown `method` value |
| `-32602` | Invalid params — payload validation failure |
| `-32000` | Session not found |

### REST API — `ToolCallOutput` Schema

```json
{
  "call_id":    "string (ULID)",
  "tool_name":  "string",
  "agent_id":   "string",
  "session_id": "string",
  "result":     "any JSON value",
  "error":      "string | null",
  "raw_token_estimate":   0,
  "execution_duration_ms": 0.0,
  "requested_disclosure_level": 3,
  "timestamp":  "ISO-8601 datetime"
}
```

### REST API — `FilteredToolCallOutput` Schema

```json
{
  "call_id":    "string",
  "tool_name":  "string",
  "agent_id":   "string",
  "session_id": "string",
  "disclosure_level_applied": 3,
  "filtered_result": "any JSON value",
  "raw_token_estimate":      312,
  "filtered_token_estimate":  89,
  "compression_ratio":       0.285,
  "budget_remaining":        7103,
  "error":      "string | null",
  "timestamp":  "ISO-8601 datetime"
}
```

---

## 5. Context Optimisation Metrics

Every tool-call interception emits a `ContextMetricsRecord` which is both logged (structlog JSON) and exposed as Prometheus metrics.

### Key Metrics

| Prometheus Metric | Type | Labels | Description |
|---|---|---|---|
| `asl_router_request_duration_seconds` | Histogram | `method`, `path`, `status` | End-to-end HTTP request latency |
| `asl_router_requests_total` | Counter | `method`, `path`, `status` | Total requests handled |
| `asl_router_payload_rejected_total` | Counter | `agent_id` | Requests rejected for oversized payload |

### Structured Log Fields (per tool call)

```json
{
  "event":              "tool_call_filtered",
  "call_id":            "01HWXY…",
  "tool_name":          "search_web",
  "raw_tokens":         312,
  "filtered_tokens":    89,
  "disclosure_level":   "STANDARD",
  "budget_remaining":   7103,
  "compression_ratio":  0.285,
  "trace_id":           "b4a9f3…",
  "session_id":         "01HWXY…",
  "agent_id":           "research-agent-v2",
  "timestamp":          "2025-01-15T14:22:31.042Z",
  "app":                "agent-state-ledger",
  "version":            "1.0.0"
}
```

### Typical Compression Ratios by Tool

| Tool Type | Raw Tokens (typical) | After STANDARD | Compression Ratio |
|---|---|---|---|
| Web search (10 results) | 2 400 | 480 | 0.20 |
| Code file read (500 lines) | 8 000 | 1 600 | 0.20 |
| Database query (100 rows) | 5 000 | 750 | 0.15 |
| Single API response | 400 | 400 | 1.00 (no-op) |
| JSON blob (nested, 3 levels) | 3 200 | 960 | 0.30 |

---

## 6. Configuration Reference

All settings use the `ASL_` prefix and can be set via environment variable or a `.env` file in the working directory.

### Router Settings (`ASL_ROUTER_*`)

| Variable | Default | Description |
|---|---|---|
| `ASL_ROUTER_HOST` | `0.0.0.0` | Bind host |
| `ASL_ROUTER_PORT` | `8000` | TCP port |
| `ASL_ROUTER_WORKERS` | `4` | Uvicorn worker count |
| `ASL_ROUTER_REQUEST_TIMEOUT_SECONDS` | `30.0` | Per-request timeout |
| `ASL_ROUTER_MAX_PAYLOAD_BYTES` | `10485760` (10 MiB) | Maximum body size |
| `ASL_ROUTER_ENABLE_COMPRESSION` | `true` | GZip response compression |
| `ASL_ROUTER_COMPRESSION_MIN_SIZE` | `4096` | Minimum size for GZip |

### Context Settings (`ASL_CONTEXT_*`)

| Variable | Default | Description |
|---|---|---|
| `ASL_CONTEXT_TOKEN_BUDGET` | `8192` | Global per-session token budget |
| `ASL_CONTEXT_DISCLOSURE_LEVELS` | `3` | Number of disclosure tiers |
| `ASL_CONTEXT_SUMMARY_RATIO` | `0.25` | Tier-1 retention fraction |
| `ASL_CONTEXT_TOKEN_COUNT_METHOD` | `char_approx` | `whitespace` \| `char_approx` \| `tiktoken` |

### Snapshot Settings (`ASL_SNAPSHOT_*`)

| Variable | Default | Description |
|---|---|---|
| `ASL_SNAPSHOT_DB_PATH` | `data/snapshots.db` | SQLite file path |
| `ASL_SNAPSHOT_INTENT_DEVIATION_THRESHOLD` | `0.35` | Score that triggers auto-rollback |
| `ASL_SNAPSHOT_MAX_ROLLBACK_DEPTH` | `10` | Generations retained per session |
| `ASL_SNAPSHOT_CHECKPOINT_INTERVAL_SECONDS` | `5.0` | Min interval between checkpoints |
| `ASL_SNAPSHOT_COMPRESSION_ENABLED` | `true` | zlib-compress snapshot blobs |
| `ASL_SNAPSHOT_RETENTION_DAYS` | `7` | Purge snapshots older than N days |

### Sandbox Settings (`ASL_SANDBOX_*`)

| Variable | Default | Description |
|---|---|---|
| `ASL_SANDBOX_IMAGE` | `agent-state-ledger-sandbox:latest` | Docker image |
| `ASL_SANDBOX_CPU_LIMIT` | `1.0` | Fractional CPU quota |
| `ASL_SANDBOX_MEMORY_LIMIT_MB` | `512` | Memory cap in MiB |
| `ASL_SANDBOX_NETWORK_MODE` | `none` | Docker network mode |
| `ASL_SANDBOX_EXECUTION_TIMEOUT_SECONDS` | `60` | Hard kill timeout |
| `ASL_SANDBOX_READ_ONLY_ROOTFS` | `true` | Read-only root filesystem |

### Observability Settings (`ASL_*`)

| Variable | Default | Description |
|---|---|---|
| `ASL_LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `ASL_LOG_FORMAT` | `json` | `json` \| `console` |
| `ASL_METRICS_ENABLED` | `true` | Expose Prometheus `/metrics` |
| `ASL_METRICS_PORT` | `9090` | Prometheus HTTP server port |
| `ASL_TRACE_HEADER` | `X-ASL-Trace-ID` | Distributed trace ID header name |

---

## 7. Step-by-Step Setup

### Prerequisites

| Requirement | Minimum Version | Notes |
|---|---|---|
| Python | 3.11+ | 3.12 recommended |
| Docker | 24.0+ | Required for sandbox execution |
| Docker Compose | 2.20+ | Required for full-stack deployment |
| Git | 2.40+ | For cloning |

### 7.1 Local Development (without Docker)

**Step 1 — Clone the repository**

```bash
git clone https://github.com/your-org/agent-state-ledger.git
cd agent-state-ledger
```

**Step 2 — Create and activate a virtual environment**

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

**Step 3 — Install the package in editable mode with dev dependencies**

```bash
pip install -e ".[dev]"
```

**Step 4 — Create a `.env` file**

```bash
cp .env.example .env
# Edit .env to match your environment
```

**Step 5 — Create the data directory**

```bash
mkdir -p data
```

**Step 6 — Start the State Router**

```bash
asl-router
# or
python -m agent_state_ledger.router.main
```

The router is now serving at `http://localhost:8000`.

**Step 7 — Verify the health endpoint**

```bash
curl http://localhost:8000/v1/health
```

Expected response:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 1.23,
  "active_sessions": 0,
  "snapshots_persisted": 0
}
```

**Step 8 (optional) — Start the snapshot monitor**

```bash
# In a separate terminal
asl-monitor
```

### 7.2 Creating Your First Session

```bash
curl -s -X POST http://localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "my-research-agent",
    "intent": {
      "primary_goal": "Search for the latest papers on LLM context management and summarise the top 5.",
      "allowed_tools": ["search_web", "read_url_content", "summarise"],
      "max_steps": 50,
      "context_budget_override": 8192
    }
  }' | python -m json.tool
```

Note the `session_id` from the response.

### 7.3 Intercepting a Tool-Call Output

```bash
curl -s -X POST http://localhost:8000/v1/tool-call \
  -H "Content-Type: application/json" \
  -d '{
    "call_id": "call-001",
    "tool_name": "search_web",
    "agent_id": "my-research-agent",
    "session_id": "<YOUR_SESSION_ID>",
    "result": {
      "results": [
        {"title": "Attention is All You Need", "url": "https://arxiv.org/abs/1706.03762", "snippet": "We propose a new simple network architecture..."},
        {"title": "Long Context LLMs", "url": "https://example.com/paper", "snippet": "Recent advances in extending context windows..."}
      ]
    },
    "requested_disclosure_level": 3
  }' | python -m json.tool
```

The response includes `compression_ratio` and `filtered_result`.

### 7.4 Triggering a Manual Rollback

```bash
curl -s -X POST http://localhost:8000/v1/rollback \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<YOUR_SESSION_ID>",
    "target_generation": 0,
    "reason": "manual",
    "initiator": "operator",
    "notes": "Resetting to genesis state for re-planning."
  }' | python -m json.tool
```

---

## 8. Running Tests

```bash
# Run the full test suite with coverage
pytest

# Run a specific test file
pytest tests/test_context_filter.py -v

# Run only snapshot engine tests
pytest tests/test_snapshot_engine.py -v

# Run API integration tests
pytest tests/test_router_api.py -v

# Generate HTML coverage report
pytest --cov-report=html
open htmlcov/index.html
```

Expected output:

```
======================== test session starts =========================
platform linux -- Python 3.12.x
tests/test_context_filter.py        ......................    22 passed
tests/test_snapshot_engine.py       ................         16 passed
tests/test_router_api.py            .....................    21 passed
======================== 59 passed in 4.82s  ========================

---------- coverage: platform linux, python 3.12 -----------
Name                                           Stmts   Miss  Cover
------------------------------------------------------------------
src/agent_state_ledger/config.py                  82      3    96%
src/agent_state_ledger/router/context_filter.py  121      4    97%
src/agent_state_ledger/router/routes.py          148      9    94%
src/agent_state_ledger/snapshot/engine.py        182     11    94%
src/agent_state_ledger/snapshot/store.py         163      8    95%
------------------------------------------------------------------
TOTAL                                            696     35    95%
```

---

## 9. Docker Deployment

### 9.1 Build all images

```bash
# Build router image
docker build -t agent-state-ledger-router:latest .

# Build sandbox image
docker build -f sandbox/Dockerfile -t agent-state-ledger-sandbox:latest .
```

### 9.2 Start the full stack

```bash
docker compose up --build
```

Services started:

| Service | Port | Description |
|---|---|---|
| `router` | `8000` | FastAPI State Router |
| `monitor` | — | Rich TUI (stdout logs) |
| `prometheus` | `9091` | Metrics scraper |

### 9.3 Override configuration at runtime

```bash
ASL_CONTEXT_TOKEN_BUDGET=16384 \
ASL_SNAPSHOT_INTENT_DEVIATION_THRESHOLD=0.25 \
ASL_LOG_LEVEL=DEBUG \
docker compose up
```

### 9.4 Scaling the router

```bash
docker compose up --scale router=3
```

> **Note:** For multi-node deployments, replace the in-memory `SessionStore` with a Redis-backed implementation. The `SnapshotStore` (SQLite) should be replaced with PostgreSQL for concurrent writes at scale.

---

## 10. Monitoring Dashboard

### Terminal Dashboard

```bash
asl-monitor
```

Opens a live Rich TUI that refreshes every 2 seconds:

```
⬡ AGENT STATE LEDGER  |  Snapshot Monitor  |  2025-01-15 14:22:31 UTC  |  Uptime: 00:01:45
|  Total Snapshots: 42  |  Rollbacks: 1

┌─ Active Sessions ──────────────────────────────────────────────────────────────────────┐
│ Session ID           │ Agent ID         │ Latest Gen │ Snapshots │ Deviation │ Last Snap│
│ 01HWXY4K9SB3…        │ research-agent   │ 8          │ 8         │ 🟢 0.142  │ 14:22:28 │
│ 01HWXY4K9XP1…        │ coding-agent     │ 3          │ 3         │ 🟡 0.380  │ 14:22:15 │
└────────────────────────────────────────────────────────────────────────────────────────┘

┌─ Recent Rollback Events ───────────────────────────────────────────────────────────────┐
│ Session ID  │ From │ To │ Reason           │ Initiator │ Deviation │ Executed At       │
│ 01HWXY4K9…  │  5   │ 2  │ intent_deviation  │ system    │ 0.391     │ 2025-01-15 14:21  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Prometheus + Grafana

After `docker compose up`, Prometheus is available at `http://localhost:9091`.

Import the community FastAPI dashboard (ID: `16110`) into Grafana and add the ASL custom metrics.

---

## 11. Production Hardening Checklist

- [ ] Replace in-memory `SessionStore` with a Redis cluster for multi-node deployments
- [ ] Replace SQLite `SnapshotStore` with PostgreSQL + `asyncpg` for concurrent writers
- [ ] Set `ASL_ROUTER_WORKERS` to `2 × CPU_cores + 1` (Uvicorn recommendation)
- [ ] Mount the snapshot volume on a high-IOPS SSD or NVMe block device
- [ ] Configure TLS termination at the load balancer (Nginx / Caddy) in front of port 8000
- [ ] Set `ASL_SANDBOX_NETWORK_MODE=none` in all production environments (default)
- [ ] Restrict Docker socket access — use a Docker-in-Docker sidecar or Kubernetes Jobs instead of host socket mount
- [ ] Set `ASL_SNAPSHOT_RETENTION_DAYS` based on compliance requirements
- [ ] Enable Prometheus alerting for: `deviation_score > 0.7`, `budget_remaining < 512`, `rollback_count > 5/hour`
- [ ] Pin the sandbox image digest (`image@sha256:…`) rather than using `:latest`
- [ ] Run `pip-audit` and `trivy image` on every release build
- [ ] Enable structured log forwarding to your SIEM (Datadog, Splunk, Loki)

---

## 12. Project Layout

```
agent-state-ledger/
├── Dockerfile                        # Router service multi-stage image
├── docker-compose.yml                # Full stack: router + monitor + prometheus
├── prometheus.yml                    # Prometheus scrape configuration
├── pyproject.toml                    # Package metadata, deps, tool config
├── .env.example                      # Template for environment configuration
├── .gitignore
├── README.md
│
├── sandbox/
│   ├── Dockerfile                    # Hardened sandbox image (distroless-style)
│   ├── entrypoint.py                 # Container entrypoint (payload → tool dispatch)
│   └── sandbox_tools.py              # Tool registry (TOOL_REGISTRY dict)
│
├── src/
│   └── agent_state_ledger/
│       ├── __init__.py               # Package metadata (__version__ etc.)
│       ├── config.py                 # Pydantic-Settings — all subsystem settings
│       ├── logging_setup.py          # Structlog configuration (JSON / console)
│       ├── models.py                 # All shared Pydantic v2 models
│       │
│       ├── router/
│       │   ├── __init__.py
│       │   ├── main.py               # FastAPI app factory + lifespan + Uvicorn entry
│       │   ├── routes.py             # All route handlers (REST + JSON-RPC)
│       │   ├── middleware.py         # 4 middleware layers
│       │   ├── context_filter.py     # Progressive disclosure engine
│       │   └── session_store.py      # In-memory session registry with async locks
│       │
│       ├── snapshot/
│       │   ├── __init__.py
│       │   ├── engine.py             # SnapshotEngine (checkpoint / rollback / loop detect)
│       │   ├── store.py              # SQLite async persistence layer
│       │   └── monitor.py            # Rich TUI dashboard + asl-monitor CLI
│       │
│       └── sandbox/
│           ├── __init__.py
│           └── executor.py           # SandboxExecutor (Docker lifecycle management)
│
└── tests/
    ├── __init__.py
    ├── conftest.py                   # Shared fixtures (stores, engine, async client)
    ├── test_context_filter.py        # 22 unit tests for disclosure engine
    ├── test_snapshot_engine.py       # 16 integration tests for store + engine
    └── test_router_api.py            # 21 end-to-end HTTP API tests
```

---

## License

MIT © Agent State Ledger Contributors
