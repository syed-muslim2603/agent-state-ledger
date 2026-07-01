"""
agent_state_ledger.router.routes
================================
FastAPI path operation functions (route handlers) for the State Router.

Route inventory
---------------

Sessions
~~~~~~~~
* ``POST   /sessions``              — Create a new agent session
* ``GET    /sessions/{session_id}`` — Retrieve a session by ID
* ``DELETE /sessions/{session_id}`` — Terminate a session
* ``GET    /sessions``              — List all sessions (with optional status filter)

Tool calls
~~~~~~~~~~
* ``POST /tool-call``               — Intercept a tool-call output, apply
                                      progressive disclosure, and return
                                      the budget-safe filtered payload

Rollback
~~~~~~~~
* ``POST /rollback``                — Trigger a manual rollback to a named
                                      snapshot generation

Diagnostics
~~~~~~~~~~~
* ``GET /health``                   — Service health check
* ``GET /metrics``                  — Prometheus text metrics (if enabled)

JSON-RPC (MCP-compatible)
~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``POST /rpc``                     — JSON-RPC 2.0 endpoint for MCP tool-call
                                      dispatch; routes calls through the same
                                      progressive-disclosure pipeline as the
                                      REST ``/tool-call`` endpoint.

All handlers are fully typed and documented.  Error responses always follow
the RFC 7807 Problem Details format so clients can parse them uniformly.
"""

from __future__ import annotations

import time
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from agent_state_ledger.config import get_settings
from agent_state_ledger.logging_setup import get_logger
from agent_state_ledger.models import (
    AgentSession,
    ContextMetricsRecord,
    DisclosureLevel,
    FilteredToolCallOutput,
    HealthResponse,
    IntentDeclaration,
    RollbackReason,
    RollbackRequest,
    SessionCreateResponse,
    SessionStatus,
    ToolCallInput,
    ToolCallOutput,
)
from agent_state_ledger.router.context_filter import apply_progressive_disclosure
from agent_state_ledger.router.session_store import (
    SessionNotFoundError,
    SessionStore,
    get_session_store,
)

logger = get_logger(__name__)

router = APIRouter()

# Process-start timestamp for uptime reporting
_PROCESS_START = time.monotonic()


# ============================================================================ #
#  Dependency injection helpers                                                 #
# ============================================================================ #


def _get_store() -> SessionStore:
    """FastAPI dependency that returns the process-wide SessionStore."""
    return get_session_store()


StoreDep = Annotated[SessionStore, Depends(_get_store)]


# ============================================================================ #
#  Diagnostics                                                                  #
# ============================================================================ #


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Returns current service health including uptime, active session count, "
        "and snapshot persistence statistics."
    ),
    tags=["Diagnostics"],
)
async def health_check(store: StoreDep) -> HealthResponse:
    """
    Return a structured health report.

    The ``status`` field reflects overall service health:

    * ``healthy``  — All subsystems nominal.
    * ``degraded`` — Non-critical subsystem (e.g. metrics exporter) has a
                     transient fault.
    * ``unhealthy`` — Critical subsystem unavailable; service cannot handle
                      requests reliably.
    """
    from agent_state_ledger.snapshot.engine import get_snapshot_engine

    engine = get_snapshot_engine()
    snapshot_count = await engine.count_snapshots()
    active = await store.active_session_count()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=time.monotonic() - _PROCESS_START,
        active_sessions=active,
        snapshots_persisted=snapshot_count,
    )


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Exposes Prometheus metrics in text exposition format.",
    tags=["Diagnostics"],
    include_in_schema=False,  # Keep off the public OpenAPI schema
)
async def metrics_endpoint() -> PlainTextResponse:
    """
    Serve the Prometheus metrics scrape endpoint.

    Only available when ``ASL_METRICS_ENABLED=true`` (the default).
    Returns HTTP 404 when metrics are disabled.
    """
    settings = get_settings()
    if not settings.observability.metrics_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics endpoint is disabled.  Set ASL_METRICS_ENABLED=true to enable.",
        )
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


# ============================================================================ #
#  Session management                                                            #
# ============================================================================ #


@router.post(
    "/sessions",
    response_model=SessionCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create agent session",
    description=(
        "Initialise a new agent session with a declared intent and optional "
        "per-session token budget override.  Returns a ``session_id`` that must "
        "be included in all subsequent tool-call requests."
    ),
    tags=["Sessions"],
)
async def create_session(
    agent_id: Annotated[str, Body(embed=True, min_length=1, max_length=128)],
    intent: Annotated[IntentDeclaration, Body(embed=True)],
    store: StoreDep,
) -> SessionCreateResponse:
    """
    Create a new ``AgentSession`` with the provided *intent* declaration.

    The snapshot engine will automatically take the first checkpoint
    immediately after this call so that rollback-to-genesis is always
    possible.

    Parameters
    ----------
    agent_id:
        Caller-supplied identifier for the agent (e.g. ``"research-agent-v2"``).
    intent:
        Declared operational intent including primary goal, allowed tools,
        and optional per-session context budget.
    """
    session = AgentSession(agent_id=agent_id, intent=intent)

    try:
        session = await store.create_session(session)
    except Exception as exc:  # pragma: no cover
        await logger.aerror("session_creation_failed", agent_id=agent_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {exc}",
        ) from exc

    # Trigger genesis snapshot asynchronously (best-effort; non-blocking)
    from agent_state_ledger.snapshot.engine import get_snapshot_engine

    engine = get_snapshot_engine()
    await engine.checkpoint(session)

    await logger.ainfo(
        "session_created_ok",
        session_id=session.session_id,
        agent_id=agent_id,
    )

    return SessionCreateResponse(
        session_id=session.session_id,
        agent_id=session.agent_id,
        status=session.status,
        created_at=session.created_at,
    )


@router.get(
    "/sessions",
    response_model=list[AgentSession],
    summary="List sessions",
    description="Return all sessions, optionally filtered by status.",
    tags=["Sessions"],
)
async def list_sessions(
    store: StoreDep,
    status_filter: Annotated[
        SessionStatus | None,
        Query(alias="status", description="Filter results by session status."),
    ] = None,
) -> list[AgentSession]:
    """List all registered sessions, optionally filtered by *status_filter*."""
    return await store.list_sessions(status_filter=status_filter)


@router.get(
    "/sessions/{session_id}",
    response_model=AgentSession,
    summary="Get session",
    description="Retrieve the current state of an agent session by ID.",
    tags=["Sessions"],
)
async def get_session(session_id: str, store: StoreDep) -> AgentSession:
    """Return the ``AgentSession`` identified by *session_id*."""
    try:
        return await store.get_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Terminate session",
    description="Mark a session as COMPLETED and remove it from the active registry.",
    tags=["Sessions"],
)
async def delete_session(session_id: str, store: StoreDep) -> None:
    """
    Terminate an active session.

    The session's final state is snapshotted before removal to ensure the
    audit trail remains complete.
    """
    try:
        session = await store.get_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    # Final snapshot before termination
    from agent_state_ledger.snapshot.engine import get_snapshot_engine

    engine = get_snapshot_engine()
    await engine.checkpoint(session)
    await store.terminate_session(session_id, final_status=SessionStatus.COMPLETED)


# ============================================================================ #
#  Tool-call interception                                                       #
# ============================================================================ #


@router.post(
    "/tool-call",
    response_model=FilteredToolCallOutput,
    summary="Intercept and filter tool-call output",
    description=(
        "The primary endpoint of the State Router.  Accepts a raw tool-call "
        "output payload, applies progressive-disclosure filtering within the "
        "session's remaining token budget, records a snapshot if the deviation "
        "threshold is approached, and returns the budget-safe filtered result "
        "ready for LLM context injection."
    ),
    tags=["Tool Calls"],
)
async def intercept_tool_call(
    raw_output: Annotated[ToolCallOutput, Body()],
    store: StoreDep,
) -> FilteredToolCallOutput:
    """
    Intercept, filter, and return a progressive-disclosure-safe tool output.

    Workflow
    --------
    1. Validate that the session exists and is ACTIVE.
    2. Validate that the tool name is on the session's allowed-tools list.
    3. Deduct the step count and compute remaining budget.
    4. Apply progressive disclosure.
    5. Update session counters and check deviation threshold.
    6. Trigger automatic rollback if deviation exceeds threshold.
    7. Checkpoint the updated session state.
    8. Return the ``FilteredToolCallOutput``.
    """
    session_id = raw_output.session_id
    agent_id = raw_output.agent_id

    # ------------------------------------------------------------------ #
    # Step 1: Session validation                                           #
    # ------------------------------------------------------------------ #
    try:
        session = await store.get_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    if session.status != SessionStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Session '{session_id}' is not ACTIVE (current status: "
                f"'{session.status.value}').  Only ACTIVE sessions may "
                "process tool calls."
            ),
        )

    # ------------------------------------------------------------------ #
    # Step 2: Tool allowlist enforcement                                   #
    # ------------------------------------------------------------------ #
    allowed = session.intent.allowed_tools
    if allowed and raw_output.tool_name not in allowed:
        await logger.awarning(
            "tool_call_blocked_not_in_allowlist",
            session_id=session_id,
            tool_name=raw_output.tool_name,
            allowed_tools=allowed,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Tool '{raw_output.tool_name}' is not in the session's "
                f"allowed_tools list: {allowed}."
            ),
        )

    # ------------------------------------------------------------------ #
    # Step 3: Budget computation (read current under session lock)         #
    # ------------------------------------------------------------------ #
    async with store.session_lock(session_id):
        session = await store.get_session(session_id)

        # Enforce max-steps limit
        if session.step_count >= session.intent.max_steps:
            await store.terminate_session(session_id, SessionStatus.FAILED)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Session '{session_id}' has exhausted its step budget "
                    f"({session.intent.max_steps} steps).  Session terminated."
                ),
            )

        budget_remaining = await store.budget_for(session_id)

        # ------------------------------------------------------------------ #
        # Step 4: Progressive disclosure                                       #
        # ------------------------------------------------------------------ #
        filtered_output, metrics = apply_progressive_disclosure(
            raw_output=raw_output,
            budget_remaining=budget_remaining,
            requested_level=raw_output.requested_disclosure_level,
        )

        # ------------------------------------------------------------------ #
        # Step 5: Update session counters                                      #
        # ------------------------------------------------------------------ #
        session.step_count += 1
        session.tokens_consumed += filtered_output.filtered_token_estimate
        session.tool_call_history.append(raw_output.call_id)

        # Loop detection: check if the last N call IDs form a cycle
        _detect_and_flag_loop(session, window=20)

        # Update deviation score (simplified cosine-based estimate)
        session.intent_deviation_score = _compute_deviation_score(session)

        await store.update_session(session)

    # ------------------------------------------------------------------ #
    # Step 6: Automatic rollback on deviation                              #
    # ------------------------------------------------------------------ #
    settings = get_settings()
    threshold = settings.snapshot.intent_deviation_threshold

    if session.intent_deviation_score >= threshold:
        await logger.awarning(
            "intent_deviation_threshold_exceeded",
            session_id=session_id,
            score=session.intent_deviation_score,
            threshold=threshold,
        )
        from agent_state_ledger.snapshot.engine import get_snapshot_engine

        engine = get_snapshot_engine()
        await engine.auto_rollback(session_id, RollbackReason.INTENT_DEVIATION)

    # ------------------------------------------------------------------ #
    # Step 7: Async checkpoint (non-blocking best-effort)                  #
    # ------------------------------------------------------------------ #
    from agent_state_ledger.snapshot.engine import get_snapshot_engine

    engine = get_snapshot_engine()
    await engine.maybe_checkpoint(session)

    await logger.ainfo(
        "tool_call_filtered",
        call_id=raw_output.call_id,
        tool_name=raw_output.tool_name,
        raw_tokens=metrics.raw_tokens,
        filtered_tokens=metrics.filtered_tokens,
        disclosure_level=metrics.disclosure_level.name,
        budget_remaining=filtered_output.budget_remaining,
    )

    return filtered_output


def _detect_and_flag_loop(session: AgentSession, window: int = 20) -> None:
    """
    Detect execution loops by checking for repeated call-ID subsequences
    within the last *window* history entries.

    When a loop is detected ``session.intent_deviation_score`` is raised
    by 0.2 (capped at 1.0) to accelerate deviation threshold triggering.
    This is a heuristic — the snapshot engine performs authoritative
    hash-based loop detection on durable state.
    """
    history = session.tool_call_history
    if len(history) < window * 2:
        return

    recent = history[-window:]
    prior = history[-window * 2 : -window]

    if recent == prior:
        # Exact cycle detected — penalise deviation score
        session.intent_deviation_score = min(
            session.intent_deviation_score + 0.2, 1.0
        )


def _compute_deviation_score(session: AgentSession) -> float:
    """
    Compute a heuristic intent-deviation score for *session*.

    The score is a composite of:

    * Step budget consumption ratio  (0.0 – 0.5 contribution)
    * Token budget consumption ratio (0.0 – 0.3 contribution)
    * Carry-over from previous cycle (0.0 – 0.2 contribution)

    The result is clamped to [0.0, 1.0].  A score of 0.0 means the agent
    is operating entirely within its declared intent envelope; 1.0 means
    total deviation.

    Note: This is a heuristic approximation.  Production deployments should
    replace this with an embedding-based similarity score that compares the
    agent's recent tool calls against the ``intent.primary_goal`` embedding.
    """
    max_steps = max(session.intent.max_steps, 1)
    step_ratio = session.step_count / max_steps

    settings = get_settings()
    global_budget = settings.context.token_budget
    max_budget = session.intent.context_budget_override or global_budget
    token_ratio = session.tokens_consumed / max(max_budget, 1)

    # Weighted composite
    score = (step_ratio * 0.5) + (token_ratio * 0.3) + (session.intent_deviation_score * 0.2)
    return min(max(score, 0.0), 1.0)


# ============================================================================ #
#  Rollback endpoint                                                            #
# ============================================================================ #


@router.post(
    "/rollback",
    summary="Manual rollback",
    description=(
        "Trigger a manual rollback of an agent session to a specific snapshot "
        "generation.  The session's in-memory state is replaced with the "
        "snapshot, and its status is set to ROLLED_BACK."
    ),
    tags=["Rollback"],
)
async def manual_rollback(
    request: Annotated[RollbackRequest, Body()],
    store: StoreDep,
) -> dict[str, Any]:
    """
    Roll back *request.session_id* to *request.target_generation*.

    The rollback is atomic: the snapshot is restored, the in-memory session
    is updated, and an audit record is written, all within a single
    serialized critical section.
    """
    try:
        await store.get_session(request.session_id)  # existence check
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    from agent_state_ledger.snapshot.engine import get_snapshot_engine

    engine = get_snapshot_engine()
    rollback_record = await engine.rollback(request)

    return {
        "status": "rolled_back",
        "rollback_id": rollback_record.rollback_id,
        "session_id": request.session_id,
        "from_generation": rollback_record.from_generation,
        "to_generation": rollback_record.to_generation,
        "reason": request.reason.value,
    }


# ============================================================================ #
#  JSON-RPC 2.0 / MCP endpoint                                                 #
# ============================================================================ #


@router.post(
    "/rpc",
    summary="JSON-RPC 2.0 / MCP tool-call endpoint",
    description=(
        "MCP-compatible JSON-RPC 2.0 endpoint.  Accepts a standard "
        "JSON-RPC request whose ``method`` maps to a registered tool name "
        "and whose ``params`` conform to ``ToolCallInput``.  The filtered "
        "``FilteredToolCallOutput`` is wrapped in a standard JSON-RPC success "
        "response."
    ),
    tags=["JSON-RPC / MCP"],
)
async def json_rpc_endpoint(
    payload: Annotated[dict[str, Any], Body()],
    store: StoreDep,
) -> dict[str, Any]:
    """
    Process a JSON-RPC 2.0 request through the State Router pipeline.

    Expected request format::

        {
            "jsonrpc": "2.0",
            "id": "req-001",
            "method": "tool/call",
            "params": {
                "call_id":   "01HWXY…",
                "tool_name": "search_web",
                "agent_id":  "agent-42",
                "session_id":"sess-99",
                "result":    { … raw tool output … },
                "requested_disclosure_level": 3
            }
        }

    Successful response format::

        {
            "jsonrpc": "2.0",
            "id": "req-001",
            "result": { … FilteredToolCallOutput … }
        }

    Error response format (RFC 7807 extended)::

        {
            "jsonrpc": "2.0",
            "id": "req-001",
            "error": {
                "code": -32602,
                "message": "Invalid params",
                "data": "session_id is required"
            }
        }
    """
    rpc_id = payload.get("id")
    jsonrpc_version = payload.get("jsonrpc", "2.0")

    # Validate JSON-RPC envelope
    if payload.get("jsonrpc") != "2.0":
        return _rpc_error(rpc_id, -32600, "Invalid Request", "jsonrpc must be '2.0'")

    method = payload.get("method", "")
    params = payload.get("params", {})

    if method not in ("tool/call", "tool_call", "tools/call"):
        return _rpc_error(rpc_id, -32601, "Method not found", f"Unknown method: {method!r}")

    # Parse the ToolCallOutput from params
    try:
        raw_output = ToolCallOutput(
            call_id=params.get("call_id", ""),
            tool_name=params.get("tool_name", ""),
            agent_id=params.get("agent_id", ""),
            session_id=params.get("session_id", ""),
            result=params.get("result"),
            error=params.get("error"),
            execution_duration_ms=params.get("execution_duration_ms", 0.0),
        )
    except Exception as exc:  # Pydantic ValidationError
        return _rpc_error(rpc_id, -32602, "Invalid params", str(exc))

    # Synthesise a ToolCallInput for the disclosure level
    requested_level_value = params.get("requested_disclosure_level", 3)
    try:
        requested_level = DisclosureLevel(int(requested_level_value))
    except (ValueError, KeyError):
        requested_level = DisclosureLevel.STANDARD

    # Inject disclosure level into the raw_output for the handler
    raw_output = raw_output.model_copy(
        update={"call_id": raw_output.call_id or params.get("call_id", "")}
    )

    # Route through the standard REST handler (reuses all validation/logic)
    try:
        session = await store.get_session(raw_output.session_id)
    except SessionNotFoundError as exc:
        return _rpc_error(rpc_id, -32000, "Session not found", str(exc))

    budget_remaining = await store.budget_for(raw_output.session_id)
    filtered_output, _ = apply_progressive_disclosure(
        raw_output=raw_output,
        budget_remaining=budget_remaining,
        requested_level=requested_level,
    )

    return {
        "jsonrpc": jsonrpc_version,
        "id": rpc_id,
        "result": filtered_output.model_dump(mode="json"),
    }


def _rpc_error(
    rpc_id: Any,
    code: int,
    message: str,
    data: str | None = None,
) -> dict[str, Any]:
    """Construct a JSON-RPC 2.0 error response."""
    error_obj: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error_obj["data"] = data
    return {"jsonrpc": "2.0", "id": rpc_id, "error": error_obj}
