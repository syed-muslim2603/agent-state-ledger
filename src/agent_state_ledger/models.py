"""
agent_state_ledger.models
=========================
Canonical Pydantic v2 data-models shared across every subsystem of the
Agent State Ledger.  Defining models centrally in a single module ensures
that JSON-RPC wire types, internal state representations, and API response
schemas are always consistent and version-controlled together.

All models derive from ``BaseModel`` with ``model_config = ConfigDict(frozen=True)``
(where immutability is appropriate) to prevent accidental mutation after
validation.  Mutable models — e.g. ``AgentSession`` which is updated in
place inside the snapshot engine — use ``ConfigDict(frozen=False)`` explicitly.

Naming conventions
------------------
* ``*Request``  — Incoming JSON-RPC / HTTP request bodies.
* ``*Response`` — Outgoing JSON-RPC / HTTP response envelopes.
* ``*Record``   — Internal data objects persisted to the snapshot store.
* ``*Metrics``  — Structured metrics payloads emitted to Prometheus /
                  the monitoring dashboard.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================================ #
#  Shared primitives                                                            #
# ============================================================================ #


def _utcnow() -> datetime:
    """Return the current UTC datetime with timezone info."""
    return datetime.now(tz=timezone.utc)


def _new_ulid() -> str:
    """Generate a new ULID string for use as a lexicographically sortable ID."""
    try:
        # python-ulid >= 2.x
        import ulid  # lazy import to keep module load fast
        return str(ulid.ULID())
    except AttributeError:
        # Fallback: some versions expose it differently
        import ulid
        return str(ulid.new())


# ============================================================================ #
#  Enumerated types                                                             #
# ============================================================================ #


class DisclosureLevel(int, Enum):
    """
    Progressive disclosure tier for context-window budget management.

    Lower numbers represent more heavily summarized payloads; higher numbers
    represent greater fidelity up to FULL which delivers the original data.
    """

    MINIMAL = 1
    """Only key-value summary lines — typically ≤25% of original token count."""

    CONDENSED = 2
    """Structured summary retaining all top-level fields — ≤50% of original."""

    STANDARD = 3
    """Truncated full payload with deep-nested arrays capped at 10 items."""

    EXPANDED = 4
    """Near-full payload; only binary blobs and base64 strings are elided."""

    FULL = 5
    """Original payload delivered verbatim; no budget enforcement."""


class RollbackReason(str, Enum):
    """Reason codes recorded when the snapshot engine initiates a rollback."""

    INTENT_DEVIATION = "intent_deviation"
    """Agent state diverged beyond the configured deviation threshold."""

    LOOP_DETECTED = "loop_detected"
    """The execution path revisited an identical state within N steps."""

    TIMEOUT = "timeout"
    """Agent session exceeded its wall-clock execution budget."""

    MANUAL = "manual"
    """Rollback was explicitly requested via the management API."""

    ERROR = "error"
    """An unrecoverable runtime error was caught inside the agent loop."""


class SessionStatus(str, Enum):
    """Lifecycle states of an agent session."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ROLLED_BACK = "rolled_back"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================ #
#  Tool-call / JSON-RPC models                                                  #
# ============================================================================ #


class ToolCallInput(BaseModel):
    """
    Represents a single tool-call invocation arriving at the State Router.

    Conforms to the MCP (Model Context Protocol) tool-call JSON-RPC schema:

    .. code-block:: json

        {
          "call_id":   "01HWXY…",
          "tool_name": "search_web",
          "agent_id":  "agent-42",
          "session_id":"sess-99",
          "arguments": {"query": "Python asyncio"},
          "metadata":  {}
        }
    """

    model_config = ConfigDict(frozen=True)

    call_id: str = Field(
        default_factory=_new_ulid,
        description="Unique identifier for this specific tool invocation.",
    )
    tool_name: str = Field(
        description="Registered name of the tool being called.",
        min_length=1,
        max_length=128,
    )
    agent_id: str = Field(
        description="Identifier of the agent issuing this tool call.",
        min_length=1,
        max_length=128,
    )
    session_id: str = Field(
        description="Identifier of the agent session this call belongs to.",
        min_length=1,
        max_length=128,
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value arguments forwarded to the tool implementation.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Caller-supplied metadata forwarded without modification.",
    )
    requested_disclosure_level: DisclosureLevel = Field(
        default=DisclosureLevel.STANDARD,
        description=(
            "Requested progressive disclosure tier for the tool output.  The "
            "router may downgrade this if the resulting payload exceeds the "
            "configured token budget."
        ),
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the tool call was received by the router.",
    )


class ToolCallOutput(BaseModel):
    """
    Raw, unfiltered output returned by a tool implementation before the
    State Router applies progressive disclosure.
    """

    model_config = ConfigDict(frozen=True)

    call_id: str = Field(description="Must match the originating ``ToolCallInput.call_id``.")
    tool_name: str = Field(description="Name of the tool that produced this output.")
    agent_id: str = Field(description="Agent that issued the originating call.")
    session_id: str = Field(description="Session to which this output belongs.")
    result: Any = Field(description="Raw tool output — any JSON-serialisable value.")
    error: str | None = Field(
        default=None,
        description=(
            "Non-null when the tool returned an error.  The ``result`` field "
            "will be ``null`` in this case."
        ),
    )
    raw_token_estimate: int = Field(
        default=0,
        ge=0,
        description="Router-computed token estimate of the raw ``result`` payload.",
    )
    execution_duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock execution time of the tool in milliseconds.",
    )
    requested_disclosure_level: DisclosureLevel = Field(
        default=DisclosureLevel.STANDARD,
        description=(
            "Requested progressive disclosure tier for this output.  The router "
            "may downgrade to a lower tier if the payload exceeds the token budget."
        ),
    )
    timestamp: datetime = Field(default_factory=_utcnow)


class FilteredToolCallOutput(BaseModel):
    """
    Progressive-disclosure-filtered output forwarded to the LLM context.

    This is the object that the State Router returns to the calling agent
    after applying budget enforcement and disclosure-level compression.
    """

    model_config = ConfigDict(frozen=True)

    call_id: str
    tool_name: str
    agent_id: str
    session_id: str
    disclosure_level_applied: DisclosureLevel = Field(
        description="The disclosure tier actually applied (may differ from requested).",
    )
    filtered_result: Any = Field(
        description="Compressed/filtered tool output safe to inject into the LLM context.",
    )
    raw_token_estimate: int = Field(
        description="Token count of the original unfiltered result.",
    )
    filtered_token_estimate: int = Field(
        description="Token count of the filtered result after disclosure compression.",
    )
    compression_ratio: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "``filtered_tokens / raw_tokens``.  Values below 1.0 indicate "
            "active compression was applied."
        ),
    )
    budget_remaining: int = Field(
        description="Estimated tokens remaining in this session's context budget.",
    )
    error: str | None = Field(default=None)
    timestamp: datetime = Field(default_factory=_utcnow)


# ============================================================================ #
#  Agent Session models                                                         #
# ============================================================================ #


class IntentDeclaration(BaseModel):
    """
    An agent's declared intent, submitted at session initialisation.

    The snapshot engine uses this declaration as the baseline against which
    all subsequent state hashes are compared to compute deviation scores.
    """

    model_config = ConfigDict(frozen=True)

    primary_goal: str = Field(
        description="A concise, human-readable statement of the agent's primary objective.",
        min_length=10,
        max_length=2048,
    )
    allowed_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Exhaustive list of tool names the agent is permitted to call.  "
            "Calls to tools outside this list will be blocked by the router."
        ),
    )
    max_steps: int = Field(
        default=100,
        ge=1,
        description="Maximum number of tool calls permitted before the session is terminated.",
    )
    context_budget_override: int | None = Field(
        default=None,
        ge=256,
        description=(
            "Optional per-session token budget override.  When set, this value "
            "supersedes the global ``ASL_CONTEXT_TOKEN_BUDGET`` setting."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary caller metadata stored alongside the intent declaration.",
    )

    @field_validator("primary_goal")
    @classmethod
    def _strip_goal(cls, value: str) -> str:
        return value.strip()


class AgentSession(BaseModel):
    """
    Mutable runtime record of an active agent session.

    One ``AgentSession`` is created per ``/sessions`` POST request and is
    held in memory by the router and persisted as a snapshot by the snapshot
    engine.
    """

    model_config = ConfigDict(frozen=False)

    session_id: str = Field(default_factory=_new_ulid)
    agent_id: str
    intent: IntentDeclaration
    status: SessionStatus = Field(default=SessionStatus.INITIALIZING)
    step_count: int = Field(default=0, ge=0)
    tokens_consumed: int = Field(default=0, ge=0)
    intent_deviation_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Running deviation score; triggers rollback when ≥ threshold.",
    )
    active_tool_calls: list[str] = Field(
        default_factory=list,
        description="call_ids of in-flight tool calls for this session.",
    )
    tool_call_history: list[str] = Field(
        default_factory=list,
        description="Ordered list of completed call_ids for loop detection.",
    )
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = Field(default=None)

    def touch(self) -> None:
        """Update ``updated_at`` to the current UTC time."""
        self.updated_at = _utcnow()


# ============================================================================ #
#  Snapshot / rollback models                                                   #
# ============================================================================ #


class SnapshotRecord(BaseModel):
    """
    Immutable point-in-time snapshot of an ``AgentSession``.

    Snapshots are the atomic unit of the rollback system.  The snapshot
    engine stores them in SQLite; each record is identified by a monotonically
    increasing ``generation`` counter within its session.
    """

    model_config = ConfigDict(frozen=True)

    snapshot_id: str = Field(default_factory=_new_ulid)
    session_id: str
    agent_id: str
    generation: int = Field(
        ge=0,
        description="Monotonically increasing snapshot generation number within a session.",
    )
    session_state_json: str = Field(
        description="JSON-serialised ``AgentSession`` at the time of the snapshot.",
    )
    state_hash: str = Field(
        description="xxHash-64 digest of ``session_state_json`` for fast equality checks.",
    )
    intent_deviation_score: float = Field(ge=0.0, le=1.0)
    is_compressed: bool = Field(
        default=False,
        description="True when ``session_state_json`` is zlib-compressed.",
    )
    created_at: datetime = Field(default_factory=_utcnow)


class RollbackRequest(BaseModel):
    """Request payload to roll back a session to a previous snapshot."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    target_generation: int = Field(
        ge=0,
        description=(
            "Snapshot generation to restore.  Use ``0`` to roll back to the "
            "very first snapshot (session initialisation state)."
        ),
    )
    reason: RollbackReason
    initiator: Literal["system", "operator", "agent"] = Field(
        default="system",
        description="Entity that triggered this rollback.",
    )
    notes: str | None = Field(
        default=None,
        description="Optional human-readable notes for the audit trail.",
    )


class RollbackRecord(BaseModel):
    """Audit record created for every rollback operation."""

    model_config = ConfigDict(frozen=True)

    rollback_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    from_generation: int
    to_generation: int
    reason: RollbackReason
    initiator: str
    deviation_score_at_rollback: float
    notes: str | None = None
    executed_at: datetime = Field(default_factory=_utcnow)


# ============================================================================ #
#  API response envelopes                                                       #
# ============================================================================ #


class HealthResponse(BaseModel):
    """Response body for the ``GET /health`` endpoint."""

    model_config = ConfigDict(frozen=True)

    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    uptime_seconds: float
    active_sessions: int
    snapshots_persisted: int
    timestamp: datetime = Field(default_factory=_utcnow)


class SessionCreateResponse(BaseModel):
    """Response body for ``POST /sessions``."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    agent_id: str
    status: SessionStatus
    created_at: datetime
    message: str = "Session initialised successfully."


class ContextMetricsRecord(BaseModel):
    """
    Per-call context optimisation metrics emitted after each filtered
    tool-call output is computed.  Used by the monitoring dashboard.
    """

    model_config = ConfigDict(frozen=True)

    call_id: str
    session_id: str
    agent_id: str
    tool_name: str
    raw_tokens: int
    filtered_tokens: int
    compression_ratio: float
    disclosure_level: DisclosureLevel
    budget_before: int
    budget_after: int
    timestamp: datetime = Field(default_factory=_utcnow)
