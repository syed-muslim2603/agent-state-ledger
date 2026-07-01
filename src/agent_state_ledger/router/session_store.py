"""
agent_state_ledger.router.session_store
========================================
In-memory session registry used by the State Router.

This module provides a thread-safe, ``asyncio``-compatible store for
``AgentSession`` objects that are currently active in the router process.
It is intentionally kept in-memory (not persisted) because the snapshot
engine handles durable persistence; the router's in-memory store is a
fast, authoritative view of live sessions that supports sub-millisecond
lookups on the hot path.

Consistency model
-----------------
``SessionStore`` is the single writer for ``AgentSession.status``,
``AgentSession.step_count``, and ``AgentSession.tokens_consumed`` during
normal operation.  The snapshot engine reads sessions via ``get_session``
and may write back a rolled-back state via ``restore_session``.

All public methods are async even though the underlying data structure is
a plain Python dict, because:

1. The interface must be compatible with potential future backends
   (e.g. Redis cluster) that are genuinely async.
2. Callers are uniformly async (FastAPI path operations) and cannot
   block the event loop.

Concurrency protection
-----------------------
Each session has a dedicated ``asyncio.Lock`` keyed by ``session_id``.
Callers that need to perform a read-modify-write cycle (e.g. incrementing
``step_count`` and deducting from ``tokens_consumed``) must hold the lock
for the duration of their transaction:

.. code-block:: python

    async with session_store.session_lock(session_id):
        session = await session_store.get_session(session_id)
        session.step_count += 1
        await session_store.update_session(session)
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from agent_state_ledger.logging_setup import get_logger
from agent_state_ledger.models import AgentSession, SessionStatus

logger = get_logger(__name__)


class SessionNotFoundError(KeyError):
    """Raised when a requested session_id does not exist in the store."""


class SessionConflictError(ValueError):
    """Raised when attempting to create a session with a duplicate session_id."""


class SessionStore:
    """
    Thread-safe in-memory registry of active ``AgentSession`` objects.

    Lifecycle
    ---------
    1. ``create_session`` — called by ``POST /sessions``
    2. ``get_session``    — called on every tool-call to read current state
    3. ``update_session`` — called after every mutation
    4. ``delete_session`` — called on session termination / rollback restore

    Parameters
    ----------
    None.  Instances should be created once and held as a FastAPI dependency.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, AgentSession] = {}
        """Primary session registry keyed by session_id."""

        self._locks: dict[str, asyncio.Lock] = {}
        """Per-session async locks for read-modify-write transactions."""

        self._creation_order: list[str] = []
        """Insertion-ordered list of session_ids for iteration."""

    # ------------------------------------------------------------------ #
    #  Lock management                                                     #
    # ------------------------------------------------------------------ #

    def _get_or_create_lock(self, session_id: str) -> asyncio.Lock:
        """Return the existing lock for *session_id*, creating one if absent."""
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    @asynccontextmanager
    async def session_lock(self, session_id: str) -> AsyncGenerator[None, None]:
        """
        Async context manager that acquires the per-session lock.

        Usage::

            async with store.session_lock(session_id):
                session = await store.get_session(session_id)
                # ... mutate session ...
                await store.update_session(session)
        """
        lock = self._get_or_create_lock(session_id)
        async with lock:
            yield

    # ------------------------------------------------------------------ #
    #  CRUD operations                                                     #
    # ------------------------------------------------------------------ #

    async def create_session(self, session: AgentSession) -> AgentSession:
        """
        Register a new agent session.

        Parameters
        ----------
        session:
            A fully initialised ``AgentSession`` instance.  Its
            ``session_id`` must be globally unique within this store.

        Returns
        -------
        AgentSession
            The registered session (same object, mutated to ACTIVE status).

        Raises
        ------
        SessionConflictError
            If ``session.session_id`` already exists in the store.
        """
        if session.session_id in self._sessions:
            raise SessionConflictError(
                f"Session '{session.session_id}' already exists.  "
                "Duplicate session IDs are not permitted."
            )

        session.status = SessionStatus.ACTIVE
        session.touch()

        self._sessions[session.session_id] = session
        self._creation_order.append(session.session_id)
        self._get_or_create_lock(session.session_id)  # pre-create the lock

        await logger.ainfo(
            "session_created",
            session_id=session.session_id,
            agent_id=session.agent_id,
            max_steps=session.intent.max_steps,
        )
        return session

    async def get_session(self, session_id: str) -> AgentSession:
        """
        Retrieve a session by ID.

        Raises
        ------
        SessionNotFoundError
            If no session with *session_id* is registered.
        """
        try:
            return self._sessions[session_id]
        except KeyError:
            raise SessionNotFoundError(
                f"Session '{session_id}' not found.  "
                "It may have been deleted or never created."
            ) from None

    async def update_session(self, session: AgentSession) -> AgentSession:
        """
        Persist an updated session back to the store.

        The session must already exist (use ``create_session`` for new ones).

        Parameters
        ----------
        session:
            The mutated ``AgentSession``.  ``session.touch()`` is called
            automatically to update ``updated_at``.

        Returns
        -------
        AgentSession
            The updated session as stored.
        """
        if session.session_id not in self._sessions:
            raise SessionNotFoundError(
                f"Cannot update non-existent session '{session.session_id}'."
            )
        session.touch()
        self._sessions[session.session_id] = session
        return session

    async def restore_session(self, session: AgentSession) -> AgentSession:
        """
        Replace the current in-memory state with a rolled-back *session*.

        Called exclusively by the snapshot engine's rollback path.  This
        differs from ``update_session`` in that it does **not** call
        ``session.touch()`` — the ``updated_at`` timestamp preserved in the
        snapshot is intentionally retained to make the audit trail clear.
        """
        if session.session_id not in self._sessions:
            raise SessionNotFoundError(
                f"Cannot restore non-existent session '{session.session_id}'."
            )
        session.status = SessionStatus.ROLLED_BACK
        self._sessions[session.session_id] = session

        await logger.awarning(
            "session_restored_from_snapshot",
            session_id=session.session_id,
            agent_id=session.agent_id,
            step_count=session.step_count,
        )
        return session

    async def delete_session(self, session_id: str) -> None:
        """
        Remove a session from the store.

        Does not raise if the session is already absent (idempotent).
        """
        self._sessions.pop(session_id, None)
        self._locks.pop(session_id, None)
        try:
            self._creation_order.remove(session_id)
        except ValueError:
            pass
        await logger.ainfo("session_deleted", session_id=session_id)

    async def terminate_session(
        self,
        session_id: str,
        final_status: SessionStatus,
    ) -> AgentSession:
        """
        Mark a session with a terminal status without removing it from the
        store.  The session remains readable for audit queries.

        Parameters
        ----------
        session_id:
            Session to terminate.
        final_status:
            Must be one of ``COMPLETED``, ``FAILED``, or ``ROLLED_BACK``.

        Returns
        -------
        AgentSession
            The session with its updated status.
        """
        if final_status not in (
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.ROLLED_BACK,
        ):
            raise ValueError(
                f"Invalid terminal status '{final_status}'.  "
                "Must be COMPLETED, FAILED, or ROLLED_BACK."
            )
        async with self.session_lock(session_id):
            session = await self.get_session(session_id)
            session.status = final_status
            session.completed_at = datetime.now(tz=timezone.utc)
            session.touch()
            return await self.update_session(session)

    # ------------------------------------------------------------------ #
    #  Query helpers                                                       #
    # ------------------------------------------------------------------ #

    async def list_sessions(
        self,
        status_filter: SessionStatus | None = None,
    ) -> list[AgentSession]:
        """
        Return all sessions, optionally filtered by *status_filter*.

        Preserves insertion order.
        """
        sessions = [self._sessions[sid] for sid in self._creation_order
                    if sid in self._sessions]
        if status_filter is not None:
            sessions = [s for s in sessions if s.status == status_filter]
        return sessions

    async def active_session_count(self) -> int:
        """Return the number of sessions currently in ACTIVE status."""
        return sum(
            1 for s in self._sessions.values()
            if s.status == SessionStatus.ACTIVE
        )

    async def budget_for(self, session_id: str) -> int:
        """
        Compute the remaining token budget for *session_id*.

        Uses the per-session override if present, otherwise falls back to
        the global context budget setting.

        Returns
        -------
        int
            Non-negative remaining token count.
        """
        from agent_state_ledger.config import get_settings

        session = await self.get_session(session_id)
        global_budget = get_settings().context.token_budget
        max_budget = session.intent.context_budget_override or global_budget
        remaining = max(max_budget - session.tokens_consumed, 0)
        return remaining


# ============================================================================ #
#  Module-level singleton                                                       #
# ============================================================================ #

#: Process-wide session store.  The FastAPI app's lifespan mounts this into
#: ``app.state.session_store`` for dependency injection.
_global_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    """
    Return the process-wide ``SessionStore`` singleton.

    Raises
    ------
    RuntimeError
        If the store has not yet been initialised (i.e. ``init_session_store``
        has not been called).
    """
    global _global_store
    if _global_store is None:
        _global_store = SessionStore()
    return _global_store


def init_session_store() -> SessionStore:
    """
    Initialise and return the process-wide ``SessionStore``.

    Idempotent — returns the existing store if already initialised.
    """
    global _global_store
    if _global_store is None:
        _global_store = SessionStore()
    return _global_store
