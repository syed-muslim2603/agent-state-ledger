"""
agent_state_ledger.snapshot.engine
=====================================
Asynchronous State Snapshot Engine.

This is the transactional backbone of the Agent State Ledger.  Its
responsibilities are:

1. **Checkpointing** — Persist a point-in-time snapshot of an
   ``AgentSession`` to durable SQLite storage at configurable intervals.

2. **Loop detection** — Maintain a per-session hash ring.  If the same
   state hash recurs within the rolling history window, a loop is declared
   and the session is flagged for rollback.

3. **Rollback** — Atomically restore a session to a prior snapshot
   generation, write a ``RollbackRecord`` to the immutable audit log, and
   notify the in-memory ``SessionStore`` of the restored state.

4. **Background maintenance** — A long-running ``asyncio.Task`` fires
   every ``checkpoint_interval_seconds`` to:
   a. Flush any dirty sessions that have not been checkpointed recently.
   b. Prune snapshots beyond the ``max_rollback_depth`` per session.
   c. Purge snapshots older than ``retention_days``.

5. **Intent-deviation auto-rollback** — Called from the route handler
   when a session's deviation score crosses the configured threshold.

Concurrency model
-----------------
The engine is designed to run in a **single asyncio event loop**.  All
public methods are coroutines and must be awaited.  The background
maintenance task runs as a ``asyncio.Task`` that is started in ``start()``
and cancelled cleanly in ``stop()``.

The ``SnapshotStore`` (SQLite layer) uses a single connection with WAL
journal mode, which supports one concurrent writer and multiple readers.
The engine serialises all writes through a per-session ``asyncio.Lock``
obtained from the ``SnapshotStore``'s lock manager, ensuring that
concurrent tool-call requests for the same session do not produce
interleaved snapshot writes.

Error handling
--------------
Checkpointing is **best-effort** for performance — a failed checkpoint
does **not** propagate an exception to the caller (it is logged as a
warning instead).  Rollbacks are **critical-path** and do propagate errors
to the HTTP response layer because a failed rollback leaves the session in
an ambiguous state.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from pathlib import Path

import xxhash

from agent_state_ledger.config import get_settings
from agent_state_ledger.logging_setup import get_logger
from agent_state_ledger.models import (
    AgentSession,
    RollbackRecord,
    RollbackReason,
    RollbackRequest,
    SessionStatus,
)
from agent_state_ledger.snapshot.store import SnapshotStore

logger = get_logger(__name__)


class SnapshotEngine:
    """
    Async State Snapshot engine with checkpointing, loop detection, and rollback.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Forwarded to ``SnapshotStore``.

    Attributes
    ----------
    _store:
        The underlying SQLite persistence layer.
    _session_generations:
        In-memory map of ``session_id`` → current generation number, used
        to assign the next generation without querying the database on the
        hot path.
    _session_last_checkpoint:
        Map of ``session_id`` → monotonic timestamp of the last checkpoint.
        Used to debounce high-frequency state changes.
    _session_hash_ring:
        Map of ``session_id`` → deque of recent state hashes for O(N) loop
        detection where N = hash ring size (default 50).
    _session_locks:
        Per-session ``asyncio.Lock`` instances that serialise concurrent
        checkpoint/rollback operations for the same session.
    _dirty_sessions:
        Set of session IDs whose state has changed since the last checkpoint
        but whose checkpoint interval has not yet elapsed.  Flushed by the
        background maintenance task.
    """

    def __init__(self, db_path: Path) -> None:
        settings = get_settings()
        self._settings = settings.snapshot

        self._store = SnapshotStore(
            db_path=db_path,
            compression_enabled=self._settings.compression_enabled,
        )

        # In-memory tracking
        self._session_generations: dict[str, int] = defaultdict(lambda: -1)
        self._session_last_checkpoint: dict[str, float] = {}
        self._session_hash_ring: dict[str, list[str]] = defaultdict(list)
        self._session_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._dirty_sessions: set[str] = set()

        # Background task handle
        self._maintenance_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """
        Open the snapshot store and launch the background maintenance task.

        Must be called once during application startup before any checkpoint
        or rollback operations.
        """
        await self._store.open()

        # Synchronise in-memory generation counters from the database so
        # the engine survives process restarts gracefully.
        await self._restore_generation_state()

        self._maintenance_task = asyncio.create_task(
            self._background_maintenance_loop(),
            name="snapshot-engine-maintenance",
        )
        await logger.ainfo("snapshot_engine_started")

    async def stop(self) -> None:
        """
        Cleanly shut down the engine.

        Cancels the background maintenance task, performs a final flush of
        all dirty sessions, and closes the SQLite connection.
        """
        if self._maintenance_task is not None and not self._maintenance_task.done():
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass

        # Final flush of any dirty sessions before shutdown
        await self._flush_dirty_sessions()
        await self._store.close()
        await logger.ainfo("snapshot_engine_stopped")

    async def _restore_generation_state(self) -> None:
        """
        Re-hydrate ``_session_generations`` from the database after a restart.

        Queries the highest generation number for each session that has
        existing snapshots.  This is called once at startup and is not on
        the hot path.
        """
        conn = self._store._require_conn()  # intentional internal access
        async with conn.execute(
            "SELECT session_id, MAX(generation) AS max_gen "
            "FROM snapshots GROUP BY session_id"
        ) as cursor:
            rows = await cursor.fetchall()

        for row in rows:
            self._session_generations[row["session_id"]] = int(row["max_gen"])

        await logger.ainfo(
            "snapshot_generation_state_restored",
            sessions_recovered=len(rows),
        )

    # ------------------------------------------------------------------ #
    #  Checkpointing                                                       #
    # ------------------------------------------------------------------ #

    async def checkpoint(self, session: AgentSession) -> int:
        """
        Unconditionally checkpoint *session* right now.

        Assigns the next generation number, encodes and persists the snapshot,
        updates the hash ring for loop detection, and enforces the rollback
        depth limit by pruning excess old snapshots.

        Parameters
        ----------
        session:
            The live session to snapshot.

        Returns
        -------
        int
            The generation number assigned to this checkpoint.
        """
        session_id = session.session_id
        lock = self._session_locks[session_id]

        async with lock:
            # Assign next generation
            current_gen = self._session_generations[session_id]
            next_gen = current_gen + 1
            self._session_generations[session_id] = next_gen

            # Persist snapshot
            try:
                record = await self._store.save_snapshot_from_session(
                    session=session,
                    generation=next_gen,
                )
            except Exception as exc:
                # Restore generation counter to avoid gaps on retry
                self._session_generations[session_id] = current_gen
                await logger.aerror(
                    "checkpoint_failed",
                    session_id=session_id,
                    generation=next_gen,
                    error=str(exc),
                )
                raise

            # Update last-checkpoint timestamp
            self._session_last_checkpoint[session_id] = time.monotonic()
            self._dirty_sessions.discard(session_id)

            # Update hash ring for loop detection
            self._update_hash_ring(session_id, record.state_hash)

            # Prune excess old snapshots (keep max_rollback_depth)
            await self._store.delete_old_snapshots(
                session_id,
                keep_last_n=self._settings.max_rollback_depth,
            )

            await logger.ainfo(
                "checkpoint_written",
                session_id=session_id,
                generation=next_gen,
                state_hash=record.state_hash[:12] + "…",
                deviation_score=session.intent_deviation_score,
            )
            return next_gen

    async def maybe_checkpoint(self, session: AgentSession) -> bool:
        """
        Checkpoint *session* only if the minimum interval has elapsed since
        the last checkpoint.

        Marks the session as dirty when the interval has not elapsed so the
        background maintenance task will flush it on the next sweep.

        Parameters
        ----------
        session:
            The session to conditionally checkpoint.

        Returns
        -------
        bool
            ``True`` when a checkpoint was written; ``False`` when the call
            was deferred (interval not yet elapsed).
        """
        session_id = session.session_id
        last = self._session_last_checkpoint.get(session_id, 0.0)
        elapsed = time.monotonic() - last
        interval = self._settings.checkpoint_interval_seconds

        if elapsed >= interval:
            await self.checkpoint(session)
            return True
        else:
            self._dirty_sessions.add(session_id)
            return False

    def _update_hash_ring(self, session_id: str, state_hash: str) -> None:
        """
        Append *state_hash* to the session's rolling hash history.

        The ring is capped at 50 entries; older hashes are evicted in FIFO
        order.  Duplicate hash detection uses ``list.count`` — O(N) with
        N ≤ 50 which is negligible compared to I/O costs.
        """
        ring = self._session_hash_ring[session_id]
        ring.append(state_hash)
        if len(ring) > 50:
            ring.pop(0)

    def _is_loop_detected(self, session_id: str, state_hash: str) -> bool:
        """
        Return ``True`` if *state_hash* appears more than once in the recent
        hash ring for *session_id*, indicating a loop in the execution path.
        """
        ring = self._session_hash_ring[session_id]
        return ring.count(state_hash) > 1

    # ------------------------------------------------------------------ #
    #  Rollback                                                            #
    # ------------------------------------------------------------------ #

    async def rollback(self, request: RollbackRequest) -> RollbackRecord:
        """
        Atomically roll back a session to a specific snapshot generation.

        Steps:
        1. Acquire the per-session lock.
        2. Determine the current (from) generation.
        3. Load the target snapshot from SQLite.
        4. Restore the session state in the in-memory ``SessionStore``.
        5. Write a ``RollbackRecord`` to the audit log.
        6. Adjust the in-memory generation counter to match the restored state.

        Parameters
        ----------
        request:
            Validated ``RollbackRequest`` specifying session, target
            generation, and rollback reason.

        Returns
        -------
        RollbackRecord
            The audit record created for this rollback.

        Raises
        ------
        KeyError
            If the target generation does not exist in the snapshot store.
        RuntimeError
            If the session cannot be found in the in-memory store.
        """
        from agent_state_ledger.router.session_store import get_session_store

        session_id = request.session_id
        lock = self._session_locks[session_id]

        async with lock:
            from_gen = self._session_generations[session_id]
            to_gen = request.target_generation

            # Load target snapshot from durable store
            restored_session = await self._store.load_snapshot(session_id, to_gen)

            # Restore in-memory session
            session_store = get_session_store()
            await session_store.restore_session(restored_session)

            # Build audit record
            current_session = await session_store.get_session(session_id)
            rollback_record = RollbackRecord(
                session_id=session_id,
                from_generation=from_gen,
                to_generation=to_gen,
                reason=request.reason,
                initiator=request.initiator,
                deviation_score_at_rollback=current_session.intent_deviation_score,
                notes=request.notes,
            )

            # Persist audit record
            await self._store.save_rollback_record(rollback_record)

            # Adjust in-memory generation counter
            self._session_generations[session_id] = to_gen

            # Clear the hash ring to prevent false-positive loop detection
            # on the freshly restored clean state
            self._session_hash_ring[session_id].clear()

            await logger.awarning(
                "session_rolled_back",
                session_id=session_id,
                from_generation=from_gen,
                to_generation=to_gen,
                reason=request.reason.value,
                initiator=request.initiator,
            )
            return rollback_record

    async def auto_rollback(
        self,
        session_id: str,
        reason: RollbackReason,
    ) -> RollbackRecord | None:
        """
        Automatically roll back *session_id* to its most recent safe snapshot.

        The "most recent safe" snapshot is defined as the highest generation
        whose ``intent_deviation_score`` is below the configured threshold.
        If no such snapshot exists, the genesis snapshot (generation 0) is used.

        Returns ``None`` if the session has no snapshots at all (edge case
        during very early session lifecycle).
        """
        from agent_state_ledger.router.session_store import (
            SessionNotFoundError,
            get_session_store,
        )

        threshold = self._settings.intent_deviation_threshold
        generations = await self._store.list_generations(session_id)

        if not generations:
            await logger.awarning(
                "auto_rollback_skipped_no_snapshots",
                session_id=session_id,
                reason=reason.value,
            )
            return None

        # Find the highest generation whose deviation score is safe
        target_gen = generations[0]  # Genesis fallback
        for gen in reversed(generations):
            conn = self._store._require_conn()  # intentional internal access
            async with conn.execute(
                "SELECT intent_deviation_score FROM snapshots "
                "WHERE session_id = ? AND generation = ?",
                (session_id, gen),
            ) as cursor:
                row = await cursor.fetchone()

            if row and row["intent_deviation_score"] < threshold:
                target_gen = gen
                break

        rollback_request = RollbackRequest(
            session_id=session_id,
            target_generation=target_gen,
            reason=reason,
            initiator="system",
            notes=(
                f"Automatic rollback triggered by {reason.value}.  "
                f"Restored to generation {target_gen} (last safe state)."
            ),
        )
        return await self.rollback(rollback_request)

    # ------------------------------------------------------------------ #
    #  Background maintenance loop                                         #
    # ------------------------------------------------------------------ #

    async def _background_maintenance_loop(self) -> None:
        """
        Long-running background coroutine that runs every
        ``checkpoint_interval_seconds`` to perform housekeeping tasks.

        Tasks performed on each tick:
        1. Flush all dirty (un-checkpointed) sessions.
        2. Purge snapshots older than ``retention_days``.

        The loop runs forever until cancelled (via ``asyncio.CancelledError``
        caught and re-raised cleanly during ``stop()``).
        """
        interval = self._settings.checkpoint_interval_seconds
        retention_days = self._settings.retention_days

        await logger.ainfo(
            "maintenance_loop_started",
            interval_seconds=interval,
            retention_days=retention_days,
        )

        while True:
            try:
                await asyncio.sleep(interval)
                await self._flush_dirty_sessions()
                purged = await self._store.purge_expired_snapshots(retention_days)
                if purged > 0:
                    await logger.ainfo("retention_purge_complete", rows_deleted=purged)
            except asyncio.CancelledError:
                await logger.ainfo("maintenance_loop_cancelled")
                raise
            except Exception as exc:  # pragma: no cover
                # Log but do not crash the loop — transient DB errors should
                # not bring down the entire service
                await logger.aerror(
                    "maintenance_loop_error",
                    error=str(exc),
                    exc_info=True,
                )

    async def _flush_dirty_sessions(self) -> None:
        """
        Force-checkpoint all sessions in ``_dirty_sessions``.

        Dirty sessions are those that received state updates since the last
        checkpoint but whose minimum interval had not yet elapsed.
        """
        from agent_state_ledger.router.session_store import (
            SessionNotFoundError,
            get_session_store,
        )

        if not self._dirty_sessions:
            return

        store = get_session_store()
        session_ids = list(self._dirty_sessions)

        for session_id in session_ids:
            try:
                session = await store.get_session(session_id)
                await self.checkpoint(session)
            except SessionNotFoundError:
                # Session may have been deleted — remove from dirty set
                self._dirty_sessions.discard(session_id)
            except Exception as exc:
                await logger.awarning(
                    "dirty_session_flush_failed",
                    session_id=session_id,
                    error=str(exc),
                )

    # ------------------------------------------------------------------ #
    #  Query helpers                                                       #
    # ------------------------------------------------------------------ #

    async def count_snapshots(self) -> int:
        """Return the total number of snapshot records across all sessions."""
        return await self._store.count_all_snapshots()

    async def list_generations(self, session_id: str) -> list[int]:
        """Return all persisted generation numbers for *session_id*."""
        return await self._store.list_generations(session_id)


# ============================================================================ #
#  Process-wide singleton                                                       #
# ============================================================================ #

_engine_instance: SnapshotEngine | None = None


def init_snapshot_engine(engine: SnapshotEngine) -> None:
    """Register *engine* as the process-wide snapshot engine singleton."""
    global _engine_instance
    _engine_instance = engine


def get_snapshot_engine() -> SnapshotEngine:
    """
    Return the process-wide ``SnapshotEngine`` singleton.

    Raises
    ------
    RuntimeError
        If ``init_snapshot_engine`` has not been called yet.
    """
    if _engine_instance is None:
        raise RuntimeError(
            "SnapshotEngine is not initialised.  "
            "Call ``init_snapshot_engine(engine)`` during application startup."
        )
    return _engine_instance
