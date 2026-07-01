"""
tests/test_snapshot_engine.py
================================
Integration tests for the async Snapshot Engine.

These tests exercise the full checkpoint → load → rollback cycle against a
real (temp) SQLite database.  They do not mock the store layer — the goal is
to validate end-to-end correctness of the persistence pipeline.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio

from agent_state_ledger.models import (
    AgentSession,
    IntentDeclaration,
    RollbackReason,
    RollbackRequest,
    SessionStatus,
)
from agent_state_ledger.snapshot.engine import SnapshotEngine
from agent_state_ledger.snapshot.store import SnapshotStore


# ============================================================================ #
#  Helper factories                                                             #
# ============================================================================ #

def _make_session(
    agent_id: str = "test-agent",
    max_steps: int = 50,
    deviation: float = 0.0,
) -> AgentSession:
    intent = IntentDeclaration(
        primary_goal="Test session for snapshot engine integration tests.",
        allowed_tools=["echo"],
        max_steps=max_steps,
    )
    session = AgentSession(
        agent_id=agent_id,
        intent=intent,
        status=SessionStatus.ACTIVE,
        intent_deviation_score=deviation,
    )
    return session


# ============================================================================ #
#  SnapshotStore tests                                                          #
# ============================================================================ #

class TestSnapshotStore:
    """Unit tests for the SnapshotStore persistence layer."""

    async def test_open_and_close(self, snapshot_store: SnapshotStore) -> None:
        """Store should open and close without errors."""
        # open() was called in the fixture — just check it's usable
        count = await snapshot_store.count_all_snapshots()
        assert count == 0

    async def test_save_and_load_snapshot(self, snapshot_store: SnapshotStore) -> None:
        """Save a session snapshot and load it back; it should be identical."""
        session = _make_session()
        record = await snapshot_store.save_snapshot_from_session(session, generation=0)

        assert record.generation == 0
        assert record.session_id == session.session_id
        assert record.state_hash  # non-empty

        restored = await snapshot_store.load_snapshot(session.session_id, generation=0)
        assert restored.session_id == session.session_id
        assert restored.agent_id == session.agent_id
        assert restored.intent.primary_goal == session.intent.primary_goal

    async def test_latest_generation_tracks_increments(
        self, snapshot_store: SnapshotStore
    ) -> None:
        """Latest generation should reflect the highest saved generation."""
        session = _make_session()
        await snapshot_store.save_snapshot_from_session(session, generation=0)
        await snapshot_store.save_snapshot_from_session(session, generation=1)
        await snapshot_store.save_snapshot_from_session(session, generation=2)

        latest = await snapshot_store.latest_generation(session.session_id)
        assert latest == 2

    async def test_load_nonexistent_snapshot_raises(
        self, snapshot_store: SnapshotStore
    ) -> None:
        """Loading a non-existent generation should raise KeyError."""
        session = _make_session()
        with pytest.raises(KeyError, match="No snapshot found"):
            await snapshot_store.load_snapshot(session.session_id, generation=99)

    async def test_delete_old_snapshots_keeps_last_n(
        self, snapshot_store: SnapshotStore
    ) -> None:
        """Pruning should retain only the N most recent generations."""
        session = _make_session()
        for gen in range(10):
            await snapshot_store.save_snapshot_from_session(session, generation=gen)

        deleted = await snapshot_store.delete_old_snapshots(session.session_id, keep_last_n=5)
        assert deleted == 5

        generations = await snapshot_store.list_generations(session.session_id)
        assert len(generations) == 5
        assert generations == list(range(5, 10))

    async def test_count_all_snapshots(self, snapshot_store: SnapshotStore) -> None:
        """Count should reflect total rows across all sessions."""
        s1 = _make_session(agent_id="a1")
        s2 = _make_session(agent_id="a2")
        await snapshot_store.save_snapshot_from_session(s1, generation=0)
        await snapshot_store.save_snapshot_from_session(s2, generation=0)
        await snapshot_store.save_snapshot_from_session(s2, generation=1)

        count = await snapshot_store.count_all_snapshots()
        assert count == 3

    async def test_compression_round_trip(self, tmp_path: Path) -> None:
        """Compressed snapshots should be loadable and equal to the original."""
        store = SnapshotStore(db_path=tmp_path / "compressed.db", compression_enabled=True)
        await store.open()
        session = _make_session()
        await store.save_snapshot_from_session(session, generation=0)
        restored = await store.load_snapshot(session.session_id, generation=0)
        assert restored.session_id == session.session_id
        await store.close()

    async def test_no_compression_round_trip(self, tmp_path: Path) -> None:
        """Uncompressed snapshots should also be loadable correctly."""
        store = SnapshotStore(db_path=tmp_path / "raw.db", compression_enabled=False)
        await store.open()
        session = _make_session()
        await store.save_snapshot_from_session(session, generation=0)
        restored = await store.load_snapshot(session.session_id, generation=0)
        assert restored.session_id == session.session_id
        await store.close()


# ============================================================================ #
#  SnapshotEngine tests                                                         #
# ============================================================================ #

class TestSnapshotEngine:
    """Integration tests for the SnapshotEngine orchestration layer."""

    async def test_checkpoint_creates_snapshot(
        self,
        snapshot_engine: SnapshotEngine,
        session_store,
    ) -> None:
        """A checkpoint call should persist a snapshot and increment the generation."""
        session = _make_session()
        await session_store.create_session(session)

        gen = await snapshot_engine.checkpoint(session)
        assert gen == 0

        total = await snapshot_engine.count_snapshots()
        assert total >= 1

    async def test_multiple_checkpoints_increment_generation(
        self,
        snapshot_engine: SnapshotEngine,
        session_store,
    ) -> None:
        """Each checkpoint call should assign the next sequential generation."""
        session = _make_session()
        await session_store.create_session(session)

        gen0 = await snapshot_engine.checkpoint(session)
        gen1 = await snapshot_engine.checkpoint(session)
        gen2 = await snapshot_engine.checkpoint(session)

        assert gen0 == 0
        assert gen1 == 1
        assert gen2 == 2

    async def test_maybe_checkpoint_defers_within_interval(
        self,
        snapshot_engine: SnapshotEngine,
        session_store,
    ) -> None:
        """Calling maybe_checkpoint twice in rapid succession should defer the second."""
        session = _make_session()
        await session_store.create_session(session)

        # First call — should checkpoint
        wrote = await snapshot_engine.maybe_checkpoint(session)
        assert wrote is True

        # Immediate second call — interval not elapsed yet (0.1s in test config)
        wrote2 = await snapshot_engine.maybe_checkpoint(session)
        assert wrote2 is False

    async def test_maybe_checkpoint_writes_after_interval(
        self,
        snapshot_engine: SnapshotEngine,
        session_store,
    ) -> None:
        """After the interval elapses, maybe_checkpoint should write."""
        session = _make_session()
        await session_store.create_session(session)

        await snapshot_engine.maybe_checkpoint(session)
        # Wait for the interval to elapse (0.1s in test settings)
        await asyncio.sleep(0.15)
        wrote = await snapshot_engine.maybe_checkpoint(session)
        assert wrote is True

    async def test_rollback_restores_session(
        self,
        snapshot_engine: SnapshotEngine,
        session_store,
    ) -> None:
        """Rolling back should restore session state to the target generation."""
        session = _make_session()
        await session_store.create_session(session)

        # Gen 0: step_count = 0
        await snapshot_engine.checkpoint(session)

        # Advance session
        session.step_count = 10
        session.tokens_consumed = 500
        await session_store.update_session(session)

        # Gen 1: step_count = 10
        await snapshot_engine.checkpoint(session)

        # Roll back to gen 0
        request = RollbackRequest(
            session_id=session.session_id,
            target_generation=0,
            reason=RollbackReason.MANUAL,
            initiator="operator",
        )
        record = await snapshot_engine.rollback(request)

        # Verify the record
        assert record.from_generation == 1
        assert record.to_generation == 0
        assert record.reason == RollbackReason.MANUAL

        # Verify the in-memory session was restored
        restored = await session_store.get_session(session.session_id)
        assert restored.step_count == 0
        assert restored.tokens_consumed == 0

    async def test_rollback_audit_record_persisted(
        self,
        snapshot_engine: SnapshotEngine,
        session_store,
    ) -> None:
        """A rollback audit record should be written to the store."""
        session = _make_session()
        await session_store.create_session(session)
        await snapshot_engine.checkpoint(session)

        request = RollbackRequest(
            session_id=session.session_id,
            target_generation=0,
            reason=RollbackReason.INTENT_DEVIATION,
            initiator="system",
            notes="Test rollback for audit.",
        )
        await snapshot_engine.rollback(request)

        records = await snapshot_engine._store.list_rollback_records(session.session_id)
        assert len(records) == 1
        assert records[0].reason == RollbackReason.INTENT_DEVIATION
        assert records[0].notes == "Test rollback for audit."

    async def test_loop_detection_flags_repeated_hash(
        self,
        snapshot_engine: SnapshotEngine,
        session_store,
    ) -> None:
        """The hash ring should detect when the same state hash reappears."""
        session = _make_session()
        await session_store.create_session(session)

        # Checkpoint the same state twice
        await snapshot_engine.checkpoint(session)

        # The hash ring should now contain one entry
        ring = snapshot_engine._session_hash_ring[session.session_id]
        assert len(ring) == 1

        # Checkpoint again with the same unchanged session state
        await snapshot_engine.checkpoint(session)

        ring = snapshot_engine._session_hash_ring[session.session_id]
        # Same hash should appear twice → loop detected
        hash_val = ring[0]
        is_loop = snapshot_engine._is_loop_detected(session.session_id, hash_val)
        assert is_loop is True
