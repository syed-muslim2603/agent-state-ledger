"""
agent_state_ledger.snapshot.store
===================================
Async SQLite persistence layer for the Snapshot Engine.

This module owns the database schema, all DDL migrations, and every SQL
query executed against the snapshot store.  It exposes a clean async API
so the engine layer never writes raw SQL.

Schema
------
Three tables are maintained:

``snapshots``
    One row per ``SnapshotRecord``.  Indexed on ``(session_id, generation)``
    for fast rollback lookups and on ``created_at`` for retention sweeps.

``rollback_audit``
    Append-only audit log of every rollback operation (``RollbackRecord``).
    Never modified after insertion; used for compliance reporting.

``session_metadata``
    Lightweight key-value sidecar for each session (total snapshots taken,
    last checkpoint wall-clock time) so the engine avoids full table scans
    for common queries.

Connection management
---------------------
A single ``aiosqlite`` connection is opened per ``SnapshotStore`` instance.
The engine creates one store instance at startup and holds it for the
process lifetime.  WAL journal mode is enabled to allow concurrent readers
without blocking the single writer, which is important because the health-
check endpoint reads snapshot counts while the background checkpoint task
is writing.

All write operations use explicit transactions (``BEGIN IMMEDIATE``) to
ensure durability and to prevent partial writes on process crash.
"""

from __future__ import annotations

import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiosqlite
import orjson

from agent_state_ledger.logging_setup import get_logger
from agent_state_ledger.models import (
    AgentSession,
    RollbackRecord,
    SnapshotRecord,
)

logger = get_logger(__name__)

# ============================================================================ #
#  DDL                                                                          #
# ============================================================================ #

_DDL_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id          TEXT    NOT NULL,
    session_id           TEXT    NOT NULL,
    agent_id             TEXT    NOT NULL,
    generation           INTEGER NOT NULL,
    session_state_json   BLOB    NOT NULL,
    state_hash           TEXT    NOT NULL,
    intent_deviation_score REAL  NOT NULL DEFAULT 0.0,
    is_compressed        INTEGER NOT NULL DEFAULT 0,
    created_at           TEXT    NOT NULL,
    PRIMARY KEY (session_id, generation)
);
"""

_DDL_SNAPSHOTS_IDX_HASH = """
CREATE INDEX IF NOT EXISTS idx_snapshots_state_hash
    ON snapshots (state_hash);
"""

_DDL_SNAPSHOTS_IDX_CREATED = """
CREATE INDEX IF NOT EXISTS idx_snapshots_created_at
    ON snapshots (created_at);
"""

_DDL_ROLLBACK_AUDIT = """
CREATE TABLE IF NOT EXISTS rollback_audit (
    rollback_id                TEXT NOT NULL PRIMARY KEY,
    session_id                 TEXT NOT NULL,
    from_generation            INTEGER NOT NULL,
    to_generation              INTEGER NOT NULL,
    reason                     TEXT NOT NULL,
    initiator                  TEXT NOT NULL,
    deviation_score_at_rollback REAL NOT NULL DEFAULT 0.0,
    notes                      TEXT,
    executed_at                TEXT NOT NULL
);
"""

_DDL_SESSION_META = """
CREATE TABLE IF NOT EXISTS session_metadata (
    session_id            TEXT    NOT NULL PRIMARY KEY,
    total_snapshots       INTEGER NOT NULL DEFAULT 0,
    last_checkpoint_at    TEXT,
    last_state_hash       TEXT
);
"""


# ============================================================================ #
#  Store implementation                                                         #
# ============================================================================ #


class SnapshotStore:
    """
    Async SQLite persistence layer for snapshots and rollback audit records.

    Parameters
    ----------
    db_path:
        Filesystem path to the SQLite database file.  The parent directory
        is created automatically if it does not exist.
    compression_enabled:
        When ``True`` (default), snapshot JSON blobs are compressed with
        zlib before storage and decompressed transparently on read.
    """

    def __init__(
        self,
        db_path: Path,
        compression_enabled: bool = True,
    ) -> None:
        self._db_path = db_path
        self._compression_enabled = compression_enabled
        self._conn: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    async def open(self) -> None:
        """
        Open the SQLite connection, apply PRAGMA settings, and run DDL
        migrations to ensure the schema is up-to-date.

        This method must be called before any other store method.
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = await aiosqlite.connect(
            database=str(self._db_path),
            timeout=30.0,
        )
        self._conn.row_factory = aiosqlite.Row

        await self._conn.execute("PRAGMA journal_mode = WAL;")
        await self._conn.execute("PRAGMA synchronous = NORMAL;")
        await self._conn.execute("PRAGMA foreign_keys = ON;")
        await self._conn.execute("PRAGMA cache_size = -8000;")  # 8 MiB page cache
        await self._conn.execute("PRAGMA temp_store = MEMORY;")

        await self._run_migrations()
        await logger.ainfo("snapshot_store_opened", db_path=str(self._db_path))

    async def close(self) -> None:
        """
        Flush pending WAL frames and close the SQLite connection.

        Idempotent — safe to call even if ``open`` was never called.
        """
        if self._conn is not None:
            await self._conn.execute("PRAGMA wal_checkpoint(FULL);")
            await self._conn.close()
            self._conn = None
            await logger.ainfo("snapshot_store_closed")

    async def _run_migrations(self) -> None:
        """Apply all DDL statements idempotently (CREATE IF NOT EXISTS)."""
        assert self._conn is not None, "Store not opened."
        async with self._conn.executescript(
            "\n".join([
                _DDL_SNAPSHOTS,
                _DDL_SNAPSHOTS_IDX_HASH,
                _DDL_SNAPSHOTS_IDX_CREATED,
                _DDL_ROLLBACK_AUDIT,
                _DDL_SESSION_META,
            ])
        ):
            pass
        await self._conn.commit()
        await logger.ainfo("snapshot_store_migrations_applied")

    def _require_conn(self) -> aiosqlite.Connection:
        """Return the open connection or raise ``RuntimeError``."""
        if self._conn is None:
            raise RuntimeError(
                "SnapshotStore is not open.  Call ``await store.open()`` first."
            )
        return self._conn

    # ------------------------------------------------------------------ #
    #  Encoding helpers                                                    #
    # ------------------------------------------------------------------ #

    def _encode(self, session: AgentSession) -> tuple[bytes, bool]:
        """
        Serialise *session* to bytes, optionally compressing with zlib.

        Returns
        -------
        tuple[bytes, bool]
            ``(blob, is_compressed)`` — the raw bytes to persist and a flag
            indicating whether the bytes are zlib-compressed.
        """
        raw_json: bytes = orjson.dumps(
            session.model_dump(mode="json"),
            option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_DATACLASS,
        )
        if self._compression_enabled:
            compressed = zlib.compress(raw_json, level=6)
            return compressed, True
        return raw_json, False

    def _decode(self, blob: bytes, is_compressed: bool) -> dict[str, Any]:
        """
        Deserialise a stored snapshot blob back to a dict.

        Parameters
        ----------
        blob:
            Raw bytes from the database ``session_state_json`` column.
        is_compressed:
            ``True`` when *blob* is zlib-compressed.

        Returns
        -------
        dict[str, Any]
            Parsed session state suitable for ``AgentSession(**data)``.
        """
        if is_compressed:
            blob = zlib.decompress(blob)
        return orjson.loads(blob)

    # ------------------------------------------------------------------ #
    #  Snapshot CRUD                                                       #
    # ------------------------------------------------------------------ #

    async def save_snapshot(self, record: SnapshotRecord) -> None:
        """
        Persist *record* to the ``snapshots`` table.

        Also upserts the ``session_metadata`` row to keep lightweight
        aggregate stats current without requiring expensive COUNT queries.

        Raises
        ------
        aiosqlite.IntegrityError
            On duplicate ``(session_id, generation)`` primary key violation
            (should never happen under normal operation since generations are
            monotonically assigned by the engine).
        """
        conn = self._require_conn()
        created_at_str = record.created_at.isoformat()

        async with conn.execute("BEGIN IMMEDIATE"):
            await conn.execute(
                """
                INSERT INTO snapshots
                    (snapshot_id, session_id, agent_id, generation,
                     session_state_json, state_hash, intent_deviation_score,
                     is_compressed, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.snapshot_id,
                    record.session_id,
                    record.agent_id,
                    record.generation,
                    record.session_state_json,
                    record.state_hash,
                    record.intent_deviation_score,
                    int(record.is_compressed),
                    created_at_str,
                ),
            )
            # Upsert session metadata
            await conn.execute(
                """
                INSERT INTO session_metadata (session_id, total_snapshots, last_checkpoint_at, last_state_hash)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    total_snapshots    = total_snapshots + 1,
                    last_checkpoint_at = excluded.last_checkpoint_at,
                    last_state_hash    = excluded.last_state_hash
                """,
                (record.session_id, created_at_str, record.state_hash),
            )
            await conn.commit()

    async def save_snapshot_from_session(
        self,
        session: AgentSession,
        generation: int,
    ) -> SnapshotRecord:
        """
        Encode *session* and persist it as snapshot generation *generation*.

        This is the primary write path called by the engine's checkpoint logic.

        Parameters
        ----------
        session:
            The live ``AgentSession`` to snapshot.
        generation:
            Monotonically increasing generation number for this session.

        Returns
        -------
        SnapshotRecord
            The constructed and persisted record.
        """
        import xxhash

        blob, is_compressed = self._encode(session)
        state_hash = xxhash.xxh64(blob).hexdigest()

        record = SnapshotRecord(
            session_id=session.session_id,
            agent_id=session.agent_id,
            generation=generation,
            session_state_json=blob.hex(),  # store as hex string for TEXT column compat
            state_hash=state_hash,
            intent_deviation_score=session.intent_deviation_score,
            is_compressed=is_compressed,
        )
        await self.save_snapshot(record)
        return record

    async def load_snapshot(self, session_id: str, generation: int) -> AgentSession:
        """
        Load and deserialise snapshot *generation* for *session_id*.

        Parameters
        ----------
        session_id:
            The session whose snapshot should be loaded.
        generation:
            Specific generation to restore.

        Returns
        -------
        AgentSession
            Reconstructed session in the state it had at *generation*.

        Raises
        ------
        KeyError
            If no snapshot exists for the given ``(session_id, generation)``.
        """
        conn = self._require_conn()
        async with conn.execute(
            "SELECT session_state_json, is_compressed FROM snapshots "
            "WHERE session_id = ? AND generation = ?",
            (session_id, generation),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            raise KeyError(
                f"No snapshot found for session '{session_id}' at generation {generation}."
            )

        blob = bytes.fromhex(row["session_state_json"])
        is_compressed = bool(row["is_compressed"])
        state_dict = self._decode(blob, is_compressed)
        return AgentSession(**state_dict)

    async def latest_generation(self, session_id: str) -> int:
        """
        Return the highest persisted generation number for *session_id*.

        Returns ``-1`` if no snapshots exist yet (new session before its
        first checkpoint).
        """
        conn = self._require_conn()
        async with conn.execute(
            "SELECT MAX(generation) AS max_gen FROM snapshots WHERE session_id = ?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None or row["max_gen"] is None:
            return -1
        return int(row["max_gen"])

    async def get_state_hash(self, session_id: str, generation: int) -> str | None:
        """Return the stored state hash for a specific snapshot, or ``None``."""
        conn = self._require_conn()
        async with conn.execute(
            "SELECT state_hash FROM snapshots WHERE session_id = ? AND generation = ?",
            (session_id, generation),
        ) as cursor:
            row = await cursor.fetchone()
        return row["state_hash"] if row else None

    async def list_generations(self, session_id: str) -> list[int]:
        """Return an ascending list of all persisted generation numbers for *session_id*."""
        conn = self._require_conn()
        async with conn.execute(
            "SELECT generation FROM snapshots WHERE session_id = ? ORDER BY generation ASC",
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [row["generation"] for row in rows]

    async def delete_old_snapshots(self, session_id: str, keep_last_n: int) -> int:
        """
        Prune old snapshots for *session_id*, keeping only the most recent
        *keep_last_n* generations.

        Returns the number of rows deleted.
        """
        conn = self._require_conn()
        generations = await self.list_generations(session_id)
        to_delete = generations[: max(0, len(generations) - keep_last_n)]
        if not to_delete:
            return 0

        placeholders = ",".join("?" * len(to_delete))
        async with conn.execute(
            f"DELETE FROM snapshots WHERE session_id = ? AND generation IN ({placeholders})",
            [session_id, *to_delete],
        ):
            pass
        await conn.commit()
        return len(to_delete)

    async def purge_expired_snapshots(self, retention_days: int) -> int:
        """
        Delete all snapshots older than *retention_days* days.

        Returns the number of rows purged.
        """
        conn = self._require_conn()
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=retention_days)).isoformat()
        async with conn.execute(
            "DELETE FROM snapshots WHERE created_at < ?",
            (cutoff,),
        ) as cursor:
            row_count = cursor.rowcount
        await conn.commit()
        await logger.ainfo(
            "expired_snapshots_purged",
            cutoff=cutoff,
            rows_deleted=row_count,
        )
        return row_count or 0

    async def count_all_snapshots(self) -> int:
        """Return the total number of snapshot rows across all sessions."""
        conn = self._require_conn()
        async with conn.execute("SELECT COUNT(*) AS cnt FROM snapshots") as cursor:
            row = await cursor.fetchone()
        return int(row["cnt"]) if row else 0

    # ------------------------------------------------------------------ #
    #  Rollback audit                                                      #
    # ------------------------------------------------------------------ #

    async def save_rollback_record(self, record: RollbackRecord) -> None:
        """Append a ``RollbackRecord`` to the immutable audit log."""
        conn = self._require_conn()
        await conn.execute(
            """
            INSERT INTO rollback_audit
                (rollback_id, session_id, from_generation, to_generation,
                 reason, initiator, deviation_score_at_rollback, notes, executed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.rollback_id,
                record.session_id,
                record.from_generation,
                record.to_generation,
                record.reason.value,
                record.initiator,
                record.deviation_score_at_rollback,
                record.notes,
                record.executed_at.isoformat(),
            ),
        )
        await conn.commit()

    async def list_rollback_records(self, session_id: str) -> list[RollbackRecord]:
        """Return all rollback audit records for *session_id*, oldest first."""
        conn = self._require_conn()
        async with conn.execute(
            "SELECT * FROM rollback_audit WHERE session_id = ? ORDER BY executed_at ASC",
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        from agent_state_ledger.models import RollbackReason

        records = []
        for row in rows:
            records.append(
                RollbackRecord(
                    rollback_id=row["rollback_id"],
                    session_id=row["session_id"],
                    from_generation=row["from_generation"],
                    to_generation=row["to_generation"],
                    reason=RollbackReason(row["reason"]),
                    initiator=row["initiator"],
                    deviation_score_at_rollback=row["deviation_score_at_rollback"],
                    notes=row["notes"],
                    executed_at=datetime.fromisoformat(row["executed_at"]),
                )
            )
        return records
