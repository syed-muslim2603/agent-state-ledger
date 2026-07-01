"""
tests/conftest.py
==================
Shared pytest fixtures for the Agent State Ledger test suite.

All fixtures that are used across multiple test modules are defined here.
Module-specific fixtures live in the corresponding test file.

Fixture design principles
--------------------------
1. ``session``-scoped fixtures that involve I/O (e.g. the snapshot DB) use
   ``tmp_path_factory`` so each test invocation gets a fresh directory.
2. ``function``-scoped fixtures are used by default to ensure test isolation.
3. All async fixtures use ``asyncio_mode = "auto"`` (configured in
   ``pyproject.toml``) so there is no need for ``@pytest.mark.asyncio``
   on individual tests.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from agent_state_ledger.config import get_settings
from agent_state_ledger.models import (
    AgentSession,
    IntentDeclaration,
    SessionStatus,
)
from agent_state_ledger.router.session_store import SessionStore
from agent_state_ledger.snapshot.engine import SnapshotEngine, init_snapshot_engine
from agent_state_ledger.snapshot.store import SnapshotStore


# ============================================================================ #
#  Environment override                                                         #
# ============================================================================ #

@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Override settings to keep tests hermetic:
    - Point the snapshot DB at a tmp directory.
    - Use char_approx token counting (no tiktoken dependency in tests).
    - Reduce checkpoint interval to 0.1s to speed up async tests.
    """
    monkeypatch.setenv("ASL_SNAPSHOT_DB_PATH", str(tmp_path / "test_snapshots.db"))
    monkeypatch.setenv("ASL_CONTEXT_TOKEN_COUNT_METHOD", "char_approx")
    monkeypatch.setenv("ASL_SNAPSHOT_CHECKPOINT_INTERVAL_SECONDS", "0.1")
    monkeypatch.setenv("ASL_METRICS_ENABLED", "false")
    monkeypatch.setenv("ASL_LOG_FORMAT", "console")
    monkeypatch.setenv("ASL_LOG_LEVEL", "WARNING")
    # Clear the cached settings so the patched env is picked up
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ============================================================================ #
#  Model factories                                                              #
# ============================================================================ #

@pytest.fixture
def sample_intent() -> IntentDeclaration:
    """Return a minimal valid ``IntentDeclaration``."""
    return IntentDeclaration(
        primary_goal="Summarise the top 5 search results for a given query.",
        allowed_tools=["search_web", "echo", "compute_hash"],
        max_steps=50,
        context_budget_override=4096,
    )


@pytest.fixture
def sample_session(sample_intent: IntentDeclaration) -> AgentSession:
    """Return a minimal active ``AgentSession`` with a known intent."""
    session = AgentSession(
        agent_id="test-agent-001",
        intent=sample_intent,
        status=SessionStatus.ACTIVE,
    )
    return session


# ============================================================================ #
#  SessionStore fixture                                                         #
# ============================================================================ #

@pytest_asyncio.fixture
async def session_store() -> AsyncGenerator[SessionStore, None]:
    """
    Provide a fresh, isolated ``SessionStore`` for each test function.

    The store's module-level singleton is reset before and after each test.
    """
    from agent_state_ledger.router import session_store as store_module

    store = SessionStore()
    store_module._global_store = store
    yield store
    store_module._global_store = None


# ============================================================================ #
#  SnapshotStore fixture                                                        #
# ============================================================================ #

@pytest_asyncio.fixture
async def snapshot_store(tmp_path: Path) -> AsyncGenerator[SnapshotStore, None]:
    """
    Provide an open ``SnapshotStore`` backed by a temp SQLite database.

    Automatically opens and closes the store around each test function.
    """
    db_path = tmp_path / "test_snapshots.db"
    store = SnapshotStore(db_path=db_path, compression_enabled=True)
    await store.open()
    yield store
    await store.close()


# ============================================================================ #
#  SnapshotEngine fixture                                                       #
# ============================================================================ #

@pytest_asyncio.fixture
async def snapshot_engine(
    tmp_path: Path,
    session_store: SessionStore,
) -> AsyncGenerator[SnapshotEngine, None]:
    """
    Provide a started ``SnapshotEngine`` for each test.

    Initialises the global singleton and tears it down after the test.
    """
    from agent_state_ledger.snapshot import engine as engine_module

    db_path = tmp_path / "test_snapshots.db"
    engine = SnapshotEngine(db_path=db_path)
    await engine.start()
    init_snapshot_engine(engine)
    yield engine
    await engine.stop()
    engine_module._engine_instance = None


# ============================================================================ #
#  FastAPI test client fixtures                                                 #
# ============================================================================ #

@pytest_asyncio.fixture
async def async_client(
    snapshot_engine: SnapshotEngine,
    session_store: SessionStore,
) -> AsyncGenerator[AsyncClient, None]:
    """
    Provide an ``httpx.AsyncClient`` targeting the FastAPI app in-process.

    The snapshot engine and session store singletons are pre-initialised
    so the lifespan context manager's startup logic is bypassed.
    """
    from agent_state_ledger.router.main import create_app

    app = create_app()

    # Pre-populate app state so routes can access the store/engine
    app.state.session_store = session_store
    app.state.snapshot_engine = snapshot_engine

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
