"""
agent_state_ledger.router.main
================================
FastAPI application factory and Uvicorn entry-point for the State Router.

Application lifecycle
----------------------
1. ``lifespan`` context manager runs **before** the first request is served:
   a. Configure structlog
   b. Initialise the in-memory SessionStore
   c. Initialise and start the SnapshotEngine (opens the SQLite connection,
      runs schema migrations, starts the background checkpoint task)
   d. Start the Prometheus metrics HTTP server on its dedicated port

2. The FastAPI ``app`` instance mounts:
   a. All middleware layers (trace, metrics, payload limit, rate limit)
   b. All route handlers from ``agent_state_ledger.router.routes``
   c. GZip compression middleware for large responses

3. ``lifespan`` shutdown:
   a. Stop the SnapshotEngine background task cleanly
   b. Close the SQLite connection
   c. Stop the Prometheus HTTP server

Running the server
------------------
From the command line (after ``pip install -e .``)::

    asl-router
    # or
    python -m agent_state_ledger.router.main

With environment overrides::

    ASL_ROUTER_PORT=9000 ASL_LOG_LEVEL=DEBUG asl-router

Inside Docker (see docker-compose.yml)::

    docker compose up router
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import start_http_server

from agent_state_ledger.config import get_settings
from agent_state_ledger.logging_setup import configure_logging, get_logger
from agent_state_ledger.router.middleware import (
    AgentRateLimitMiddleware,
    PayloadSizeLimitMiddleware,
    RequestMetricsMiddleware,
    TraceContextMiddleware,
)
from agent_state_ledger.router.routes import router as api_router
from agent_state_ledger.router.session_store import init_session_store
from agent_state_ledger.snapshot.engine import SnapshotEngine, init_snapshot_engine

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Everything before ``yield`` runs at startup; everything after runs at
    shutdown.  Using a single lifespan avoids the deprecated
    ``on_startup`` / ``on_shutdown`` event hooks.
    """
    settings = get_settings()

    # ------------------------------------------------------------------ #
    # STARTUP                                                              #
    # ------------------------------------------------------------------ #
    configure_logging()
    await logger.ainfo("starting_agent_state_ledger_router", version="1.0.0")

    # 1. Session store
    store = init_session_store()
    app.state.session_store = store
    await logger.ainfo("session_store_initialised")

    # 2. Snapshot engine
    engine = SnapshotEngine(db_path=settings.snapshot.db_path)
    await engine.start()
    init_snapshot_engine(engine)
    app.state.snapshot_engine = engine
    await logger.ainfo(
        "snapshot_engine_started",
        db_path=str(settings.snapshot.db_path),
    )

    # 3. Prometheus metrics HTTP server (separate port from the main API)
    if settings.observability.metrics_enabled:
        start_http_server(port=settings.observability.metrics_port)
        await logger.ainfo(
            "prometheus_metrics_server_started",
            port=settings.observability.metrics_port,
        )

    await logger.ainfo(
        "router_ready",
        host=settings.router.host,
        port=settings.router.port,
    )

    # ------------------------------------------------------------------ #
    # HAND OFF TO FastAPI                                                  #
    # ------------------------------------------------------------------ #
    yield

    # ------------------------------------------------------------------ #
    # SHUTDOWN                                                             #
    # ------------------------------------------------------------------ #
    await logger.ainfo("shutting_down_agent_state_ledger_router")
    await engine.stop()
    await logger.ainfo("snapshot_engine_stopped")


def create_app() -> FastAPI:
    """
    Construct and return the configured FastAPI application instance.

    Separated from the module-level ``app`` variable so that test suites can
    call ``create_app()`` to get a fresh application without triggering
    side-effects at import time.

    Returns
    -------
    FastAPI
        Fully configured application with middleware and routes mounted.
    """
    settings = get_settings()

    app = FastAPI(
        title="Agent State Ledger — State Router",
        description=(
            "A production-grade virtual memory controller for multi-agent LLM "
            "environments.  Intercepts tool-call outputs, applies progressive "
            "disclosure to enforce context-window budgets, and provides "
            "transactional rollback via the async Snapshot Engine."
        ),
        version="1.0.0",
        contact={
            "name": "Agent State Ledger Contributors",
            "url": "https://github.com/your-org/agent-state-ledger",
        },
        license_info={"name": "MIT"},
        openapi_tags=[
            {"name": "Sessions",      "description": "Agent session lifecycle management."},
            {"name": "Tool Calls",    "description": "Progressive-disclosure tool-call interception."},
            {"name": "Rollback",      "description": "Snapshot-based state rollback operations."},
            {"name": "JSON-RPC / MCP","description": "MCP-compatible JSON-RPC 2.0 interface."},
            {"name": "Diagnostics",   "description": "Health checks and metrics."},
        ],
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------ #
    # Middleware — order matters: outermost middleware wraps all inner     #
    # ------------------------------------------------------------------ #

    # GZip compression (must be added first / outermost so it wraps everything)
    if settings.router.enable_compression:
        app.add_middleware(
            GZipMiddleware,
            minimum_size=settings.router.compression_min_size,
        )

    # Trace context propagation
    app.add_middleware(TraceContextMiddleware)

    # Request duration / count metrics
    app.add_middleware(RequestMetricsMiddleware)

    # Body size guard
    app.add_middleware(PayloadSizeLimitMiddleware)

    # Per-agent rate limiter (300 req/min default)
    app.add_middleware(AgentRateLimitMiddleware, max_requests_per_minute=300)

    # ------------------------------------------------------------------ #
    # Routes                                                               #
    # ------------------------------------------------------------------ #
    app.include_router(api_router, prefix="/v1")

    return app


#: Module-level application singleton used by Uvicorn and test clients.
app: FastAPI = create_app()


def run_server() -> None:
    """
    CLI entry-point.  Reads configuration from environment / .env and
    starts a production Uvicorn server.

    This function is registered as the ``asl-router`` console script in
    ``pyproject.toml``.
    """
    settings = get_settings()
    obs = settings.observability

    uvicorn.run(
        "agent_state_ledger.router.main:app",
        host=settings.router.host,
        port=settings.router.port,
        workers=settings.router.workers,
        log_level=obs.log_level.lower(),
        access_log=True,
        # Let structlog handle all formatting; disable Uvicorn's default
        # coloured output in production JSON mode.
        use_colors=(obs.log_format == "console"),
        timeout_graceful_shutdown=10,
    )


if __name__ == "__main__":
    run_server()
