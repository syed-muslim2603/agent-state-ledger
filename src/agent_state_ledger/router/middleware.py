"""
agent_state_ledger.router.middleware
=====================================
FastAPI middleware stack for the State Router.

Middleware layers (applied outermost-first):

1. ``TraceContextMiddleware``  — Reads or generates a distributed trace ID
   from the ``X-ASL-Trace-ID`` header and binds it to structlog's context
   variable store so every log statement in the request chain automatically
   carries the trace ID.

2. ``RequestMetricsMiddleware`` — Records per-request latency and status code
   as Prometheus metrics (histogram + counter).

3. ``PayloadSizeLimitMiddleware`` — Rejects request bodies that exceed the
   configured ``ASL_ROUTER_MAX_PAYLOAD_BYTES`` limit before they are parsed,
   preventing memory exhaustion from oversized payloads.

4. ``AgentRateLimitMiddleware`` — Enforces a per-agent token-bucket rate
   limit to prevent a single misbehaving agent from monopolising the router.

All middleware are implemented as ``starlette.middleware.base.BaseHTTPMiddleware``
subclasses for compatibility with the full ASGI stack.
"""

from __future__ import annotations

import time
import uuid
from typing import Awaitable, Callable

import structlog
from fastapi import Request, Response
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from agent_state_ledger.config import get_settings
from agent_state_ledger.logging_setup import get_logger

logger = get_logger(__name__)

# ============================================================================ #
#  Prometheus instruments — defined at module level so they are registered     #
#  exactly once with the default Prometheus registry.                          #
# ============================================================================ #

REQUEST_DURATION = Histogram(
    "asl_router_request_duration_seconds",
    "HTTP request duration in seconds, labelled by method, path, and status.",
    labelnames=["method", "path", "status"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

REQUEST_COUNT = Counter(
    "asl_router_requests_total",
    "Total number of HTTP requests handled, labelled by method, path, and status.",
    labelnames=["method", "path", "status"],
)

PAYLOAD_REJECTED_COUNT = Counter(
    "asl_router_payload_rejected_total",
    "Number of requests rejected due to oversized payload.",
    labelnames=["agent_id"],
)


# ============================================================================ #
#  Middleware implementations                                                   #
# ============================================================================ #


class TraceContextMiddleware(BaseHTTPMiddleware):
    """
    Propagate or generate a distributed trace ID per request.

    Reads the ``X-ASL-Trace-ID`` header from the incoming request.  If the
    header is absent a new UUID-4 is generated.  The trace ID is:

    * Bound to ``structlog.contextvars`` so it appears in every log line
      emitted during this request's lifetime.
    * Written back to the ``X-ASL-Trace-ID`` response header so upstream
      load-balancers and clients can correlate requests.
    * Stored on ``request.state.trace_id`` for use by path operation
      functions that need to forward it in outbound calls.

    The structlog context is reset at the end of every request to prevent
    context bleeding between concurrent async requests.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        settings = get_settings()
        self._header_name = settings.observability.trace_header

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        trace_id = request.headers.get(self._header_name) or str(uuid.uuid4())
        request.state.trace_id = trace_id

        # Bind to structlog context variables for the duration of this request
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
        )

        try:
            response = await call_next(request)
        finally:
            # Always clear context, even on exception, to prevent leakage
            structlog.contextvars.clear_contextvars()

        response.headers[self._header_name] = trace_id
        return response


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    """
    Record HTTP request latency and count as Prometheus metrics.

    Labels: ``method``, ``path`` (templated route, not full URL to avoid
    high-cardinality explosions), ``status`` (integer HTTP status code).

    The path label is derived from ``request.scope.get("route")`` which
    gives the FastAPI route pattern (e.g. ``/sessions/{session_id}``) rather
    than the concrete URL (e.g. ``/sessions/01HWX…``).  This is critical for
    keeping Prometheus cardinality bounded regardless of traffic volume.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start_ns = time.perf_counter_ns()
        response = await call_next(request)
        elapsed_s = (time.perf_counter_ns() - start_ns) / 1e9

        # Derive a low-cardinality path label from the matched FastAPI route
        route = request.scope.get("route")
        path_label = route.path if route else request.url.path  # type: ignore[union-attr]
        status_label = str(response.status_code)
        method_label = request.method

        REQUEST_DURATION.labels(
            method=method_label,
            path=path_label,
            status=status_label,
        ).observe(elapsed_s)

        REQUEST_COUNT.labels(
            method=method_label,
            path=path_label,
            status=status_label,
        ).inc()

        return response


class PayloadSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Reject request bodies exceeding ``ASL_ROUTER_MAX_PAYLOAD_BYTES``.

    Reads the ``Content-Length`` header for a fast O(1) check.  When
    ``Content-Length`` is absent (chunked transfer encoding), the body is
    read up to the limit and buffered; this prevents streaming uploads from
    bypassing the guard.

    Returns HTTP 413 Request Entity Too Large with a structured JSON body
    if the limit is exceeded.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._limit = get_settings().router.max_payload_bytes

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Fast path: Content-Length header available
        content_length_header = request.headers.get("content-length")
        if content_length_header is not None:
            try:
                declared_size = int(content_length_header)
            except ValueError:
                declared_size = 0

            if declared_size > self._limit:
                agent_id = request.headers.get("X-ASL-Agent-ID", "unknown")
                PAYLOAD_REJECTED_COUNT.labels(agent_id=agent_id).inc()
                await logger.awarning(
                    "payload_size_limit_exceeded",
                    declared_size=declared_size,
                    limit=self._limit,
                    agent_id=agent_id,
                )
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "payload_too_large",
                        "detail": (
                            f"Request body of {declared_size} bytes exceeds "
                            f"the maximum allowed size of {self._limit} bytes."
                        ),
                        "limit_bytes": self._limit,
                    },
                )

        return await call_next(request)


class AgentRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Per-agent request rate limiter using a fixed-window counter.

    Tracks request counts per ``X-ASL-Agent-ID`` header within a 60-second
    sliding window.  When an agent exceeds ``max_requests_per_minute`` the
    middleware returns HTTP 429 Too Many Requests.

    This is a best-effort in-process limiter suitable for single-node
    deployments.  For multi-node deployments, replace the in-memory counters
    with a shared Redis counter (see the architecture notes in README.md).

    Parameters
    ----------
    max_requests_per_minute:
        Hard limit on requests per agent per 60-second window.
        Default: 300 req/min (~5 req/s).
    """

    def __init__(self, app: ASGIApp, max_requests_per_minute: int = 300) -> None:
        super().__init__(app)
        self._max_rpm = max_requests_per_minute
        # {agent_id: (window_start_timestamp, request_count)}
        self._windows: dict[str, tuple[float, int]] = {}

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        agent_id = request.headers.get("X-ASL-Agent-ID", "anonymous")
        now = time.monotonic()
        window_start, count = self._windows.get(agent_id, (now, 0))

        if now - window_start > 60.0:
            # New window
            self._windows[agent_id] = (now, 1)
        elif count >= self._max_rpm:
            retry_after = int(60.0 - (now - window_start)) + 1
            await logger.awarning(
                "rate_limit_exceeded",
                agent_id=agent_id,
                count=count,
                window_start=window_start,
            )
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry_after)},
                content={
                    "error": "rate_limit_exceeded",
                    "detail": (
                        f"Agent '{agent_id}' has exceeded {self._max_rpm} "
                        f"requests per minute.  Retry after {retry_after}s."
                    ),
                    "retry_after_seconds": retry_after,
                },
            )
        else:
            self._windows[agent_id] = (window_start, count + 1)

        return await call_next(request)
