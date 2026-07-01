"""
agent_state_ledger.logging_setup
=================================
Initializes ``structlog`` with a renderer appropriate for the configured
environment (JSON for production, coloured console for development).

Call ``configure_logging()`` once at application start-up — typically from
the FastAPI lifespan or the CLI entry-point — before any log statements
are emitted.

Structlog is used instead of the standard-library ``logging`` module because
it provides:

* Context-variable binding (agent_id, session_id, trace_id) without
  thread-local state
* First-class async support (``structlog.contextvars``)
* Deterministic JSON serialisation via ``orjson``
* Zero-overhead no-op processors in production for DEBUG-level events

All bound context variables are reset per-request by the FastAPI middleware
defined in ``agent_state_ledger.router.middleware``.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

from agent_state_ledger.config import get_settings


def _add_app_info(
    logger: WrappedLogger,  # noqa: ARG001 — required by structlog processor signature
    method_name: str,  # noqa: ARG001
    event_dict: EventDict,
) -> EventDict:
    """
    Structlog processor that injects static application metadata into every
    log record so downstream aggregators (e.g. Loki, Datadog) can filter
    without needing a wrapper.
    """
    event_dict["app"] = "agent-state-ledger"
    event_dict["version"] = "1.0.0"
    return event_dict


def _drop_color_message_key(
    logger: WrappedLogger,  # noqa: ARG001
    method_name: str,  # noqa: ARG001
    event_dict: EventDict,
) -> EventDict:
    """
    Uvicorn emits a ``color_message`` key alongside its plain ``message``
    key when running in a terminal.  This processor strips it before the
    final renderer sees the event so JSON output stays clean.
    """
    event_dict.pop("color_message", None)
    return event_dict


def configure_logging() -> None:
    """
    Configure structlog and the standard-library ``logging`` bridge.

    Must be called **once** at application startup.  Subsequent calls are
    idempotent and have no effect because structlog tracks whether it has
    already been configured.

    Behaviour
    ---------
    * ``log_format="json"``  → ``structlog.processors.JSONRenderer`` backed
      by ``orjson`` for maximum throughput and RFC-3339 timestamps.
    * ``log_format="console"`` → ``structlog.dev.ConsoleRenderer`` with
      colour-coded levels for local development.
    * The stdlib ``logging`` root logger is bridged into structlog via
      ``structlog.stdlib.ProcessorFormatter`` so that third-party libraries
      (Uvicorn, SQLAlchemy, httpx) emit structured output automatically.
    """
    settings = get_settings()
    obs = settings.observability

    log_level_int: int = getattr(logging, obs.log_level, logging.INFO)

    shared_processors: list[Any] = [
        # Inject caller context variables bound via structlog.contextvars
        structlog.contextvars.merge_contextvars,
        # Attach logger name so records from third-party libs are identifiable
        structlog.stdlib.add_logger_name,
        # Attach log level as a string field
        structlog.stdlib.add_log_level,
        # RFC-3339 / ISO-8601 timestamp
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # Inject app-level metadata
        _add_app_info,
        # Remove Uvicorn color artefacts
        _drop_color_message_key,
        # Render exceptions as structured dicts rather than multi-line strings
        structlog.processors.dict_tracebacks,
    ]

    if obs.log_format == "json":
        final_renderer: Any = structlog.processors.JSONRenderer(serializer=_orjson_dumps)
    else:
        final_renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors
        + [
            # Bridge stdlib log records emitted by third-party libraries
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # ------------------------------------------------------------------ #
    # Bridge stdlib logging → structlog so Uvicorn, httpx, etc. are routed
    # through the same pipeline and appear in the same structured stream.
    # ------------------------------------------------------------------ #
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            final_renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level_int)


def _orjson_dumps(obj: Any, **kwargs: Any) -> str:  # noqa: ANN401
    """
    Thin wrapper around ``orjson.dumps`` that returns a ``str`` rather than
    ``bytes``, which is what structlog's JSONRenderer protocol expects.
    """
    import orjson  # lazy import to avoid hard dep at module-load time

    return orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS).decode("utf-8")


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Retrieve a bound logger instance.

    Parameters
    ----------
    name:
        Optional logger name.  Defaults to the calling module's ``__name__``
        when omitted.

    Returns
    -------
    structlog.stdlib.BoundLogger
        A context-variable-aware logger ready for use.
    """
    return structlog.get_logger(name)
