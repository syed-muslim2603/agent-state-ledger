"""
Agent State Ledger
==================
A production-grade virtual memory controller for multi-agent LLM environments.

This package exposes three primary subsystems:

    router   — FastAPI/FastMCP State Router that intercepts tool-call outputs,
               applies progressive-disclosure filters, and enforces per-agent
               context budgets before forwarding payloads to the LLM.

    snapshot — Asynchronous State Snapshot engine managing transactional
               checkpoints, intent-deviation detection, and rollback paths.

    sandbox  — Docker-backed execution sandbox that isolates agent runtime
               loops and enforces resource quotas at the cgroup level.

Public version and metadata are accessible via the standard __version__,
__author__, and __description__ attributes.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("agent-state-ledger")
except PackageNotFoundError:
    # Fallback when the package is executed directly from source without
    # being installed (e.g., during local development with `pip install -e .`)
    __version__ = "1.0.0-dev"

__author__: str = "Agent State Ledger Contributors"
__description__: str = (
    "Production-grade virtual memory controller preventing context-window "
    "bloat and state divergence in multi-agent LLM environments."
)

__all__: list[str] = ["__version__", "__author__", "__description__"]
