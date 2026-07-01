"""
agent_state_ledger.config
=========================
Central, environment-driven configuration for every subsystem of the
Agent State Ledger.  All settings are loaded once at import time via
Pydantic-Settings and cached as a module-level singleton so that
configuration is shared consistently across the entire process.

Environment variables follow the prefix ``ASL_`` and are read from both
the OS environment and an optional ``.env`` file in the working directory.

Example ``.env`` file::

    ASL_ROUTER_HOST=0.0.0.0
    ASL_ROUTER_PORT=8000
    ASL_CONTEXT_TOKEN_BUDGET=8192
    ASL_SNAPSHOT_DB_PATH=/data/snapshots.db
    ASL_INTENT_DEVIATION_THRESHOLD=0.35
    ASL_MAX_ROLLBACK_DEPTH=10
    ASL_SANDBOX_IMAGE=agent-state-ledger-sandbox:latest
    ASL_LOG_LEVEL=INFO
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RouterSettings(BaseSettings):
    """Settings scoped to the FastAPI/FastMCP State Router subsystem."""

    model_config = SettingsConfigDict(env_prefix="ASL_ROUTER_", env_file=".env")

    host: str = Field(
        default="0.0.0.0",
        description="Bind host for the HTTP server.",
    )
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="TCP port the router listens on.",
    )
    workers: int = Field(
        default=4,
        ge=1,
        le=64,
        description="Number of Uvicorn worker processes.",
    )
    request_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        description="Per-request timeout enforced by the router middleware.",
    )
    max_payload_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10 MiB
        ge=1024,
        description="Maximum raw body size accepted per JSON-RPC call.",
    )
    enable_compression: bool = Field(
        default=True,
        description="GZip-compress HTTP responses above the threshold.",
    )
    compression_min_size: int = Field(
        default=4096,
        ge=256,
        description="Minimum response size (bytes) before GZip is applied.",
    )


class ContextSettings(BaseSettings):
    """Settings that govern context-window budget management."""

    model_config = SettingsConfigDict(env_prefix="ASL_CONTEXT_", env_file=".env")

    token_budget: int = Field(
        default=8192,
        ge=512,
        description=(
            "Maximum number of tokens any single agent payload may consume "
            "in a single context injection. Payloads exceeding this limit are "
            "compressed via progressive disclosure."
        ),
    )
    disclosure_levels: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "Number of progressive disclosure tiers.  The router will try "
            "tier-1 (most summarized) first and escalate toward tier-N "
            "(full fidelity) only when downstream agents request expansion."
        ),
    )
    summary_ratio: float = Field(
        default=0.25,
        gt=0.0,
        lt=1.0,
        description=(
            "Fraction of the original payload retained at disclosure tier-1. "
            "Each subsequent tier multiplies this ratio by 2 until the budget "
            "is exhausted or full fidelity is reached."
        ),
    )
    token_count_method: Literal["whitespace", "char_approx", "tiktoken"] = Field(
        default="char_approx",
        description=(
            "Token counting strategy used for budget enforcement.  "
            "'whitespace' splits on whitespace, 'char_approx' divides "
            "character count by 4 (GPT-4 approximation), and 'tiktoken' "
            "uses the exact cl100k_base encoder (requires tiktoken installed)."
        ),
    )


class SnapshotSettings(BaseSettings):
    """Settings for the async State Snapshot engine."""

    model_config = SettingsConfigDict(env_prefix="ASL_SNAPSHOT_", env_file=".env")

    db_path: Path = Field(
        default=Path("data/snapshots.db"),
        description="Filesystem path for the SQLite snapshot store.",
    )
    intent_deviation_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=(
            "Normalized score (0–1) above which an agent's state is considered "
            "to have deviated from its declared intent.  Triggers automatic "
            "rollback when exceeded."
        ),
    )
    max_rollback_depth: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of snapshot generations retained per agent session.",
    )
    checkpoint_interval_seconds: float = Field(
        default=5.0,
        ge=0.01,
        description=(
            "Minimum wall-clock interval between automatic checkpoints for "
            "a given agent session.  Prevents write amplification under "
            "high-frequency state changes."
        ),
    )
    compression_enabled: bool = Field(
        default=True,
        description="Compress snapshot payloads with zlib before persisting.",
    )
    retention_days: int = Field(
        default=7,
        ge=1,
        description="Age (days) after which expired snapshots are purged.",
    )


class SandboxSettings(BaseSettings):
    """Settings for the Docker execution sandbox."""

    model_config = SettingsConfigDict(env_prefix="ASL_SANDBOX_", env_file=".env")

    image: str = Field(
        default="agent-state-ledger-sandbox:latest",
        description="Docker image used to spawn isolated agent runtime containers.",
    )
    cpu_limit: float = Field(
        default=1.0,
        gt=0.0,
        description="Fractional CPU quota allocated to each sandbox container.",
    )
    memory_limit_mb: int = Field(
        default=512,
        ge=64,
        description="Hard memory cap (MiB) per sandbox container.",
    )
    network_mode: Literal["none", "bridge", "host"] = Field(
        default="none",
        description=(
            "Docker network mode for sandbox containers.  'none' disables all "
            "external network access for maximum isolation."
        ),
    )
    execution_timeout_seconds: int = Field(
        default=60,
        ge=5,
        description="Hard wall-clock timeout before a sandbox container is force-killed.",
    )
    read_only_rootfs: bool = Field(
        default=True,
        description="Mount the container's root filesystem as read-only.",
    )
    work_dir: Path = Field(
        default=Path("/workspace"),
        description="Working directory inside the sandbox container.",
    )
    allowed_syscalls: list[str] = Field(
        default_factory=lambda: [
            "read", "write", "open", "close", "stat", "fstat",
            "lstat", "poll", "lseek", "mmap", "mprotect", "munmap",
            "brk", "rt_sigaction", "rt_sigprocmask", "ioctl", "access",
            "pipe", "select", "sched_yield", "mremap", "msync", "mincore",
            "madvise", "dup", "dup2", "nanosleep", "getpid", "socket",
            "connect", "sendto", "recvfrom", "shutdown", "bind", "listen",
            "getsockname", "getpeername", "fork", "execve", "exit",
            "wait4", "kill", "uname", "fcntl", "flock", "fsync",
            "getdents", "getcwd", "chdir", "rename", "mkdir", "rmdir",
            "unlink", "readlink", "chmod", "chown", "umask", "gettimeofday",
            "getrlimit", "getrusage", "sysinfo", "times", "getuid", "getgid",
            "geteuid", "getegid", "setuid", "setgid", "getgroups",
            "sigaltstack", "statfs", "fstatfs", "arch_prctl", "futex",
            "set_tid_address", "clock_gettime", "exit_group", "openat",
            "newfstatat", "set_robust_list", "prlimit64", "getrandom",
        ],
        description="Seccomp allowlist of Linux syscalls available inside the sandbox.",
    )

    @field_validator("image")
    @classmethod
    def _validate_image(cls, value: str) -> str:
        """Ensure the Docker image string is non-empty and well-formed."""
        if not value.strip():
            raise ValueError("Sandbox Docker image name must not be empty.")
        return value.strip()


class ObservabilitySettings(BaseSettings):
    """Logging and metrics settings."""

    model_config = SettingsConfigDict(env_prefix="ASL_", env_file=".env")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Minimum log level for structured stdout output.",
    )
    log_format: Literal["json", "console"] = Field(
        default="json",
        description=(
            "Structured log output format.  Use 'console' for human-readable "
            "development output and 'json' for production log ingestion."
        ),
    )
    metrics_enabled: bool = Field(
        default=True,
        description="Expose Prometheus metrics on /metrics.",
    )
    metrics_port: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Port on which the Prometheus metrics HTTP server listens.",
    )
    trace_header: str = Field(
        default="X-ASL-Trace-ID",
        description="HTTP header name used to propagate distributed trace IDs.",
    )


class Settings(BaseSettings):
    """
    Aggregated top-level settings object.

    Each subsystem settings object is instantiated lazily via ``@lru_cache``
    to ensure environment variables are only read once per process lifecycle.
    This object is the single source of truth for all configuration and
    should be accessed via ``get_settings()`` rather than constructed directly.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ------------------------------------------------------------------ #
    # Compose subsystem settings as nested attributes so callers can write
    # ``settings.router.port`` instead of separate per-subsystem imports.
    # ------------------------------------------------------------------ #

    @property
    def router(self) -> RouterSettings:
        return RouterSettings()

    @property
    def context(self) -> ContextSettings:
        return ContextSettings()

    @property
    def snapshot(self) -> SnapshotSettings:
        return SnapshotSettings()

    @property
    def sandbox(self) -> SandboxSettings:
        return SandboxSettings()

    @property
    def observability(self) -> ObservabilitySettings:
        return ObservabilitySettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the process-wide Settings singleton.

    The result is cached after the first call so that repeated access does
    not re-parse environment variables.  The cache can be cleared in tests
    via ``get_settings.cache_clear()``.

    Returns
    -------
    Settings
        Fully validated, immutable configuration object.
    """
    return Settings()
