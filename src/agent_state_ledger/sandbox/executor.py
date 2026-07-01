"""
agent_state_ledger.sandbox.executor
======================================
Docker-backed execution sandbox for isolating agent runtime loops.

Architecture overview
---------------------
Every agent that requires executing untrusted or unpredictable code (tool
implementations, sub-agent scripts, dynamic reasoning chains) is run inside
an ephemeral Docker container managed by this module.

Security posture
----------------
Each sandbox container is launched with the following hardening flags
(all configurable via ``SandboxSettings``):

* ``--network none``           — No external network access.
* ``--read-only``              — Root filesystem is mounted read-only.
* ``--memory`` / ``--cpus``    — Hard cgroup resource limits.
* ``--cap-drop ALL``           — All Linux capabilities dropped.
* ``--security-opt no-new-privileges`` — Prevents privilege escalation via setuid.
* ``--security-opt seccomp:<profile>`` — Strict seccomp allowlist.
* ``--rm``                     — Container removed immediately on exit.
* ``--user 65534:65534``       — Runs as ``nobody:nogroup`` (non-root).

Execution lifecycle
-------------------
1. ``SandboxExecutor.run()`` is called with an agent payload dict.
2. The payload is JSON-serialised and passed as a base64-encoded environment
   variable (``ASL_AGENT_PAYLOAD``) to avoid shell injection.
3. The container executes ``/sandbox/entrypoint.py``, which imports and
   calls the registered agent tool function.
4. stdout is captured; stderr is captured separately for error reporting.
5. The container exits; its stdout is parsed as JSON and returned as the
   tool result.
6. If the container exceeds ``execution_timeout_seconds`` it is force-killed
   (``docker kill``) and a ``SandboxTimeoutError`` is raised.

This module requires the Docker CLI to be present in ``$PATH`` and the
Docker daemon to be accessible (either socket or remote via ``DOCKER_HOST``).

For production deployments, consider replacing the subprocess-based Docker
invocation with the ``docker-py`` SDK (``pip install docker``) for better
error handling and event streaming.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_state_ledger.config import get_settings
from agent_state_ledger.logging_setup import get_logger

logger = get_logger(__name__)


# ============================================================================ #
#  Custom exceptions                                                            #
# ============================================================================ #


class SandboxError(RuntimeError):
    """Base class for all sandbox execution errors."""


class SandboxTimeoutError(SandboxError):
    """Raised when a container exceeds its execution time budget."""


class SandboxRuntimeError(SandboxError):
    """
    Raised when the container exits with a non-zero exit code.

    Attributes
    ----------
    exit_code:
        The container's exit code.
    stderr:
        Captured stderr output from the container.
    """

    def __init__(self, message: str, exit_code: int, stderr: str) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr


class SandboxUnavailableError(SandboxError):
    """
    Raised when the Docker daemon is unreachable or ``docker`` is not in
    ``$PATH``.  Used to provide a clear actionable error message rather than
    a cryptic ``FileNotFoundError``.
    """


# ============================================================================ #
#  Result dataclass                                                             #
# ============================================================================ #


@dataclass(frozen=True)
class SandboxExecutionResult:
    """
    Immutable result of a completed sandbox execution.

    Attributes
    ----------
    container_id:
        The short container ID assigned by Docker (first 12 chars).
    exit_code:
        Process exit code of the container's entrypoint.
    stdout:
        Raw stdout bytes captured from the container.
    stderr:
        Raw stderr bytes captured from the container.
    result:
        Parsed JSON result extracted from stdout.  ``None`` if parsing
        failed or the process exited with a non-zero code.
    execution_duration_ms:
        Wall-clock duration of the container run in milliseconds.
    started_at:
        UTC datetime when the container was launched.
    finished_at:
        UTC datetime when the container exited or was force-killed.
    timed_out:
        ``True`` when the container was killed due to timeout.
    image_used:
        Docker image name that was run.
    metadata:
        Arbitrary caller-supplied metadata forwarded without modification.
    """

    container_id: str
    exit_code: int
    stdout: bytes
    stderr: bytes
    result: Any
    execution_duration_ms: float
    started_at: datetime
    finished_at: datetime
    timed_out: bool
    image_used: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================ #
#  Seccomp profile helper                                                       #
# ============================================================================ #


def _build_seccomp_profile(allowed_syscalls: list[str]) -> dict[str, Any]:
    """
    Build a Docker-compatible seccomp profile JSON object from an allowlist
    of syscall names.

    The profile uses a default action of ``SCMP_ACT_ERRNO`` (return EPERM)
    for all syscalls not in the allow list.

    Parameters
    ----------
    allowed_syscalls:
        List of syscall names to permit.  Typically sourced from
        ``SandboxSettings.allowed_syscalls``.

    Returns
    -------
    dict[str, Any]
        A seccomp profile dict suitable for serialisation and writing to a
        temporary file passed to ``docker run --security-opt seccomp:``.
    """
    return {
        "defaultAction": "SCMP_ACT_ERRNO",
        "architectures": ["SCMP_ARCH_X86_64", "SCMP_ARCH_X86", "SCMP_ARCH_X32"],
        "syscalls": [
            {
                "names": allowed_syscalls,
                "action": "SCMP_ACT_ALLOW",
            }
        ],
    }


# ============================================================================ #
#  SandboxExecutor                                                              #
# ============================================================================ #


class SandboxExecutor:
    """
    Manages ephemeral Docker container lifecycles for sandboxed agent execution.

    One ``SandboxExecutor`` instance may be used concurrently for multiple
    execution requests — each call to ``run()`` spawns an independent
    container.  There is no shared state between concurrent executions.

    Parameters
    ----------
    seccomp_profile_dir:
        Directory where the generated seccomp profile JSON is written.
        Defaults to a ``sandbox/`` subdirectory inside the working directory.
    """

    def __init__(self, seccomp_profile_dir: Path | None = None) -> None:
        self._settings = get_settings().sandbox
        self._seccomp_dir = seccomp_profile_dir or (Path.cwd() / "sandbox" / "seccomp")
        self._seccomp_profile_path: Path | None = None

    async def ensure_seccomp_profile(self) -> Path:
        """
        Write the seccomp allowlist profile to disk once and cache the path.

        The profile is written lazily on the first call and reused on all
        subsequent calls.  This avoids repeated disk I/O on the hot path.

        Returns
        -------
        Path
            Absolute path to the seccomp JSON profile file.
        """
        if self._seccomp_profile_path is not None:
            return self._seccomp_profile_path

        self._seccomp_dir.mkdir(parents=True, exist_ok=True)
        profile_path = self._seccomp_dir / "asl-sandbox-seccomp.json"

        profile = _build_seccomp_profile(self._settings.allowed_syscalls)
        profile_path.write_text(json.dumps(profile, indent=2))

        self._seccomp_profile_path = profile_path
        await logger.ainfo(
            "seccomp_profile_written",
            path=str(profile_path),
            syscall_count=len(self._settings.allowed_syscalls),
        )
        return profile_path

    def _build_docker_command(
        self,
        container_name: str,
        payload_b64: str,
        seccomp_profile_path: Path,
        env_overrides: dict[str, str] | None = None,
    ) -> list[str]:
        """
        Construct the ``docker run`` command as a list of string tokens.

        Using a list (rather than a shell string) prevents shell injection
        because ``asyncio.create_subprocess_exec`` bypasses the shell entirely.

        Parameters
        ----------
        container_name:
            Unique name assigned to this container run.
        payload_b64:
            Base64-encoded JSON agent payload passed via environment variable.
        seccomp_profile_path:
            Path to the seccomp JSON profile file on the host.
        env_overrides:
            Optional additional environment variables to inject into the
            container.  These are merged with the built-in ``ASL_AGENT_PAYLOAD``
            variable.

        Returns
        -------
        list[str]
            Fully assembled ``docker run`` command tokens.
        """
        cmd = [
            "docker", "run",
            "--name", container_name,
            "--rm",                                     # Auto-remove on exit
            "--network", self._settings.network_mode,   # No external network
            "--memory", f"{self._settings.memory_limit_mb}m",
            "--cpus", str(self._settings.cpu_limit),
            "--cap-drop", "ALL",                        # Drop all capabilities
            "--security-opt", "no-new-privileges",
            "--security-opt", f"seccomp={seccomp_profile_path}",
            "--user", "65534:65534",                    # nobody:nogroup
            "--workdir", str(self._settings.work_dir),
            "--env", f"ASL_AGENT_PAYLOAD={payload_b64}",
            "--env", "PYTHONUNBUFFERED=1",
        ]

        if self._settings.read_only_rootfs:
            cmd.append("--read-only")

        # Inject additional environment overrides
        if env_overrides:
            for key, value in env_overrides.items():
                cmd.extend(["--env", f"{key}={value}"])

        cmd.append(self._settings.image)
        return cmd

    async def run(
        self,
        agent_payload: dict[str, Any],
        session_id: str,
        env_overrides: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SandboxExecutionResult:
        """
        Execute *agent_payload* inside an isolated Docker container.

        Parameters
        ----------
        agent_payload:
            Arbitrary JSON-serialisable dict passed to the sandbox entrypoint.
            Typically contains the tool name, arguments, and session context.
        session_id:
            The calling agent's session ID — embedded in the container name
            for traceability in ``docker ps`` output.
        env_overrides:
            Additional environment variables to inject into the container.
        metadata:
            Caller metadata forwarded into the ``SandboxExecutionResult``.

        Returns
        -------
        SandboxExecutionResult
            Structured result including stdout, stderr, exit code, and
            parsed JSON result.

        Raises
        ------
        SandboxUnavailableError
            If the ``docker`` binary is not in ``$PATH``.
        SandboxTimeoutError
            If the container exceeds ``execution_timeout_seconds``.
        SandboxRuntimeError
            If the container exits with a non-zero exit code.
        """
        settings = self._settings
        container_name = f"asl-{session_id[:8]}-{uuid.uuid4().hex[:8]}"
        payload_b64 = base64.b64encode(
            json.dumps(agent_payload).encode("utf-8")
        ).decode("ascii")

        # Ensure seccomp profile is on disk
        seccomp_path = await self.ensure_seccomp_profile()

        cmd = self._build_docker_command(
            container_name=container_name,
            payload_b64=payload_b64,
            seccomp_profile_path=seccomp_path,
            env_overrides=env_overrides,
        )

        await logger.ainfo(
            "sandbox_container_starting",
            session_id=session_id,
            container_name=container_name,
            image=settings.image,
            network_mode=settings.network_mode,
            memory_mb=settings.memory_limit_mb,
            cpu_limit=settings.cpu_limit,
        )

        started_at = datetime.now(tz=timezone.utc)
        start_mono = asyncio.get_event_loop().time()

        # ------------------------------------------------------------------ #
        # Launch the container process                                         #
        # ------------------------------------------------------------------ #
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise SandboxUnavailableError(
                "Docker binary not found in PATH.  Ensure Docker is installed "
                "and the daemon is running before using the sandbox executor."
            ) from exc

        # ------------------------------------------------------------------ #
        # Enforce wall-clock timeout                                           #
        # ------------------------------------------------------------------ #
        timed_out = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=float(settings.execution_timeout_seconds),
            )
        except asyncio.TimeoutError:
            timed_out = True
            stdout_bytes = b""
            stderr_bytes = b""

            # Force-kill the container using the named container reference
            await logger.awarning(
                "sandbox_container_timeout",
                container_name=container_name,
                timeout_seconds=settings.execution_timeout_seconds,
            )
            try:
                kill_proc = await asyncio.create_subprocess_exec(
                    "docker", "kill", container_name,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await kill_proc.wait()
            except Exception as kill_exc:
                await logger.aerror(
                    "sandbox_force_kill_failed",
                    container_name=container_name,
                    error=str(kill_exc),
                )

            proc.kill()

        finished_at = datetime.now(tz=timezone.utc)
        elapsed_ms = (asyncio.get_event_loop().time() - start_mono) * 1000.0
        exit_code = proc.returncode if proc.returncode is not None else -1

        # ------------------------------------------------------------------ #
        # Parse result from stdout                                             #
        # ------------------------------------------------------------------ #
        parsed_result: Any = None
        if not timed_out and exit_code == 0 and stdout_bytes:
            try:
                parsed_result = json.loads(stdout_bytes.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                # Non-JSON stdout — return raw string
                parsed_result = stdout_bytes.decode("utf-8", errors="replace")

        result = SandboxExecutionResult(
            container_id=container_name,
            exit_code=exit_code,
            stdout=stdout_bytes,
            stderr=stderr_bytes,
            result=parsed_result,
            execution_duration_ms=elapsed_ms,
            started_at=started_at,
            finished_at=finished_at,
            timed_out=timed_out,
            image_used=settings.image,
            metadata=metadata or {},
        )

        await logger.ainfo(
            "sandbox_container_finished",
            container_name=container_name,
            exit_code=exit_code,
            duration_ms=round(elapsed_ms, 2),
            timed_out=timed_out,
            stdout_bytes=len(stdout_bytes),
            stderr_bytes=len(stderr_bytes),
        )

        # ------------------------------------------------------------------ #
        # Raise on error conditions                                            #
        # ------------------------------------------------------------------ #
        if timed_out:
            raise SandboxTimeoutError(
                f"Sandbox container '{container_name}' exceeded the "
                f"{settings.execution_timeout_seconds}s execution timeout."
            )

        if exit_code != 0:
            stderr_str = stderr_bytes.decode("utf-8", errors="replace")
            raise SandboxRuntimeError(
                f"Sandbox container '{container_name}' exited with code {exit_code}.",
                exit_code=exit_code,
                stderr=stderr_str,
            )

        return result

    async def pull_image(self) -> None:
        """
        Pull the configured sandbox Docker image from the registry.

        Call this during application startup to ensure the image is present
        before the first ``run()`` invocation.  Avoids the latency of a cold
        pull during a live request.

        Raises
        ------
        SandboxUnavailableError
            If ``docker`` is not in ``$PATH``.
        """
        image = self._settings.image
        await logger.ainfo("sandbox_image_pull_starting", image=image)
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "pull", image,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr_bytes = await proc.communicate()
        except FileNotFoundError as exc:
            raise SandboxUnavailableError(
                "Docker binary not found in PATH."
            ) from exc

        if proc.returncode != 0:
            await logger.awarning(
                "sandbox_image_pull_failed",
                image=image,
                stderr=stderr_bytes.decode("utf-8", errors="replace")[:512],
            )
        else:
            await logger.ainfo("sandbox_image_pull_complete", image=image)

    async def check_docker_available(self) -> bool:
        """
        Return ``True`` if the Docker daemon is reachable, ``False`` otherwise.

        Used by the health-check endpoint to report sandbox availability.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return proc.returncode == 0
        except (FileNotFoundError, OSError):
            return False
