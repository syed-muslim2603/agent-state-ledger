#!/usr/bin/env python3
"""
sandbox/entrypoint.py
======================
Minimal entrypoint script that runs **inside** the Docker sandbox container.

This script is the sole process that executes inside the ephemeral container.
It:

1. Reads the base64-encoded JSON agent payload from the ``ASL_AGENT_PAYLOAD``
   environment variable.
2. Validates that the payload contains a ``tool_name`` and ``arguments`` field.
3. Dispatches the call to the matching registered tool function from
   ``sandbox_tools.py`` (located alongside this script inside the image at
   ``/sandbox/``).
4. Serialises the tool's return value to JSON and writes it to stdout.
5. Exits 0 on success, 1 on tool error, 2 on payload validation error.

Security notes
--------------
* This script has no network access (``--network none``).
* The root filesystem is read-only; only ``/tmp`` is writable (via tmpfs).
* It runs as UID 65534 (``nobody``) with no Linux capabilities.
* It imports only stdlib modules and the co-located ``sandbox_tools`` module;
  no internet calls can be made at import time.

Stdout protocol
---------------
The caller (``SandboxExecutor.run()``) expects stdout to be a single-line
JSON object.  Any additional output (print statements in tool code) must be
written to stderr instead.  Non-JSON stdout will be returned as a raw string.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import traceback
from typing import Any


def _load_payload() -> dict[str, Any]:
    """
    Decode and validate the agent payload from the environment.

    Returns
    -------
    dict[str, Any]
        Validated payload containing at minimum ``tool_name`` and
        ``arguments``.

    Raises
    ------
    SystemExit(2)
        If the environment variable is missing, malformed, or the payload
        fails schema validation.
    """
    raw_b64 = os.environ.get("ASL_AGENT_PAYLOAD")
    if not raw_b64:
        sys.stderr.write(
            "FATAL: ASL_AGENT_PAYLOAD environment variable is not set.\n"
        )
        sys.exit(2)

    try:
        decoded_bytes = base64.b64decode(raw_b64)
        payload = json.loads(decoded_bytes.decode("utf-8"))
    except Exception as exc:
        sys.stderr.write(f"FATAL: Failed to decode ASL_AGENT_PAYLOAD: {exc}\n")
        sys.exit(2)

    if not isinstance(payload, dict):
        sys.stderr.write("FATAL: ASL_AGENT_PAYLOAD must be a JSON object.\n")
        sys.exit(2)

    for required_field in ("tool_name", "arguments"):
        if required_field not in payload:
            sys.stderr.write(
                f"FATAL: Required field '{required_field}' missing from payload.\n"
            )
            sys.exit(2)

    return payload


def _dispatch(payload: dict[str, Any]) -> Any:
    """
    Dynamically import ``sandbox_tools`` and call the requested tool.

    Tool functions are registered in ``/sandbox/sandbox_tools.py`` as a
    ``TOOL_REGISTRY`` dict mapping tool names to callables.

    Parameters
    ----------
    payload:
        Validated payload dict with ``tool_name`` and ``arguments``.

    Returns
    -------
    Any
        JSON-serialisable return value from the tool function.

    Raises
    ------
    ValueError
        If the requested tool name is not registered.
    """
    # Add /sandbox to the path so we can import sandbox_tools
    sandbox_dir = os.path.dirname(os.path.abspath(__file__))
    if sandbox_dir not in sys.path:
        sys.path.insert(0, sandbox_dir)

    try:
        import sandbox_tools  # type: ignore[import]
    except ImportError as exc:
        raise ValueError(
            f"sandbox_tools module not found at {sandbox_dir}: {exc}"
        ) from exc

    registry = getattr(sandbox_tools, "TOOL_REGISTRY", {})
    tool_name = payload["tool_name"]

    if tool_name not in registry:
        raise ValueError(
            f"Unknown tool '{tool_name}'.  "
            f"Registered tools: {sorted(registry.keys())}"
        )

    tool_fn = registry[tool_name]
    arguments = payload.get("arguments", {})

    # Call the tool with keyword arguments
    return tool_fn(**arguments)


def main() -> None:
    """
    Main entrypoint.  Loads the payload, dispatches the tool call, and
    writes the JSON result to stdout.
    """
    payload = _load_payload()
    tool_name = payload.get("tool_name", "unknown")

    try:
        result = _dispatch(payload)
    except ValueError as exc:
        # Tool not found or argument mismatch — exit code 2 (caller error)
        sys.stderr.write(f"ERROR [{tool_name}]: {exc}\n")
        sys.exit(2)
    except Exception:
        # Tool raised an unexpected exception — exit code 1 (tool error)
        sys.stderr.write(f"ERROR [{tool_name}] Unhandled exception:\n")
        sys.stderr.write(traceback.format_exc())
        sys.exit(1)

    # Serialise result to stdout
    try:
        output = json.dumps(result, default=str)
    except (TypeError, ValueError) as exc:
        sys.stderr.write(
            f"ERROR [{tool_name}] Failed to serialise result to JSON: {exc}\n"
        )
        sys.exit(1)

    sys.stdout.write(output)
    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
