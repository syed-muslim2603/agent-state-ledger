"""
sandbox/sandbox_tools.py
==========================
Tool registry for the Agent State Ledger sandbox entrypoint.

This file is packaged **inside** the Docker sandbox image at ``/sandbox/``.
Add tool implementations here that are safe to execute inside the isolated
container environment (no network, read-only filesystem).

Tool registration
-----------------
Every tool is registered in the ``TOOL_REGISTRY`` dict at the bottom of this
file.  The key is the tool name (string) that agents use when calling
``tool_name`` in their payload; the value is the callable that implements
the tool.

Tool function contract
----------------------
* Arguments must be keyword-only and JSON-serialisable.
* Return value must be JSON-serialisable (dict, list, str, int, float, bool,
  or None).
* Raise ``ValueError`` for invalid arguments (caller error → exit code 2).
* Raise any other exception for runtime errors (tool error → exit code 1).
* Write any diagnostic output to ``sys.stderr``, never to ``sys.stdout``,
  because ``stdout`` is reserved for the JSON result protocol.

Example tool
------------
.. code-block:: python

    def my_tool(query: str, max_results: int = 10) -> dict:
        # ... implementation ...
        return {"results": [...]}

    TOOL_REGISTRY["my_tool"] = my_tool
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from typing import Any


# ============================================================================ #
#  Built-in sandbox tools                                                       #
# ============================================================================ #


def echo(message: str) -> dict[str, str]:
    """
    Echo tool — returns the input message unchanged.

    Useful for integration tests and liveness probes.

    Parameters
    ----------
    message:
        Any string value.

    Returns
    -------
    dict[str, str]
        ``{"echo": message}``
    """
    return {"echo": message}


def compute_hash(
    data: str,
    algorithm: str = "sha256",
) -> dict[str, str]:
    """
    Compute a cryptographic hash of the provided string data.

    Parameters
    ----------
    data:
        Input string to hash (UTF-8 encoded before hashing).
    algorithm:
        Hash algorithm to use.  Supported: ``md5``, ``sha1``, ``sha256``,
        ``sha512``.  Defaults to ``sha256``.

    Returns
    -------
    dict[str, str]
        ``{"algorithm": str, "hex_digest": str, "input_length": int}``

    Raises
    ------
    ValueError
        If *algorithm* is not in the supported set.
    """
    supported = {"md5", "sha1", "sha256", "sha512"}
    if algorithm not in supported:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'.  "
            f"Choose from: {sorted(supported)}"
        )

    h = hashlib.new(algorithm)
    h.update(data.encode("utf-8"))
    return {
        "algorithm": algorithm,
        "hex_digest": h.hexdigest(),
        "input_length": len(data),
    }


def parse_json(json_string: str) -> Any:
    """
    Parse a JSON string and return the resulting Python object.

    Parameters
    ----------
    json_string:
        A valid JSON string.

    Returns
    -------
    Any
        The parsed JSON value.

    Raises
    ------
    ValueError
        If *json_string* is not valid JSON.
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc


def calculate(expression: str) -> dict[str, Any]:
    """
    Safely evaluate a mathematical expression using Python's ``math`` module.

    Only numeric literals and functions from the ``math`` module are permitted.
    All other names and builtins are blocked to prevent code injection.

    Parameters
    ----------
    expression:
        A mathematical expression string, e.g. ``"sqrt(2) + log(10)"``.

    Returns
    -------
    dict[str, Any]
        ``{"expression": str, "result": float | int}``

    Raises
    ------
    ValueError
        If the expression contains disallowed characters or fails evaluation.
    """
    # Allow only safe characters: digits, operators, parentheses, dots,
    # letters (for math function names), and whitespace.
    if not re.match(r"^[0-9+\-*/().,\s\w]+$", expression):
        raise ValueError(
            f"Expression contains disallowed characters: {expression!r}"
        )

    # Restrict evaluation namespace to math functions and constants only
    safe_globals: dict[str, Any] = {
        "__builtins__": {},
        **{name: getattr(math, name) for name in dir(math) if not name.startswith("_")},
        "abs": abs,
        "round": round,
        "int": int,
        "float": float,
        "pow": pow,
    }

    try:
        result = eval(expression, safe_globals, {})  # noqa: S307 — controlled sandbox
    except Exception as exc:
        raise ValueError(f"Failed to evaluate expression '{expression}': {exc}") from exc

    if not isinstance(result, (int, float)):
        raise ValueError(
            f"Expression must evaluate to a numeric type, got {type(result).__name__}."
        )

    return {"expression": expression, "result": result}


def text_statistics(text: str) -> dict[str, Any]:
    """
    Compute basic statistics about a text string.

    Parameters
    ----------
    text:
        Input text to analyse.

    Returns
    -------
    dict[str, Any]
        Dictionary with character count, word count, sentence count, and
        average word length.
    """
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "avg_word_length": 0.0,
            "unique_words": 0,
        }

    words = re.findall(r"\b\w+\b", text.lower())
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    avg_word_length = sum(len(w) for w in words) / max(len(words), 1)

    return {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_word_length": round(avg_word_length, 2),
        "unique_words": len(set(words)),
    }


def list_registered_tools() -> dict[str, list[str]]:
    """
    Return the list of all registered tool names.

    Useful for agent introspection — an agent can call this tool to discover
    what capabilities are available in the current sandbox environment.

    Returns
    -------
    dict[str, list[str]]
        ``{"tools": [list of tool names]}``
    """
    return {"tools": sorted(TOOL_REGISTRY.keys())}


# ============================================================================ #
#  Tool registry                                                                #
# ============================================================================ #

TOOL_REGISTRY: dict[str, Any] = {
    "echo": echo,
    "compute_hash": compute_hash,
    "parse_json": parse_json,
    "calculate": calculate,
    "text_statistics": text_statistics,
    "list_registered_tools": list_registered_tools,
}
