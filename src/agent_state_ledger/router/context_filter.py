"""
agent_state_ledger.router.context_filter
========================================
The progressive-disclosure engine — the intellectual core of the State Router.

Responsibility
--------------
Given a raw tool-call output payload and a per-session token budget, this
module computes the highest-fidelity ``DisclosureLevel`` whose token count
fits within the budget, applies the corresponding compression transformation,
and returns a ``FilteredToolCallOutput`` ready for LLM context injection.

Disclosure tiers (defined in ``agent_state_ledger.models.DisclosureLevel``):

    MINIMAL (1)   — Scalar summary: only top-level string/numeric key-value
                    pairs are retained.  Lists are replaced with a count
                    sentinel.  Token reduction: ~75–85%.

    CONDENSED (2) — All top-level keys preserved; list elements capped at 3;
                    nested dicts are replaced with ``{…N keys}`` placeholders.
                    Token reduction: ~40–60%.

    STANDARD (3)  — Full structure preserved; list elements capped at 10;
                    strings truncated at 512 characters.
                    Token reduction: ~20–40%.

    EXPANDED (4)  — Full structure preserved; list elements capped at 50;
                    binary/base64 blobs elided.  Token reduction: ~5–15%.

    FULL (5)      — Verbatim original payload; no budget enforcement.

Progressive algorithm
---------------------
1. Estimate token count of the raw result via the configured method.
2. If raw_tokens ≤ budget → return FULL (no compression).
3. Otherwise, try each level from MINIMAL upward until the estimate fits
   the budget, or until FULL is reached as a fallback.
4. Emit ``ContextMetricsRecord`` for Prometheus instrumentation.

Thread-safety
-------------
The filter is implemented as a stateless function; all session budget
tracking is performed by the router layer which passes ``budget_remaining``
as an argument.
"""

from __future__ import annotations

import math
import re
from typing import Any

import orjson

from agent_state_ledger.config import get_settings
from agent_state_ledger.models import (
    ContextMetricsRecord,
    DisclosureLevel,
    FilteredToolCallOutput,
    ToolCallOutput,
)

# ============================================================================ #
#  Token estimation                                                             #
# ============================================================================ #

_WHITESPACE_RE = re.compile(r"\s+")


def estimate_tokens(text: str, method: str | None = None) -> int:
    """
    Estimate the number of tokens in *text* using the configured strategy.

    Parameters
    ----------
    text:
        The string whose token count should be estimated.
    method:
        Override the globally configured token-count method.  Useful in
        tests.  Accepts ``"whitespace"``, ``"char_approx"``, or
        ``"tiktoken"``.

    Returns
    -------
    int
        Non-negative token count estimate.  Always returns at least 1 for
        non-empty input so that budget arithmetic never divides by zero.
    """
    if not text:
        return 0

    settings = get_settings()
    chosen_method = method or settings.context.token_count_method

    if chosen_method == "whitespace":
        count = len(_WHITESPACE_RE.split(text.strip()))
    elif chosen_method == "tiktoken":
        count = _tiktoken_count(text)
    else:
        # "char_approx" — GPT-4 approximation: ~4 chars per token
        count = math.ceil(len(text) / 4)

    return max(count, 1)


def _tiktoken_count(text: str) -> int:
    """
    Exact token count using the ``cl100k_base`` encoder (GPT-4 / Claude).

    Falls back to the ``char_approx`` method if tiktoken is not installed.
    """
    try:
        import tiktoken  # optional heavy dependency

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return math.ceil(len(text) / 4)


def _payload_to_text(payload: Any) -> str:
    """Serialize *payload* to a UTF-8 string for token estimation."""
    if isinstance(payload, str):
        return payload
    try:
        return orjson.dumps(payload, option=orjson.OPT_NON_STR_KEYS).decode("utf-8")
    except (TypeError, ValueError):
        return str(payload)


# ============================================================================ #
#  Disclosure transformations                                                   #
# ============================================================================ #


def _apply_minimal(payload: Any) -> Any:
    """
    MINIMAL disclosure — tier 1.

    Retain only top-level scalar (str, int, float, bool, None) key-value
    pairs.  Lists become ``"[N items]"`` sentinel strings.  Nested dicts
    become ``"{N keys}"`` sentinel strings.
    """
    if not isinstance(payload, dict):
        # For non-dict payloads (lists, primitives) return a terse summary
        if isinstance(payload, list):
            return f"[{len(payload)} items]"
        raw_str = str(payload)
        return raw_str[:256] + ("…" if len(raw_str) > 256 else "")

    summary: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            # Truncate long strings
            if isinstance(value, str) and len(value) > 256:
                summary[key] = value[:256] + "…"
            else:
                summary[key] = value
        elif isinstance(value, list):
            summary[key] = f"[{len(value)} items]"
        elif isinstance(value, dict):
            summary[key] = f"{{{len(value)} keys}}"
        else:
            summary[key] = f"<{type(value).__name__}>"

    return summary


def _apply_condensed(payload: Any, list_cap: int = 3) -> Any:
    """
    CONDENSED disclosure — tier 2.

    All top-level keys preserved.  List items capped at *list_cap* elements
    with a ``"… N more"`` trailer appended when truncated.  Nested dicts
    replaced by ``{N keys}`` placeholder strings.
    """
    if not isinstance(payload, dict):
        if isinstance(payload, list):
            truncated = payload[:list_cap]
            trailer = [f"… {len(payload) - list_cap} more"] if len(payload) > list_cap else []
            return truncated + trailer  # type: ignore[operator]
        return payload

    condensed: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            truncated = value[:list_cap]
            if len(value) > list_cap:
                truncated = list(truncated) + [f"… {len(value) - list_cap} more"]
            condensed[key] = truncated
        elif isinstance(value, dict):
            condensed[key] = f"{{{len(value)} keys}}"
        elif isinstance(value, str) and len(value) > 512:
            condensed[key] = value[:512] + "…"
        else:
            condensed[key] = value

    return condensed


def _apply_standard(payload: Any, list_cap: int = 10, str_cap: int = 512) -> Any:
    """
    STANDARD disclosure — tier 3.

    Full structure preserved; list items capped at *list_cap*; strings
    truncated at *str_cap* characters.
    """
    if isinstance(payload, dict):
        return {
            key: _apply_standard(value, list_cap, str_cap)
            for key, value in payload.items()
        }
    elif isinstance(payload, list):
        truncated = [_apply_standard(item, list_cap, str_cap) for item in payload[:list_cap]]
        if len(payload) > list_cap:
            truncated.append(f"… {len(payload) - list_cap} more items")
        return truncated
    elif isinstance(payload, str) and len(payload) > str_cap:
        return payload[:str_cap] + "…"
    else:
        return payload


def _apply_expanded(payload: Any, list_cap: int = 50) -> Any:
    """
    EXPANDED disclosure — tier 4.

    Full structure preserved; list items capped at *list_cap*; binary /
    base64 blobs (strings longer than 4096 chars matching base64 pattern)
    are replaced with ``<binary blob N bytes>`` sentinels.
    """
    _BASE64_RE = re.compile(r"^[A-Za-z0-9+/\-_]+=*$")

    def _transform(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _transform(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            result = [_transform(item) for item in obj[:list_cap]]
            if len(obj) > list_cap:
                result.append(f"… {len(obj) - list_cap} more items")
            return result
        elif isinstance(obj, str):
            if len(obj) > 4096 and _BASE64_RE.match(obj):
                return f"<binary blob ~{len(obj) * 3 // 4} bytes>"
            return obj
        elif isinstance(obj, bytes):
            return f"<binary blob {len(obj)} bytes>"
        else:
            return obj

    return _transform(payload)


# ============================================================================ #
#  Public filter API                                                            #
# ============================================================================ #

_LEVEL_TRANSFORMS = {
    DisclosureLevel.MINIMAL: _apply_minimal,
    DisclosureLevel.CONDENSED: _apply_condensed,
    DisclosureLevel.STANDARD: _apply_standard,
    DisclosureLevel.EXPANDED: _apply_expanded,
    DisclosureLevel.FULL: lambda p: p,  # identity — no transformation
}


def apply_progressive_disclosure(
    raw_output: ToolCallOutput,
    budget_remaining: int,
    requested_level: DisclosureLevel = DisclosureLevel.STANDARD,
) -> tuple[FilteredToolCallOutput, ContextMetricsRecord]:
    """
    Apply progressive disclosure to *raw_output* within *budget_remaining* tokens.

    Algorithm
    ---------
    1. Serialize raw result → text → estimate tokens.
    2. If raw_tokens ≤ budget → honour *requested_level* directly (may still
       apply transformation if caller requested < FULL).
    3. Otherwise iterate from MINIMAL → FULL until the transformed payload
       fits the budget or we've exhausted all tiers.
    4. Build and return a ``FilteredToolCallOutput`` and a ``ContextMetricsRecord``.

    Parameters
    ----------
    raw_output:
        Unfiltered ``ToolCallOutput`` from the tool implementation.
    budget_remaining:
        Tokens available in this session's context budget.
    requested_level:
        Caller-preferred disclosure tier.  The router may downgrade this if
        the result does not fit in *budget_remaining*.

    Returns
    -------
    tuple[FilteredToolCallOutput, ContextMetricsRecord]
        The budget-safe filtered output and the corresponding metrics record.
    """
    # ------------------------------------------------------------------ #
    # Step 1: Estimate raw token cost                                     #
    # ------------------------------------------------------------------ #
    raw_text = _payload_to_text(raw_output.result)
    raw_tokens = estimate_tokens(raw_text)

    # ------------------------------------------------------------------ #
    # Step 2: Short-circuit when payload already fits the budget          #
    # ------------------------------------------------------------------ #
    if raw_tokens <= budget_remaining:
        # Even if it fits, we still apply the requested transformation so
        # the caller gets the fidelity level they asked for.
        transform_fn = _LEVEL_TRANSFORMS[requested_level]
        filtered_result = transform_fn(raw_output.result)
        filtered_text = _payload_to_text(filtered_result)
        filtered_tokens = estimate_tokens(filtered_text)

        compression_ratio = filtered_tokens / max(raw_tokens, 1)

        filtered = FilteredToolCallOutput(
            call_id=raw_output.call_id,
            tool_name=raw_output.tool_name,
            agent_id=raw_output.agent_id,
            session_id=raw_output.session_id,
            disclosure_level_applied=requested_level,
            filtered_result=filtered_result,
            raw_token_estimate=raw_tokens,
            filtered_token_estimate=filtered_tokens,
            compression_ratio=compression_ratio,
            budget_remaining=max(budget_remaining - filtered_tokens, 0),
            error=raw_output.error,
        )
        metrics = _build_metrics(raw_output, filtered, budget_remaining)
        return filtered, metrics

    # ------------------------------------------------------------------ #
    # Step 3: Iterate from MINIMAL upward to find the best-fitting tier  #
    # ------------------------------------------------------------------ #
    levels_ascending = sorted(DisclosureLevel, key=lambda lvl: lvl.value)

    chosen_level = DisclosureLevel.MINIMAL
    chosen_result: Any = raw_output.result
    chosen_filtered_tokens = raw_tokens

    for level in levels_ascending:
        transform_fn = _LEVEL_TRANSFORMS[level]
        candidate_result = transform_fn(raw_output.result)
        candidate_text = _payload_to_text(candidate_result)
        candidate_tokens = estimate_tokens(candidate_text)

        chosen_level = level
        chosen_result = candidate_result
        chosen_filtered_tokens = candidate_tokens

        if candidate_tokens <= budget_remaining:
            # This tier fits — stop escalating
            break

    # If we reached FULL and it still doesn't fit, we deliver FULL anyway
    # to avoid data loss; the budget tracker will reflect the overage.
    compression_ratio = chosen_filtered_tokens / max(raw_tokens, 1)

    filtered = FilteredToolCallOutput(
        call_id=raw_output.call_id,
        tool_name=raw_output.tool_name,
        agent_id=raw_output.agent_id,
        session_id=raw_output.session_id,
        disclosure_level_applied=chosen_level,
        filtered_result=chosen_result,
        raw_token_estimate=raw_tokens,
        filtered_token_estimate=chosen_filtered_tokens,
        compression_ratio=compression_ratio,
        budget_remaining=max(budget_remaining - chosen_filtered_tokens, 0),
        error=raw_output.error,
    )
    metrics = _build_metrics(raw_output, filtered, budget_remaining)
    return filtered, metrics


def _build_metrics(
    raw: ToolCallOutput,
    filtered: FilteredToolCallOutput,
    budget_before: int,
) -> ContextMetricsRecord:
    """Construct a ``ContextMetricsRecord`` from a completed filter operation."""
    return ContextMetricsRecord(
        call_id=raw.call_id,
        session_id=raw.session_id,
        agent_id=raw.agent_id,
        tool_name=raw.tool_name,
        raw_tokens=filtered.raw_token_estimate,
        filtered_tokens=filtered.filtered_token_estimate,
        compression_ratio=filtered.compression_ratio,
        disclosure_level=filtered.disclosure_level_applied,
        budget_before=budget_before,
        budget_after=filtered.budget_remaining,
    )
