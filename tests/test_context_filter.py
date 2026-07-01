"""
tests/test_context_filter.py
==============================
Unit tests for the progressive-disclosure context filter engine.

Tests cover:
- Token estimation for all three configured methods.
- Each individual disclosure transformation tier (MINIMAL through FULL).
- The full ``apply_progressive_disclosure`` pipeline with budget enforcement.
- Edge cases: empty payloads, non-dict payloads, exact budget fits, budget overages.
"""

from __future__ import annotations

import pytest

from agent_state_ledger.models import (
    DisclosureLevel,
    ToolCallOutput,
)
from agent_state_ledger.router.context_filter import (
    _apply_condensed,
    _apply_expanded,
    _apply_minimal,
    _apply_standard,
    apply_progressive_disclosure,
    estimate_tokens,
)


# ============================================================================ #
#  Token estimation tests                                                       #
# ============================================================================ #

class TestEstimateTokens:
    """Tests for the ``estimate_tokens`` function."""

    def test_empty_string_returns_zero(self) -> None:
        assert estimate_tokens("") == 0

    def test_whitespace_method_counts_words(self) -> None:
        text = "hello world foo bar"
        result = estimate_tokens(text, method="whitespace")
        assert result == 4

    def test_char_approx_method(self) -> None:
        # 40 characters → ceil(40/4) = 10 tokens
        text = "a" * 40
        result = estimate_tokens(text, method="char_approx")
        assert result == 10

    def test_char_approx_minimum_one(self) -> None:
        # Even a single character should return at least 1
        result = estimate_tokens("x", method="char_approx")
        assert result >= 1

    def test_single_word_whitespace(self) -> None:
        assert estimate_tokens("hello", method="whitespace") == 1

    def test_multi_line_text(self) -> None:
        text = "line one\nline two\nline three"
        result = estimate_tokens(text, method="whitespace")
        # "line", "one", "line", "two", "line", "three" = 6 tokens
        assert result == 6


# ============================================================================ #
#  Disclosure transformation tests                                              #
# ============================================================================ #

class TestMinimalDisclosure:
    """Tests for the MINIMAL (tier-1) transformation."""

    def test_scalars_preserved(self) -> None:
        payload = {"name": "Alice", "score": 42, "active": True, "tag": None}
        result = _apply_minimal(payload)
        assert result["name"] == "Alice"
        assert result["score"] == 42
        assert result["active"] is True
        assert result["tag"] is None

    def test_lists_become_sentinels(self) -> None:
        payload = {"items": [1, 2, 3, 4, 5]}
        result = _apply_minimal(payload)
        assert "[5 items]" in result["items"]

    def test_nested_dicts_become_sentinels(self) -> None:
        payload = {"metadata": {"a": 1, "b": 2, "c": 3}}
        result = _apply_minimal(payload)
        assert "{3 keys}" in result["metadata"]

    def test_long_strings_truncated(self) -> None:
        payload = {"description": "x" * 500}
        result = _apply_minimal(payload)
        assert len(result["description"]) <= 260  # 256 chars + "…"
        assert result["description"].endswith("…")

    def test_non_dict_list_payload(self) -> None:
        result = _apply_minimal([1, 2, 3, 4, 5, 6])
        assert "[6 items]" in result

    def test_non_dict_string_payload(self) -> None:
        result = _apply_minimal("hello world")
        assert "hello world" in result


class TestCondensedDisclosure:
    """Tests for the CONDENSED (tier-2) transformation."""

    def test_lists_capped_at_three(self) -> None:
        payload = {"items": list(range(10))}
        result = _apply_condensed(payload, list_cap=3)
        items = result["items"]
        # First 3 items + "... N more" trailer
        assert len(items) == 4
        assert "more" in str(items[-1])

    def test_lists_within_cap_unchanged(self) -> None:
        payload = {"items": [1, 2]}
        result = _apply_condensed(payload, list_cap=3)
        assert result["items"] == [1, 2]

    def test_nested_dicts_become_placeholders(self) -> None:
        payload = {"meta": {"x": 1, "y": 2}}
        result = _apply_condensed(payload)
        assert "{2 keys}" in result["meta"]

    def test_long_strings_truncated_at_512(self) -> None:
        payload = {"text": "a" * 1000}
        result = _apply_condensed(payload)
        assert result["text"].endswith("…")
        assert len(result["text"]) <= 515  # 512 + "…"

    def test_non_dict_list_payload(self) -> None:
        result = _apply_condensed(list(range(20)), list_cap=3)
        assert len(result) == 4  # 3 items + trailer


class TestStandardDisclosure:
    """Tests for the STANDARD (tier-3) transformation."""

    def test_deeply_nested_dict_preserved(self) -> None:
        payload = {"a": {"b": {"c": "deep"}}}
        result = _apply_standard(payload)
        assert result["a"]["b"]["c"] == "deep"

    def test_lists_capped_at_ten(self) -> None:
        payload = {"items": list(range(25))}
        result = _apply_standard(payload, list_cap=10)
        assert len(result["items"]) == 11  # 10 items + trailer
        assert "more" in str(result["items"][-1])

    def test_strings_truncated_at_512(self) -> None:
        payload = {"text": "b" * 1000}
        result = _apply_standard(payload, str_cap=512)
        assert result["text"].endswith("…")
        assert len(result["text"]) <= 515

    def test_nested_list_of_dicts(self) -> None:
        payload = {"results": [{"id": i, "name": f"item-{i}"} for i in range(15)]}
        result = _apply_standard(payload, list_cap=10)
        assert len(result["results"]) == 11


class TestExpandedDisclosure:
    """Tests for the EXPANDED (tier-4) transformation."""

    def test_binary_blob_elided(self) -> None:
        # A long base64-like string should be replaced
        fake_b64 = "A" * 5000
        payload = {"blob": fake_b64}
        result = _apply_expanded(payload)
        assert "<binary blob" in result["blob"]

    def test_normal_strings_preserved(self) -> None:
        payload = {"text": "hello world"}
        result = _apply_expanded(payload)
        assert result["text"] == "hello world"

    def test_lists_capped_at_fifty(self) -> None:
        payload = {"items": list(range(100))}
        result = _apply_expanded(payload, list_cap=50)
        assert len(result["items"]) == 51  # 50 + trailer


# ============================================================================ #
#  Progressive disclosure pipeline tests                                        #
# ============================================================================ #

class TestApplyProgressiveDisclosure:
    """Integration tests for the full ``apply_progressive_disclosure`` pipeline."""

    def _make_raw_output(self, result: object, call_id: str = "call-001") -> ToolCallOutput:
        """Helper to create a ``ToolCallOutput``."""
        return ToolCallOutput(
            call_id=call_id,
            tool_name="test_tool",
            agent_id="agent-001",
            session_id="session-001",
            result=result,
        )

    def test_small_payload_passes_through_at_requested_level(self) -> None:
        """A payload within budget is delivered at the requested disclosure level."""
        raw = self._make_raw_output({"key": "value"})
        filtered, metrics = apply_progressive_disclosure(
            raw_output=raw,
            budget_remaining=8192,
            requested_level=DisclosureLevel.STANDARD,
        )
        assert filtered.disclosure_level_applied == DisclosureLevel.STANDARD
        assert filtered.filtered_result is not None
        assert metrics.raw_tokens > 0

    def test_oversized_payload_downgraded_to_minimal(self) -> None:
        """A payload exceeding the budget should be downgraded toward MINIMAL."""
        # Create a large payload
        large_result = {"data": ["item " * 100 for _ in range(200)]}
        raw = self._make_raw_output(large_result)

        # Very tight budget — forces minimal disclosure
        filtered, metrics = apply_progressive_disclosure(
            raw_output=raw,
            budget_remaining=50,  # Only 50 tokens available
            requested_level=DisclosureLevel.FULL,
        )
        # Compression ratio must be < 1 (compression happened)
        assert filtered.compression_ratio < 1.0
        assert filtered.disclosure_level_applied.value <= DisclosureLevel.STANDARD.value

    def test_compression_ratio_computed_correctly(self) -> None:
        """Compression ratio should equal filtered_tokens / raw_tokens."""
        raw = self._make_raw_output({"a": "x" * 400})
        filtered, metrics = apply_progressive_disclosure(
            raw_output=raw,
            budget_remaining=8192,
        )
        expected_ratio = metrics.filtered_tokens / max(metrics.raw_tokens, 1)
        assert abs(filtered.compression_ratio - expected_ratio) < 0.01

    def test_budget_remaining_decremented(self) -> None:
        """Budget remaining should be budget_before minus filtered tokens."""
        budget = 1000
        raw = self._make_raw_output({"x": "hello"})
        filtered, metrics = apply_progressive_disclosure(
            raw_output=raw,
            budget_remaining=budget,
        )
        assert filtered.budget_remaining == max(budget - filtered.filtered_token_estimate, 0)

    def test_error_field_preserved(self) -> None:
        """Error string from raw output should be forwarded to filtered output."""
        raw = ToolCallOutput(
            call_id="err-call",
            tool_name="broken_tool",
            agent_id="a1",
            session_id="s1",
            result=None,
            error="Something went wrong",
        )
        filtered, _ = apply_progressive_disclosure(
            raw_output=raw,
            budget_remaining=8192,
        )
        assert filtered.error == "Something went wrong"

    def test_full_disclosure_returns_original(self) -> None:
        """FULL disclosure with sufficient budget returns the original result verbatim."""
        original = {"nested": {"list": [1, 2, 3], "text": "hello"}}
        raw = self._make_raw_output(original)
        filtered, _ = apply_progressive_disclosure(
            raw_output=raw,
            budget_remaining=8192,
            requested_level=DisclosureLevel.FULL,
        )
        assert filtered.disclosure_level_applied == DisclosureLevel.FULL
        assert filtered.filtered_result == original

    def test_metrics_record_fields_consistent_with_filtered_output(self) -> None:
        """ContextMetricsRecord fields must mirror FilteredToolCallOutput fields."""
        raw = self._make_raw_output({"data": "test"})
        filtered, metrics = apply_progressive_disclosure(
            raw_output=raw,
            budget_remaining=8192,
        )
        assert metrics.call_id == raw.call_id
        assert metrics.session_id == raw.session_id
        assert metrics.tool_name == raw.tool_name
        assert metrics.raw_tokens == filtered.raw_token_estimate
        assert metrics.filtered_tokens == filtered.filtered_token_estimate

    def test_empty_result_handled(self) -> None:
        """Empty dict result should not cause exceptions."""
        raw = self._make_raw_output({})
        filtered, _ = apply_progressive_disclosure(
            raw_output=raw,
            budget_remaining=100,
        )
        assert filtered is not None

    def test_non_dict_list_result(self) -> None:
        """List result (non-dict) should be processed without errors."""
        raw = self._make_raw_output([{"id": i} for i in range(30)])
        filtered, _ = apply_progressive_disclosure(
            raw_output=raw,
            budget_remaining=100,
        )
        assert filtered is not None
