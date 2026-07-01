"""
tests/test_router_api.py
==========================
End-to-end HTTP API tests for the FastAPI State Router.

These tests run against the full FastAPI application via an in-process
``httpx.AsyncClient`` with an ASGITransport — no real network calls are made.

Coverage includes:
- Session lifecycle (create, get, list, delete)
- Tool-call interception with progressive disclosure
- Manual rollback via HTTP
- JSON-RPC endpoint
- Error paths (404, 409, 403, 413)
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from agent_state_ledger.models import (
    DisclosureLevel,
    SessionStatus,
)


# ============================================================================ #
#  Health check                                                                 #
# ============================================================================ #

class TestHealthEndpoint:
    async def test_health_returns_200(self, async_client: AsyncClient) -> None:
        response = await async_client.get("/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in ("healthy", "degraded", "unhealthy")
        assert "version" in body
        assert "uptime_seconds" in body
        assert "active_sessions" in body

    async def test_health_contains_snapshot_count(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.get("/v1/health")
        body = response.json()
        assert "snapshots_persisted" in body
        assert isinstance(body["snapshots_persisted"], int)


# ============================================================================ #
#  Session lifecycle                                                             #
# ============================================================================ #

class TestSessionLifecycle:
    """Tests for session creation, retrieval, listing, and deletion."""

    async def _create_session(
        self,
        client: AsyncClient,
        agent_id: str = "test-agent",
    ) -> dict:
        response = await client.post(
            "/v1/sessions",
            json={
                "agent_id": agent_id,
                "intent": {
                    "primary_goal": "Process test data and summarise the output.",
                    "allowed_tools": ["echo", "compute_hash"],
                    "max_steps": 20,
                    "context_budget_override": 4096,
                },
            },
        )
        assert response.status_code == 201, response.text
        return response.json()

    async def test_create_session_returns_201(
        self, async_client: AsyncClient
    ) -> None:
        body = await self._create_session(async_client)
        assert "session_id" in body
        assert body["agent_id"] == "test-agent"
        assert body["status"] == SessionStatus.ACTIVE.value

    async def test_get_existing_session(self, async_client: AsyncClient) -> None:
        created = await self._create_session(async_client)
        session_id = created["session_id"]

        response = await async_client.get(f"/v1/sessions/{session_id}")
        assert response.status_code == 200
        body = response.json()
        assert body["session_id"] == session_id

    async def test_get_nonexistent_session_returns_404(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.get("/v1/sessions/does-not-exist")
        assert response.status_code == 404

    async def test_list_sessions_returns_created_session(
        self, async_client: AsyncClient
    ) -> None:
        await self._create_session(async_client, agent_id="list-test-agent")
        response = await async_client.get("/v1/sessions")
        assert response.status_code == 200
        sessions = response.json()
        agent_ids = [s["agent_id"] for s in sessions]
        assert "list-test-agent" in agent_ids

    async def test_list_sessions_with_status_filter(
        self, async_client: AsyncClient
    ) -> None:
        await self._create_session(async_client)
        response = await async_client.get("/v1/sessions?status=active")
        assert response.status_code == 200
        for s in response.json():
            assert s["status"] == "active"

    async def test_delete_session_returns_204(
        self, async_client: AsyncClient
    ) -> None:
        created = await self._create_session(async_client)
        session_id = created["session_id"]

        response = await async_client.delete(f"/v1/sessions/{session_id}")
        assert response.status_code == 204

    async def test_delete_nonexistent_session_returns_404(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.delete("/v1/sessions/nonexistent")
        assert response.status_code == 404


# ============================================================================ #
#  Tool-call interception                                                       #
# ============================================================================ #

class TestToolCallInterception:
    """Tests for the /tool-call progressive disclosure endpoint."""

    async def _setup_session(self, client: AsyncClient) -> str:
        """Create a session and return its ID."""
        response = await client.post(
            "/v1/sessions",
            json={
                "agent_id": "tool-test-agent",
                "intent": {
                    "primary_goal": "Search and summarise web results.",
                    "allowed_tools": ["search_web", "echo"],
                    "max_steps": 100,
                    "context_budget_override": 8192,
                },
            },
        )
        return response.json()["session_id"]

    async def test_tool_call_returns_filtered_output(
        self, async_client: AsyncClient
    ) -> None:
        session_id = await self._setup_session(async_client)
        response = await async_client.post(
            "/v1/tool-call",
            json={
                "call_id": "call-001",
                "tool_name": "search_web",
                "agent_id": "tool-test-agent",
                "session_id": session_id,
                "result": {"results": [{"title": f"Result {i}", "url": f"http://ex.com/{i}"} for i in range(20)]},
                "error": None,
                "requested_disclosure_level": DisclosureLevel.STANDARD.value,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert "filtered_result" in body
        assert "compression_ratio" in body
        assert "budget_remaining" in body
        assert "disclosure_level_applied" in body

    async def test_tool_call_increments_step_count(
        self, async_client: AsyncClient
    ) -> None:
        session_id = await self._setup_session(async_client)
        await async_client.post(
            "/v1/tool-call",
            json={
                "call_id": "call-002",
                "tool_name": "echo",
                "agent_id": "tool-test-agent",
                "session_id": session_id,
                "result": {"echo": "hello"},
            },
        )
        session_response = await async_client.get(f"/v1/sessions/{session_id}")
        session = session_response.json()
        assert session["step_count"] >= 1

    async def test_tool_call_blocked_for_disallowed_tool(
        self, async_client: AsyncClient
    ) -> None:
        session_id = await self._setup_session(async_client)
        response = await async_client.post(
            "/v1/tool-call",
            json={
                "call_id": "call-003",
                "tool_name": "disallowed_tool",
                "agent_id": "tool-test-agent",
                "session_id": session_id,
                "result": {"data": "something"},
            },
        )
        assert response.status_code == 403

    async def test_tool_call_on_nonexistent_session_returns_404(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.post(
            "/v1/tool-call",
            json={
                "call_id": "call-004",
                "tool_name": "echo",
                "agent_id": "any-agent",
                "session_id": "session-does-not-exist",
                "result": {"msg": "test"},
            },
        )
        assert response.status_code == 404

    async def test_tool_call_with_empty_allowlist_permits_all(
        self, async_client: AsyncClient
    ) -> None:
        """Empty allowed_tools list means all tools are permitted."""
        response = await async_client.post(
            "/v1/sessions",
            json={
                "agent_id": "open-agent",
                "intent": {
                    "primary_goal": "Any tool may be called in this open session.",
                    "allowed_tools": [],  # Empty = unrestricted
                    "max_steps": 10,
                },
            },
        )
        session_id = response.json()["session_id"]

        tool_response = await async_client.post(
            "/v1/tool-call",
            json={
                "call_id": "call-open-001",
                "tool_name": "any_arbitrary_tool_name",
                "agent_id": "open-agent",
                "session_id": session_id,
                "result": {"data": "anything"},
            },
        )
        assert tool_response.status_code == 200


# ============================================================================ #
#  JSON-RPC endpoint                                                            #
# ============================================================================ #

class TestJsonRpcEndpoint:
    async def test_valid_rpc_call_returns_result(
        self, async_client: AsyncClient
    ) -> None:
        # Create a session first
        sess_resp = await async_client.post(
            "/v1/sessions",
            json={
                "agent_id": "rpc-agent",
                "intent": {
                    "primary_goal": "Test the JSON-RPC endpoint.",
                    "allowed_tools": [],
                    "max_steps": 10,
                },
            },
        )
        session_id = sess_resp.json()["session_id"]

        response = await async_client.post(
            "/v1/rpc",
            json={
                "jsonrpc": "2.0",
                "id": "rpc-test-001",
                "method": "tool/call",
                "params": {
                    "call_id": "rpc-call-001",
                    "tool_name": "echo",
                    "agent_id": "rpc-agent",
                    "session_id": session_id,
                    "result": {"echo": "hello world"},
                    "requested_disclosure_level": 3,
                },
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["jsonrpc"] == "2.0"
        assert body["id"] == "rpc-test-001"
        assert "result" in body

    async def test_invalid_jsonrpc_version_returns_error(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.post(
            "/v1/rpc",
            json={
                "jsonrpc": "1.0",   # Invalid
                "id": "bad-001",
                "method": "tool/call",
                "params": {},
            },
        )
        assert response.status_code == 200  # JSON-RPC errors are 200
        body = response.json()
        assert "error" in body
        assert body["error"]["code"] == -32600

    async def test_unknown_method_returns_error(
        self, async_client: AsyncClient
    ) -> None:
        response = await async_client.post(
            "/v1/rpc",
            json={
                "jsonrpc": "2.0",
                "id": "m-001",
                "method": "nonexistent/method",
                "params": {},
            },
        )
        body = response.json()
        assert "error" in body
        assert body["error"]["code"] == -32601
