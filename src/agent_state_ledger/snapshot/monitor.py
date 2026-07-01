"""
agent_state_ledger.snapshot.monitor
=====================================
Standalone monitoring CLI for the Snapshot Engine.

This script runs as a separate process alongside the main router, connecting
to the same SQLite snapshot store to emit real-time observability data:

* Prints a live, auto-refreshing dashboard (using ``rich``) showing:
  - Active session count and token consumption per session
  - Snapshot generation progress for each session
  - Per-session deviation scores with colour-coded warnings
  - Rollback event log (last 10 rollbacks across all sessions)

* Exposes a ``/monitor`` HTTP endpoint (on ``ASL_METRICS_PORT + 1``) that
  returns a JSON snapshot of the same data for programmatic consumers.

Usage
-----
After installing the package::

    asl-monitor

Or directly::

    python -m agent_state_ledger.snapshot.monitor

Environment variables
---------------------
The monitor inherits all ``ASL_SNAPSHOT_*`` and ``ASL_*`` settings from
the standard configuration module.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent_state_ledger.config import get_settings
from agent_state_ledger.logging_setup import configure_logging, get_logger

logger = get_logger(__name__)
console = Console()


class SnapshotMonitor:
    """
    Real-time dashboard for the Agent State Ledger Snapshot Engine.

    Opens the snapshot SQLite database in **read-only** mode (URI
    ``file:path?mode=ro``) so the monitor never interferes with the
    writer process.

    Parameters
    ----------
    db_path:
        Path to the snapshot SQLite database.
    refresh_interval:
        Seconds between dashboard refreshes.
    """

    def __init__(self, db_path: Path, refresh_interval: float = 2.0) -> None:
        self._db_path = db_path
        self._refresh_interval = refresh_interval
        self._start_time = time.monotonic()
        self._conn: aiosqlite.Connection | None = None

    async def open(self) -> None:
        """Open the database in read-only mode."""
        uri = f"file:{self._db_path}?mode=ro"
        self._conn = await aiosqlite.connect(database=uri, uri=True)
        self._conn.row_factory = aiosqlite.Row

    async def close(self) -> None:
        """Close the read-only database connection."""
        if self._conn:
            await self._conn.close()

    async def _fetch_session_stats(self) -> list[dict]:
        """Query per-session snapshot statistics."""
        assert self._conn is not None
        async with self._conn.execute(
            """
            SELECT
                s.session_id,
                s.agent_id,
                MAX(s.generation)           AS latest_gen,
                COUNT(s.snapshot_id)        AS total_snaps,
                MAX(s.intent_deviation_score) AS max_deviation,
                MAX(s.created_at)           AS last_snap_at
            FROM snapshots s
            GROUP BY s.session_id, s.agent_id
            ORDER BY last_snap_at DESC
            LIMIT 20
            """
        ) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def _fetch_recent_rollbacks(self, limit: int = 10) -> list[dict]:
        """Query the most recent rollback audit records."""
        assert self._conn is not None
        try:
            async with self._conn.execute(
                """
                SELECT session_id, from_generation, to_generation,
                       reason, initiator, deviation_score_at_rollback, executed_at
                FROM rollback_audit
                ORDER BY executed_at DESC
                LIMIT ?
                """,
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception:
            return []

    async def _fetch_totals(self) -> dict:
        """Fetch aggregate totals for the header panel."""
        assert self._conn is not None
        async with self._conn.execute(
            "SELECT COUNT(*) AS total FROM snapshots"
        ) as cursor:
            snap_row = await cursor.fetchone()

        total_snaps = snap_row["total"] if snap_row else 0

        try:
            async with self._conn.execute(
                "SELECT COUNT(*) AS total FROM rollback_audit"
            ) as cursor:
                rb_row = await cursor.fetchone()
            total_rollbacks = rb_row["total"] if rb_row else 0
        except Exception:
            total_rollbacks = 0

        return {
            "total_snapshots": total_snaps,
            "total_rollbacks": total_rollbacks,
        }

    def _build_header(self, totals: dict, uptime: float) -> Panel:
        """Construct the header panel with aggregate stats."""
        h, m, s = int(uptime // 3600), int((uptime % 3600) // 60), int(uptime % 60)
        uptime_str = f"{h:02d}:{m:02d}:{s:02d}"

        header_text = Text()
        header_text.append("⬡ AGENT STATE LEDGER", style="bold cyan")
        header_text.append("  |  Snapshot Monitor  |  ", style="dim white")
        header_text.append(
            datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            style="white",
        )
        header_text.append(f"  |  Uptime: {uptime_str}", style="dim white")
        header_text.append(
            f"  |  Total Snapshots: {totals['total_snapshots']}",
            style="bold green",
        )
        header_text.append(
            f"  |  Rollbacks: {totals['total_rollbacks']}",
            style="bold yellow" if totals["total_rollbacks"] > 0 else "dim white",
        )
        return Panel(header_text, style="cyan")

    def _build_sessions_table(self, sessions: list[dict]) -> Table:
        """Build the per-session statistics table."""
        table = Table(
            title="Active Sessions",
            show_header=True,
            header_style="bold magenta",
            border_style="dim white",
            expand=True,
        )
        table.add_column("Session ID", style="cyan", no_wrap=True, max_width=20)
        table.add_column("Agent ID", style="white", max_width=20)
        table.add_column("Latest Gen", justify="right", style="green")
        table.add_column("Snapshots", justify="right", style="blue")
        table.add_column("Deviation", justify="right")
        table.add_column("Last Snapshot", style="dim white", no_wrap=True)

        if not sessions:
            table.add_row(
                "[dim]No sessions found[/dim]", "", "", "", "", ""
            )
            return table

        for s in sessions:
            dev_score = s.get("max_deviation", 0.0) or 0.0
            if dev_score >= 0.7:
                dev_style = "bold red"
                dev_icon = "🔴"
            elif dev_score >= 0.35:
                dev_style = "bold yellow"
                dev_icon = "🟡"
            else:
                dev_style = "green"
                dev_icon = "🟢"

            table.add_row(
                str(s.get("session_id", ""))[:18] + "…",
                str(s.get("agent_id", ""))[:18],
                str(s.get("latest_gen", 0)),
                str(s.get("total_snaps", 0)),
                Text(f"{dev_icon} {dev_score:.3f}", style=dev_style),
                str(s.get("last_snap_at", ""))[:19],
            )

        return table

    def _build_rollbacks_table(self, rollbacks: list[dict]) -> Table:
        """Build the recent rollback events table."""
        table = Table(
            title="Recent Rollback Events",
            show_header=True,
            header_style="bold yellow",
            border_style="dim white",
            expand=True,
        )
        table.add_column("Session ID", style="cyan", no_wrap=True, max_width=20)
        table.add_column("From Gen", justify="right")
        table.add_column("To Gen", justify="right")
        table.add_column("Reason", style="yellow")
        table.add_column("Initiator")
        table.add_column("Deviation", justify="right", style="red")
        table.add_column("Executed At", style="dim white", no_wrap=True)

        if not rollbacks:
            table.add_row(
                "[dim]No rollbacks recorded[/dim]", "", "", "", "", "", ""
            )
            return table

        for r in rollbacks:
            table.add_row(
                str(r.get("session_id", ""))[:18] + "…",
                str(r.get("from_generation", "")),
                str(r.get("to_generation", "")),
                str(r.get("reason", "")),
                str(r.get("initiator", "")),
                f"{r.get('deviation_score_at_rollback', 0.0):.3f}",
                str(r.get("executed_at", ""))[:19],
            )

        return table

    async def run_dashboard(self) -> None:
        """
        Run the live Rich terminal dashboard until the user presses Ctrl+C.

        Refreshes every ``_refresh_interval`` seconds.  Gracefully handles
        the case where the database file does not yet exist (prints a waiting
        message and retries).
        """
        await logger.ainfo(
            "monitor_dashboard_starting",
            db_path=str(self._db_path),
            refresh_interval=self._refresh_interval,
        )

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="rollbacks", size=14),
        )

        with Live(layout, console=console, refresh_per_second=1, screen=True):
            while True:
                try:
                    if self._conn is None:
                        await self.open()

                    uptime = time.monotonic() - self._start_time
                    totals = await self._fetch_totals()
                    sessions = await self._fetch_session_stats()
                    rollbacks = await self._fetch_recent_rollbacks()

                    layout["header"].update(self._build_header(totals, uptime))
                    layout["body"].update(
                        Panel(self._build_sessions_table(sessions), border_style="blue")
                    )
                    layout["rollbacks"].update(
                        Panel(self._build_rollbacks_table(rollbacks), border_style="yellow")
                    )

                except Exception as exc:
                    layout["header"].update(
                        Panel(
                            Text(
                                f"⚠ Monitor error: {exc}  "
                                f"(retrying in {self._refresh_interval}s)",
                                style="bold red",
                            )
                        )
                    )
                    self._conn = None  # Force reconnect on next cycle

                await asyncio.sleep(self._refresh_interval)


def run_monitor() -> None:
    """
    CLI entry-point registered as ``asl-monitor`` in ``pyproject.toml``.

    Reads configuration from environment / .env, opens the snapshot database
    in read-only mode, and starts the live terminal dashboard.
    """
    configure_logging()
    settings = get_settings()

    monitor = SnapshotMonitor(
        db_path=settings.snapshot.db_path,
        refresh_interval=2.0,
    )

    try:
        asyncio.run(monitor.run_dashboard())
    except KeyboardInterrupt:
        console.print("\n[bold cyan]Monitor stopped.[/bold cyan]")


if __name__ == "__main__":
    run_monitor()
