"""
SQLite persistence for ARIA research sessions and outcomes.

Three tables:
  sessions  — one row per completed research run: query, memo sections, model metadata
  sources   — web sources gathered during the run, each tagged with search purpose
  outcomes  — user-recorded verdict on whether a thesis proved correct over time

The outcome table is what makes this a system rather than a demo: it creates
a feedback loop that lets you track prediction quality across sessions.
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Uppercase words that look like tickers but aren't.
_TICKER_STOP = {
    "I", "A", "AN", "THE", "IS", "IN", "ON", "AT", "OR", "AND", "FOR",
    "TO", "OF", "US", "EU", "UK", "IF", "NO", "DO", "BE", "AS", "BY",
    "IT", "ITS", "ALL", "ANY", "NEW", "NOW", "HOW", "WHO", "WHY", "ETF",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id                   TEXT PRIMARY KEY,
    created_at           TEXT NOT NULL,
    query                TEXT NOT NULL,
    ticker               TEXT,
    thesis               TEXT,
    counter_evidence TEXT,
    failure_conditions   TEXT,
    confidence           TEXT,
    baselines            TEXT,
    model_name           TEXT,
    steps_taken          INTEGER,
    fallback_used        INTEGER NOT NULL DEFAULT 0,
    mode                 TEXT,
    thesis_status        TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS sources (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    url          TEXT,
    title        TEXT,
    purpose      TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS outcomes (
    session_id   TEXT PRIMARY KEY,
    result       TEXT NOT NULL,
    note         TEXT,
    evaluated_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS monitor_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    checked_at   TEXT NOT NULL,
    status       TEXT NOT NULL,
    summary      TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
"""

# Migration: add columns/tables that were introduced after initial schema creation.
_MIGRATIONS = [
    "ALTER TABLE sessions ADD COLUMN thesis_status TEXT NOT NULL DEFAULT 'active'",
    "ALTER TABLE sessions RENAME COLUMN adversarial_challenge TO counter_evidence",
]


class ResearchDatabase:
    """
    Thin wrapper around a SQLite database for ARIA session persistence.

    Designed to be instantiated per-operation — sqlite3 connections are
    cheap and this avoids threading concerns.
    """

    def __init__(self, path: Path = Path("aria_research.db")) -> None:
        self._path = Path(path)
        self._init_schema()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def save_session(
        self,
        session_id: str,
        query: str,
        memo_state: Dict[str, str],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Persist a completed research session.
        Idempotent: re-saving the same session_id replaces the previous row.
        """
        ticker = _detect_ticker(query)
        created_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions (
                    id, created_at, query, ticker,
                    thesis, counter_evidence, failure_conditions,
                    confidence, baselines,
                    model_name, steps_taken, fallback_used, mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    created_at,
                    query,
                    ticker,
                    memo_state.get("thesis"),
                    memo_state.get("counter_evidence"),
                    memo_state.get("failure_conditions"),
                    memo_state.get("confidence"),
                    memo_state.get("baselines"),
                    metadata.get("model_name"),
                    metadata.get("steps_taken"),
                    int(bool(metadata.get("fallback_used", False))),
                    metadata.get("mode"),
                ),
            )

            # Delete any stale sources for this session before re-inserting.
            conn.execute("DELETE FROM sources WHERE session_id = ?", (session_id,))
            for src in metadata.get("web_search_sources", []):
                conn.execute(
                    "INSERT INTO sources (session_id, url, title, purpose) VALUES (?, ?, ?, ?)",
                    (session_id, src.get("url"), src.get("title"), src.get("purpose")),
                )

    def save_outcome(
        self,
        session_id: str,
        result: str,
        note: str = "",
    ) -> bool:
        """
        Record whether a thesis held up.
        Returns False if session_id is not found.
        """
        evaluated_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                """
                INSERT OR REPLACE INTO outcomes (session_id, result, note, evaluated_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, result, note, evaluated_at),
            )
        return True

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return a single session row with its outcome (if any), or None."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT s.*, o.result AS outcome_result, o.note AS outcome_note,
                       o.evaluated_at AS outcome_evaluated_at
                FROM sessions s
                LEFT JOIN outcomes o ON s.id = o.session_id
                WHERE s.id = ?
                """,
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_sessions(
        self,
        ticker: Optional[str] = None,
        unresolved_only: bool = False,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Return recent sessions, newest first.

        ticker        — filter to rows where the detected ticker matches
        unresolved_only — only sessions that have no outcome recorded yet
        """
        sql = """
            SELECT s.*, o.result AS outcome_result
            FROM sessions s
            LEFT JOIN outcomes o ON s.id = o.session_id
        """
        conditions: List[str] = []
        params: List[Any] = []

        if ticker:
            conditions.append("s.ticker = ?")
            params.append(ticker.upper())
        if unresolved_only:
            conditions.append("o.session_id IS NULL")

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        sql += " ORDER BY s.created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Monitor
    # ------------------------------------------------------------------

    def update_thesis_status(self, session_id: str, status: str) -> None:
        """Update thesis_status for a session. status: 'active' | 'challenged' | 'resolved'"""
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET thesis_status = ? WHERE id = ?",
                (status, session_id),
            )

    def get_active_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return sessions with no recorded outcome and thesis_status = 'active'."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT s.*
                FROM sessions s
                LEFT JOIN outcomes o ON s.id = o.session_id
                WHERE o.session_id IS NULL
                  AND s.thesis_status = 'active'
                  AND s.failure_conditions IS NOT NULL
                ORDER BY s.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def save_monitor_run(
        self,
        session_id: str,
        status: str,
        summary: str,
        checked_at: str,
    ) -> None:
        """Record the result of a monitoring check."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO monitor_runs (session_id, checked_at, status, summary)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, checked_at, status, summary),
            )

    def get_monitor_history(
        self,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Return recent monitor run records, newest first."""
        sql = "SELECT * FROM monitor_runs"
        params: List[Any] = []
        if session_id:
            sql += " WHERE session_id = ?"
            params.append(session_id)
        sql += " ORDER BY checked_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_sources_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Return all web sources linked to a session."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM sources WHERE session_id = ?", (session_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all related data (sources, outcomes, monitor_runs).
        Returns False if the session does not exist.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return False
            conn.execute("DELETE FROM sources WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM outcomes WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM monitor_runs WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return True

    def get_session_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics for the database."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            with_outcomes = conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
            outcome_breakdown: Dict[str, int] = {}
            for result in ("correct", "incorrect", "partial"):
                outcome_breakdown[result] = conn.execute(
                    "SELECT COUNT(*) FROM outcomes WHERE result = ?", (result,)
                ).fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE thesis_status = 'active'"
            ).fetchone()[0]
            challenged = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE thesis_status = 'challenged'"
            ).fetchone()[0]
            monitor_runs = conn.execute("SELECT COUNT(*) FROM monitor_runs").fetchone()[0]
            oldest = conn.execute("SELECT MIN(created_at) FROM sessions").fetchone()[0]
            newest = conn.execute("SELECT MAX(created_at) FROM sessions").fetchone()[0]
        return {
            "total_sessions": total,
            "sessions_with_outcomes": with_outcomes,
            "outcome_breakdown": outcome_breakdown,
            "active_theses": active,
            "challenged_theses": challenged,
            "total_monitor_runs": monitor_runs,
            "oldest_session": oldest,
            "newest_session": newest,
        }

    def export_sessions(self) -> List[Dict[str, Any]]:
        """Return all sessions joined with outcomes, newest first."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT s.*, o.result AS outcome_result, o.note AS outcome_note,
                       o.evaluated_at AS outcome_evaluated_at
                FROM sessions s
                LEFT JOIN outcomes o ON s.id = o.session_id
                ORDER BY s.created_at DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def resolve_session_id(self, prefix: str) -> Optional[str]:
        """
        Resolve a full or partial (≥8-char) session ID to a full ID.

        Returns the full ID on an unambiguous match, None if no match,
        or raises ValueError listing all candidates if the prefix is ambiguous.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM sessions WHERE id LIKE ?", (f"{prefix}%",)
            ).fetchall()
        ids = [r[0] for r in rows]
        if not ids:
            return None
        if len(ids) == 1:
            return ids[0]
        raise ValueError(
            f"Prefix {prefix!r} matches {len(ids)} sessions:\n"
            + "\n".join(f"  {sid}" for sid in ids)
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._path))

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        self._run_migrations()

    def _run_migrations(self) -> None:
        with self._connect() as conn:
            for sql in _MIGRATIONS:
                try:
                    conn.execute(sql)
                except sqlite3.OperationalError:
                    pass  # Column/table already exists


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _detect_ticker(text: str) -> Optional[str]:
    """
    Return the first plausible ticker symbol in text, or None.
    Looks for 2–5 uppercase letters that aren't common English words.
    """
    candidates = [
        t for t in re.findall(r"\b[A-Z]{2,5}\b", text)
        if t not in _TICKER_STOP
    ]
    return candidates[0] if candidates else None
