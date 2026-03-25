"""
Tests for the monitoring subsystem.

Covers:
  - _parse_monitor_response: LLM output parsing (status + reason extraction)
  - ResearchDatabase: CRUD and thesis status tracking using an in-memory SQLite file
"""
from __future__ import annotations

import pytest
from pathlib import Path

from aria.monitor.checker import _parse_monitor_response
from aria.storage.db import ResearchDatabase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path) -> ResearchDatabase:
    """Fresh in-memory-equivalent database for each test."""
    return ResearchDatabase(path=tmp_path / "test.db")


def _save(db: ResearchDatabase, session_id: str, **overrides: object) -> None:
    """Helper: save a session with sensible defaults, allowing field overrides."""
    defaults: dict = {
        "query": "Will gold outperform equities in 2026?",
        "memo_state": {
            "thesis": "Gold outperforms.",
            "failure_conditions": "Central bank purchases fall below 300t/year.",
        },
        "metadata": {
            "model_name": "test-model",
            "steps_taken": 5,
            "fallback_used": False,
            "mode": "autonomous",
        },
    }
    defaults.update(overrides)
    db.save_session(session_id=session_id, **defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _parse_monitor_response
# ---------------------------------------------------------------------------

class TestParseMonitorResponse:
    def test_ok_status_extracted(self) -> None:
        text = "STATUS: ok\nREASON: The thesis remains intact. Conditions have not triggered."
        status, reason = _parse_monitor_response(text)
        assert status == "ok"
        assert "intact" in reason

    def test_challenged_status_extracted(self) -> None:
        text = "STATUS: challenged\nREASON: Condition 2 has triggered — revenue declined below threshold."
        status, reason = _parse_monitor_response(text)
        assert status == "challenged"
        assert "triggered" in reason

    def test_case_insensitive_status_parsing(self) -> None:
        text = "STATUS: OK\nREASON: All clear."
        status, _ = _parse_monitor_response(text)
        assert status == "ok"

        text2 = "STATUS: CHALLENGED\nREASON: Condition fired."
        status2, _ = _parse_monitor_response(text2)
        assert status2 == "challenged"

    def test_missing_status_defaults_to_ok(self) -> None:
        text = "The thesis appears intact. Central bank buying continues at expected levels."
        status, _ = _parse_monitor_response(text)
        assert status == "ok"

    def test_reason_spans_multiple_lines(self) -> None:
        text = (
            "STATUS: challenged\n"
            "REASON: Condition 1 has triggered.\n"
            "Revenue growth fell below 40% YoY in Q2 2026.\n"
            "This directly violates the stated failure condition."
        )
        status, reason = _parse_monitor_response(text)
        assert status == "challenged"
        assert "40%" in reason
        assert "Q2 2026" in reason

    def test_reason_stripped_of_leading_whitespace(self) -> None:
        text = "STATUS: ok\nREASON:   Thesis intact."
        _, reason = _parse_monitor_response(text)
        assert not reason.startswith(" ")

    def test_malformed_response_returns_full_text_as_reason(self) -> None:
        text = "Something went wrong with the LLM output."
        _, reason = _parse_monitor_response(text)
        assert reason == text.strip()


# ---------------------------------------------------------------------------
# ResearchDatabase
# ---------------------------------------------------------------------------

class TestResearchDatabase:
    def test_save_and_retrieve_session(self, db: ResearchDatabase) -> None:
        _save(db, "s-001")
        session = db.get_session("s-001")
        assert session is not None
        assert session["id"] == "s-001"
        assert session["query"] == "Will gold outperform equities in 2026?"
        assert session["thesis"] == "Gold outperforms."
        assert session["model_name"] == "test-model"

    def test_get_nonexistent_session_returns_none(self, db: ResearchDatabase) -> None:
        assert db.get_session("does-not-exist") is None

    def test_save_session_idempotent(self, db: ResearchDatabase) -> None:
        """Re-saving with same session_id replaces the row."""
        _save(db, "s-002", memo_state={"thesis": "Original."})
        _save(db, "s-002", memo_state={"thesis": "Replaced."})
        session = db.get_session("s-002")
        assert session is not None
        assert session["thesis"] == "Replaced."

    def test_save_and_retrieve_outcome(self, db: ResearchDatabase) -> None:
        _save(db, "s-003")
        ok = db.save_outcome("s-003", "correct", note="Thesis held precisely.")
        assert ok is True
        session = db.get_session("s-003")
        assert session is not None
        assert session["outcome_result"] == "correct"
        assert session["outcome_note"] == "Thesis held precisely."

    def test_save_outcome_for_missing_session_returns_false(self, db: ResearchDatabase) -> None:
        assert db.save_outcome("nonexistent", "correct") is False

    def test_list_sessions_returns_all(self, db: ResearchDatabase) -> None:
        for i in range(3):
            _save(db, f"s-{i}")
        sessions = db.list_sessions(limit=10)
        assert len(sessions) == 3
        ids = {s["id"] for s in sessions}
        assert ids == {"s-0", "s-1", "s-2"}

    def test_list_sessions_limit_respected(self, db: ResearchDatabase) -> None:
        for i in range(5):
            _save(db, f"s-{i}")
        sessions = db.list_sessions(limit=3)
        assert len(sessions) == 3

    def test_list_sessions_unresolved_only(self, db: ResearchDatabase) -> None:
        _save(db, "resolved")
        _save(db, "open")
        db.save_outcome("resolved", "correct")
        sessions = db.list_sessions(unresolved_only=True)
        ids = {s["id"] for s in sessions}
        assert "open" in ids
        assert "resolved" not in ids

    def test_update_thesis_status(self, db: ResearchDatabase) -> None:
        _save(db, "s-status")
        db.update_thesis_status("s-status", "challenged")
        session = db.get_session("s-status")
        assert session is not None
        assert session["thesis_status"] == "challenged"

    def test_get_active_sessions_requires_failure_conditions(self, db: ResearchDatabase) -> None:
        """Sessions without failure_conditions should not appear in active list."""
        _save(db, "no-fc", memo_state={"thesis": "No failure conditions here."})
        active = db.get_active_sessions()
        ids = {s["id"] for s in active}
        assert "no-fc" not in ids

    def test_get_active_sessions_returns_eligible(self, db: ResearchDatabase) -> None:
        """Session with failure_conditions, no outcome, thesis_status='active' is returned."""
        _save(db, "eligible")  # includes failure_conditions in default memo_state
        active = db.get_active_sessions()
        ids = {s["id"] for s in active}
        assert "eligible" in ids

    def test_get_active_sessions_excludes_challenged(self, db: ResearchDatabase) -> None:
        _save(db, "challenged")
        db.update_thesis_status("challenged", "challenged")
        active = db.get_active_sessions()
        ids = {s["id"] for s in active}
        assert "challenged" not in ids

    def test_save_monitor_run_and_retrieve_history(self, db: ResearchDatabase) -> None:
        _save(db, "s-monitor")
        db.save_monitor_run(
            session_id="s-monitor",
            status="ok",
            summary="Thesis intact, no conditions triggered.",
            checked_at="2026-03-20T12:00:00+00:00",
        )
        history = db.get_monitor_history(session_id="s-monitor")
        assert len(history) == 1
        assert history[0]["status"] == "ok"
        assert history[0]["session_id"] == "s-monitor"
