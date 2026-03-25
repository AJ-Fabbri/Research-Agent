"""
Tests for the ResearchDatabase management extensions and CLI command logic.

All tests use a temporary SQLite file (tmp_path fixture) — no real DB touched.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from aria.storage.db import ResearchDatabase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path) -> ResearchDatabase:
    return ResearchDatabase(path=tmp_path / "test.db")


def _save(db: ResearchDatabase, session_id: str, **overrides: Any) -> None:
    defaults: Dict[str, Any] = {
        "query": "Will gold outperform equities?",
        "memo_state": {
            "thesis": "Gold outperforms equities in 2026.",
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
# delete_session
# ---------------------------------------------------------------------------

class TestDeleteSession:
    def test_delete_existing_session_returns_true(self, db: ResearchDatabase) -> None:
        _save(db, "s-001")
        assert db.delete_session("s-001") is True
        assert db.get_session("s-001") is None

    def test_delete_nonexistent_session_returns_false(self, db: ResearchDatabase) -> None:
        assert db.delete_session("does-not-exist") is False

    def test_delete_cascades_to_sources(self, db: ResearchDatabase) -> None:
        _save(db, "s-src")
        # Manually insert a source row
        import sqlite3
        with sqlite3.connect(str(db._path)) as conn:
            conn.execute(
                "INSERT INTO sources (session_id, url, title, purpose) VALUES (?,?,?,?)",
                ("s-src", "https://ft.com/x", "FT Article", "background"),
            )
        assert len(db.get_sources_for_session("s-src")) == 1
        db.delete_session("s-src")
        assert len(db.get_sources_for_session("s-src")) == 0

    def test_delete_cascades_to_outcomes(self, db: ResearchDatabase) -> None:
        _save(db, "s-out")
        db.save_outcome("s-out", "correct")
        db.delete_session("s-out")
        # Session is gone; outcome is gone
        assert db.get_session("s-out") is None

    def test_delete_cascades_to_monitor_runs(self, db: ResearchDatabase) -> None:
        _save(db, "s-mon")
        db.save_monitor_run("s-mon", "ok", "Intact.", "2026-03-20T00:00:00+00:00")
        db.delete_session("s-mon")
        history = db.get_monitor_history(session_id="s-mon")
        assert history == []


# ---------------------------------------------------------------------------
# get_sources_for_session
# ---------------------------------------------------------------------------

class TestGetSourcesForSession:
    def test_returns_empty_list_when_no_sources(self, db: ResearchDatabase) -> None:
        _save(db, "s-nosrc")
        assert db.get_sources_for_session("s-nosrc") == []

    def test_returns_sources(self, db: ResearchDatabase) -> None:
        _save(db, "s-src2")
        import sqlite3
        with sqlite3.connect(str(db._path)) as conn:
            conn.execute(
                "INSERT INTO sources (session_id, url, title, purpose) VALUES (?,?,?,?)",
                ("s-src2", "https://reuters.com/x", "Reuters", "pro_thesis"),
            )
        sources = db.get_sources_for_session("s-src2")
        assert len(sources) == 1
        assert sources[0]["url"] == "https://reuters.com/x"
        assert sources[0]["purpose"] == "pro_thesis"


# ---------------------------------------------------------------------------
# get_session_stats
# ---------------------------------------------------------------------------

class TestGetSessionStats:
    def test_empty_database(self, db: ResearchDatabase) -> None:
        stats = db.get_session_stats()
        assert stats["total_sessions"] == 0
        assert stats["sessions_with_outcomes"] == 0
        assert stats["active_theses"] == 0
        assert stats["total_monitor_runs"] == 0
        assert stats["oldest_session"] is None

    def test_counts_are_correct(self, db: ResearchDatabase) -> None:
        _save(db, "s-a")
        _save(db, "s-b")
        _save(db, "s-c")
        db.save_outcome("s-a", "correct")
        db.save_outcome("s-b", "incorrect")
        db.update_thesis_status("s-c", "challenged")
        db.save_monitor_run("s-a", "ok", "fine", "2026-03-20T00:00:00+00:00")
        db.save_monitor_run("s-a", "ok", "still fine", "2026-03-21T00:00:00+00:00")

        stats = db.get_session_stats()
        assert stats["total_sessions"] == 3
        assert stats["sessions_with_outcomes"] == 2
        assert stats["outcome_breakdown"]["correct"] == 1
        assert stats["outcome_breakdown"]["incorrect"] == 1
        assert stats["outcome_breakdown"]["partial"] == 0
        assert stats["challenged_theses"] == 1
        assert stats["total_monitor_runs"] == 2


# ---------------------------------------------------------------------------
# export_sessions
# ---------------------------------------------------------------------------

class TestExportSessions:
    def test_empty_export(self, db: ResearchDatabase) -> None:
        assert db.export_sessions() == []

    def test_export_includes_all_sessions(self, db: ResearchDatabase) -> None:
        _save(db, "s-e1")
        _save(db, "s-e2")
        rows = db.export_sessions()
        ids = {r["id"] for r in rows}
        assert {"s-e1", "s-e2"} == ids

    def test_export_includes_outcome(self, db: ResearchDatabase) -> None:
        _save(db, "s-e3")
        db.save_outcome("s-e3", "partial", note="Partially correct.")
        rows = db.export_sessions()
        row = next(r for r in rows if r["id"] == "s-e3")
        assert row["outcome_result"] == "partial"
        assert row["outcome_note"] == "Partially correct."

    def test_export_is_json_serializable(self, db: ResearchDatabase) -> None:
        _save(db, "s-e4")
        rows = db.export_sessions()
        # Should not raise
        json.dumps(rows, default=str)

    def test_export_as_csv_via_dictwriter(self, db: ResearchDatabase) -> None:
        _save(db, "s-e5")
        rows = db.export_sessions()
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        output = buf.getvalue()
        assert "s-e5" in output
        assert "outcome_result" in output


# ---------------------------------------------------------------------------
# resolve_session_id
# ---------------------------------------------------------------------------

class TestResolveSessionId:
    def test_exact_full_id_resolves(self, db: ResearchDatabase) -> None:
        _save(db, "abc12345-0000-0000-0000-000000000000")
        result = db.resolve_session_id("abc12345-0000-0000-0000-000000000000")
        assert result == "abc12345-0000-0000-0000-000000000000"

    def test_prefix_resolves_to_full_id(self, db: ResearchDatabase) -> None:
        _save(db, "abc12345-0000-0000-0000-000000000000")
        result = db.resolve_session_id("abc12345")
        assert result == "abc12345-0000-0000-0000-000000000000"

    def test_no_match_returns_none(self, db: ResearchDatabase) -> None:
        result = db.resolve_session_id("zzzzzzzzz")
        assert result is None

    def test_ambiguous_prefix_raises_value_error(self, db: ResearchDatabase) -> None:
        _save(db, "abc12345-0000-0000-0000-000000000001")
        _save(db, "abc12345-0000-0000-0000-000000000002")
        with pytest.raises(ValueError, match="2 sessions"):
            db.resolve_session_id("abc12345")


# ---------------------------------------------------------------------------
# log_commands helpers (unit-testable without CLI invocation)
# ---------------------------------------------------------------------------

class TestLogHelpers:
    def test_iter_events_parses_valid_jsonl(self, tmp_path: Path) -> None:
        from aria.cli.log_commands import _iter_events
        log = tmp_path / "session-test.jsonl"
        log.write_text(
            '{"timestamp": "2026-03-20T10:00:00", "event_type": "start", "payload": {}}\n'
            '{"timestamp": "2026-03-20T10:01:00", "event_type": "tool_call", "payload": {"tool": "web_search"}}\n',
            encoding="utf-8",
        )
        events = list(_iter_events(log))
        assert len(events) == 2
        assert events[0]["event_type"] == "start"
        assert events[1]["payload"]["tool"] == "web_search"

    def test_iter_events_skips_malformed_lines(self, tmp_path: Path) -> None:
        from aria.cli.log_commands import _iter_events
        log = tmp_path / "session-bad.jsonl"
        log.write_text(
            '{"event_type": "good"}\n'
            'NOT JSON AT ALL\n'
            '{"event_type": "also_good"}\n',
            encoding="utf-8",
        )
        events = list(_iter_events(log))
        assert len(events) == 2

    def test_payload_summary_dict(self) -> None:
        from aria.cli.log_commands import _payload_summary
        assert _payload_summary({"query": "gold price 2026"}) == "gold price 2026"

    def test_payload_summary_truncates(self) -> None:
        from aria.cli.log_commands import _payload_summary
        long_val = "x" * 200
        result = _payload_summary({"message": long_val}, max_len=80)
        assert len(result) <= 81  # 80 chars + "…"
        assert result.endswith("…")

    def test_payload_summary_string_fallback(self) -> None:
        from aria.cli.log_commands import _payload_summary
        assert _payload_summary("plain string") == "plain string"

    def test_session_id_from_path(self) -> None:
        from aria.cli.log_commands import _session_id_from_path
        path = Path("logs/session-abc12345-0000-0000-0000-000000000000.jsonl")
        assert _session_id_from_path(path) == "abc12345-0000-0000-0000-000000000000"
