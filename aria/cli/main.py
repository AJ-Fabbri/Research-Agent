"""
Command-line entrypoint for ARIA.

Usage:
  aria "your research query"                        — run research (autonomous or checkpoint)
  aria outcome <session_id> --result correct|...   — record whether a thesis held up
  aria history [--ticker NVDA] [--unresolved]      — browse past sessions
  aria monitor run [--session <id>]                — check active theses now
  aria monitor start                               — run checks on a schedule (blocking)
  aria monitor history [--session <id>]            — view past monitor runs
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

# Ensure project root is on sys.path so 'aria' is importable when run directly
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from aria.agent import AgentLoop, AgentMode, OutputFormat  # noqa: E402
from aria.config import load_config  # noqa: E402
from aria.storage.db import ResearchDatabase  # noqa: E402


# ---------------------------------------------------------------------------
# Research command (default)
# ---------------------------------------------------------------------------

def _parse_research_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="aria", description="ARIA research agent CLI")
    parser.add_argument(
        "query",
        nargs="?",
        help="Research query to execute. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--mode",
        choices=[m.value for m in AgentMode],
        default=None,
        help="Execution mode: autonomous or checkpoint (default from config).",
    )
    parser.add_argument(
        "--format",
        choices=[f.value for f in OutputFormat],
        default=OutputFormat.AUTO.value,
        help="Output format: auto, memo, direct, or matrix.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="aria_config.yaml",
        help="Path to aria_config.yaml (default: ./aria_config.yaml).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the markdown output.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=True,
        help="Stream model output as it is generated (default: on).",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming; print full response at the end.",
    )
    return parser.parse_args(argv)


def _read_query_from_stdin() -> str:
    sys.stdout.write("Enter query (press Enter when done):\n")
    return input().strip()


def _print_sources(metadata: dict[str, Any]) -> None:
    sources = metadata.get("web_search_sources", [])
    if not sources:
        return
    print("\n--- Sources ---")
    for i, s in enumerate(sources, 1):
        purpose = s.get("purpose", "")
        tag = f" [{purpose}]" if purpose else ""
        print(f"[{i}]{tag} {s['title']}")
        print(f"    {s['url']}")


def _print_session_id(metadata: dict[str, Any]) -> None:
    session_id = metadata.get("session_id")
    if not session_id:
        return
    print(f"\nSession ID: {session_id}")
    print(f"Record outcome: aria outcome {session_id} --result correct|incorrect|partial")


def _memo_path(session_id: str, partial: bool = False) -> Path:
    """Return the auto-save path for a memo: memos/YYYY-MM-DD_<id[:8]>[_partial].md"""
    from datetime import date
    suffix = "_partial" if partial else ""
    return Path("memos") / f"{date.today().isoformat()}_{session_id[:8]}{suffix}.md"


class _Spinner:
    """Displays an animated spinner on stdout while a blocking call runs."""

    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, message: str = "Thinking") -> None:
        self._message = message
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self) -> None:
        for frame in itertools.cycle(self._FRAMES):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r{self._message} {frame} ")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write(f"\r{' ' * (len(self._message) + 3)}\r")
        sys.stdout.flush()

    def __enter__(self) -> "_Spinner":
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        self._thread.join()


def _extract_research_question(content: str) -> str:
    """
    Extract the sharpened research question from ARIA's clarification output.
    Looks for '**Research question:**' prefix; falls back to the full content.
    """
    m = re.search(r'\*\*Research question:\*\*\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return content.strip()


def _cmd_research(argv: Optional[list[str]] = None) -> None:
    args = _parse_research_args(argv)
    query = args.query or _read_query_from_stdin()
    if not query:
        print("No query provided.")
        raise SystemExit(1)

    config = load_config(args.config)
    agent = AgentLoop.from_config(config)

    mode = AgentMode(args.mode) if args.mode is not None else None
    output_format = OutputFormat(args.format)
    use_stream = getattr(args, "stream", True) and not getattr(args, "no_stream", False)
    stream_callback = (lambda s: print(s, end="", flush=True)) if use_stream else None

    # Checkpoint UX: sharpen question → user confirms or revises → full agentic analysis.
    if (mode or AgentMode(config.mode.default)) is AgentMode.CHECKPOINT:
        clarification_query = query
        sharpened = ""

        while True:
            print("Confirming research question…")
            sys.stdout.flush()

            with _Spinner("Thinking"):
                clarification = agent.run(
                    clarification_query,
                    mode=AgentMode.CHECKPOINT,
                    output_format=output_format,
                    search_query=clarification_query,
                    stream_callback=None,  # always collect; <think> blocks stripped before display
                )
            print(clarification.content)
            print()

            sharpened = _extract_research_question(clarification.content)

            resp = input(
                "\nInvestigate this? [y to confirm / type a revision / n to abort]: "
            ).strip()

            if not resp or resp.lower() == "n":
                print("Aborted.")
                return
            elif resp.lower() == "y":
                break
            else:
                # User typed feedback — re-run clarification incorporating it.
                clarification_query = (
                    f"{query}\n\nFeedback on the proposed question: {resp}\n"
                    f"Revise the research question accordingly."
                )

        print("\nRunning full analysis…")
        sys.stdout.flush()
        result = agent.run(
            sharpened,
            mode=AgentMode.AUTONOMOUS,
            output_format=output_format,
            search_query=sharpened,
            stream_callback=stream_callback,
        )
    else:
        print("Running analysis…")
        sys.stdout.flush()
        result = agent.run(
            query,
            mode=mode,
            output_format=output_format,
            search_query=query,
            stream_callback=stream_callback,
        )

    print()
    if result.content:
        print(result.content)
    elif result.metadata.get("fallback_used"):
        print("Analysis incomplete — the model stopped before writing all sections.")

    _print_sources(result.metadata)
    _print_session_id(result.metadata)

    # Auto-save all runs that produced content, complete or partial.
    if result.content:
        session_id = result.metadata.get("session_id")
        fallback_used = result.metadata.get("fallback_used", False)
        if args.save:
            save_path = Path(args.save)
        elif session_id:
            save_path = _memo_path(session_id, partial=False)
        else:
            # Partial run: no session_id, generate a short id from content hash.
            short_id = hashlib.sha256(result.content.encode()).hexdigest()[:8]
            save_path = _memo_path(short_id, partial=True)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(result.content, encoding="utf-8")
        label = "Partial memo saved" if fallback_used else "Memo saved"
        print(f"\n{label}: {save_path}")


# ---------------------------------------------------------------------------
# Outcome command
# ---------------------------------------------------------------------------

def _cmd_outcome(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="aria outcome",
        description="Record whether a thesis held up over time.",
    )
    parser.add_argument("session_id", help="Session ID printed at the end of a research run.")
    parser.add_argument(
        "--result",
        choices=["correct", "incorrect", "partial"],
        required=True,
        help="Did the thesis prove correct?",
    )
    parser.add_argument("--note", default="", help="Optional free-text note.")
    parser.add_argument("--config", default="aria_config.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    db = ResearchDatabase(Path(config.agent.db_path))

    # Show the session's thesis before recording so the user can confirm context.
    session = db.get_session(args.session_id)
    if session is None:
        print(f"Session not found: {args.session_id}")
        raise SystemExit(1)

    thesis = session.get("thesis") or "(no thesis recorded)"
    failure = session.get("failure_conditions") or ""
    print(f"\nSession:  {args.session_id[:8]}…")
    print(f"Thesis:   {thesis}")
    if failure:
        print(f"Failure:  {failure}")

    ok = db.save_outcome(args.session_id, args.result, args.note)
    if ok:
        print(f"\nRecorded: {args.result}")
        if args.note:
            print(f"Note:     {args.note}")
    else:
        print("Failed to save outcome.")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# History command
# ---------------------------------------------------------------------------

def _cmd_history(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="aria history",
        description="Browse past research sessions.",
    )
    parser.add_argument("--ticker", default=None, help="Filter by detected ticker symbol.")
    parser.add_argument(
        "--unresolved",
        action="store_true",
        help="Show only sessions with no outcome recorded.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Maximum rows to show (default 20).")
    parser.add_argument("--config", default="aria_config.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    db = ResearchDatabase(Path(config.agent.db_path))
    sessions = db.list_sessions(
        ticker=args.ticker,
        unresolved_only=args.unresolved,
        limit=args.limit,
    )

    if not sessions:
        print("No sessions found.")
        return

    print(f"\n{'ID':8}  {'Date':10}  {'Ticker':6}  {'Outcome':11}  Thesis")
    print("-" * 80)
    for s in sessions:
        sid = (s.get("id") or "")[:8]
        created = (s.get("created_at") or "")[:10]
        ticker = (s.get("ticker") or "—")[:6]
        outcome = (s.get("outcome_result") or "pending")[:11]
        thesis_raw = s.get("thesis") or s.get("query") or ""
        thesis = thesis_raw[:55] + "…" if len(thesis_raw) > 55 else thesis_raw
        print(f"{sid}  {created:10}  {ticker:6}  {outcome:11}  {thesis}")


# ---------------------------------------------------------------------------
# Monitor command
# ---------------------------------------------------------------------------

def _cmd_monitor(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="aria monitor",
        description="Monitor active theses against their failure conditions.",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    # aria monitor run
    run_p = subparsers.add_parser("run", help="Check active theses immediately.")
    run_p.add_argument("--session", default=None, help="Check a specific session ID only.")
    run_p.add_argument("--config", default="aria_config.yaml")

    # aria monitor start
    start_p = subparsers.add_parser("start", help="Run checks on a recurring schedule (blocking).")
    start_p.add_argument("--config", default="aria_config.yaml")

    # aria monitor history
    hist_p = subparsers.add_parser("history", help="View past monitoring runs.")
    hist_p.add_argument("--session", default=None, help="Filter to a specific session ID.")
    hist_p.add_argument("--limit", type=int, default=20)
    hist_p.add_argument("--config", default="aria_config.yaml")

    args = parser.parse_args(argv)

    from aria.monitor import run_once, start_scheduler
    from aria.storage.db import ResearchDatabase

    config = load_config(args.config)

    if args.action == "run":
        run_once(config, session_id=args.session)

    elif args.action == "start":
        start_scheduler(config)

    elif args.action == "history":
        db = ResearchDatabase(Path(config.agent.db_path))
        rows = db.get_monitor_history(session_id=args.session, limit=args.limit)
        if not rows:
            print("No monitor runs found.")
            return
        print(f"\n{'Session':8}  {'Date':10}  {'Status':11}  Summary")
        print("-" * 80)
        for row in rows:
            sid = (row.get("session_id") or "")[:8]
            checked = (row.get("checked_at") or "")[:10]
            status = (row.get("status") or "")[:11]
            summary_raw = row.get("summary") or ""
            summary = summary_raw[:55] + "…" if len(summary_raw) > 55 else summary_raw
            print(f"{sid}  {checked:10}  {status:11}  {summary}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    args = list(argv if argv is not None else sys.argv[1:])

    # Route subcommands before argparse so the research UX stays unchanged.
    if args and args[0] == "outcome":
        _cmd_outcome(args[1:])
        return
    if args and args[0] == "history":
        _cmd_history(args[1:])
        return
    if args and args[0] == "monitor":
        _cmd_monitor(args[1:])
        return
    if args and args[0] == "db":
        from aria.cli.db_commands import _cmd_db
        _cmd_db(args[1:])
        return
    if args and args[0] == "logs":
        from aria.cli.log_commands import _cmd_logs
        _cmd_logs(args[1:])
        return

    _cmd_research(args if args else None)


if __name__ == "__main__":
    main()
