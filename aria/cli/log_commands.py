"""
`aria logs` subcommand group — browse and search structured session JSONL logs.

Commands:
  aria logs show   <session_id> [--raw] [--limit N] [--event-type TYPE]
  aria logs search <pattern>    [--session ID] [--event-type TYPE] [--limit N]
  aria logs list                [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


_LOG_DIR = Path("logs")


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _log_path(session_id: str) -> Path:
    return _LOG_DIR / f"session-{session_id}.jsonl"


def _iter_events(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file, skipping malformed lines."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _all_log_files() -> List[Path]:
    if not _LOG_DIR.exists():
        return []
    return sorted(_LOG_DIR.glob("session-*.jsonl"))


def _session_id_from_path(path: Path) -> str:
    """Extract the session UUID from a log filename."""
    # filename: session-<uuid>.jsonl
    return path.stem.removeprefix("session-")


def _payload_summary(payload: Any, max_len: int = 80) -> str:
    """Compact one-line summary of an event payload."""
    if isinstance(payload, dict):
        # Show the most informative fields first
        for key in ("query", "section", "tool", "message", "result", "error"):
            if key in payload:
                val = str(payload[key])
                return val[:max_len] + "…" if len(val) > max_len else val
        # Fallback: first key=value pair
        for k, v in payload.items():
            val = f"{k}={v!r}"
            return val[:max_len] + "…" if len(val) > max_len else val
    text = str(payload)
    return text[:max_len] + "…" if len(text) > max_len else text


# ---------------------------------------------------------------------------
# aria logs list
# ---------------------------------------------------------------------------

def _cmd_logs_list(args: argparse.Namespace) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich import box

    files = _all_log_files()
    if not files:
        print("No session log files found in logs/.")
        return

    console = Console()
    table = Table(box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Session ID", style="cyan", width=38)
    table.add_column("Size", width=8)
    table.add_column("Events", width=7)
    table.add_column("Modified", width=10)

    for path in reversed(files[-args.limit :]):
        sid = _session_id_from_path(path)
        size_kb = f"{path.stat().st_size / 1024:.1f}K"
        events = sum(1 for _ in _iter_events(path))
        mtime = path.stat().st_mtime
        from datetime import datetime
        modified = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
        table.add_row(sid, size_kb, str(events), modified)

    console.print(table)


# ---------------------------------------------------------------------------
# aria logs show
# ---------------------------------------------------------------------------

def _resolve_log_path(session_prefix: str) -> Optional[Path]:
    """Find a log file by full or partial session ID."""
    # Try exact match first
    exact = _log_path(session_prefix)
    if exact.exists():
        return exact

    # Glob for prefix
    if not _LOG_DIR.exists():
        return None
    matches = list(_LOG_DIR.glob(f"session-{session_prefix}*.jsonl"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(
            f"Ambiguous prefix {session_prefix!r} matches:\n"
            + "\n".join(f"  {_session_id_from_path(m)}" for m in matches),
            file=sys.stderr,
        )
        return None
    return None


def _cmd_logs_show(args: argparse.Namespace) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich import box

    path = _resolve_log_path(args.session_id)
    if path is None:
        print(f"No log file found for session {args.session_id!r}.", file=sys.stderr)
        raise SystemExit(1)

    if args.raw:
        sys.stdout.write(path.read_text(encoding="utf-8"))
        return

    events = list(_iter_events(path))
    if args.event_type:
        events = [e for e in events if e.get("event_type") == args.event_type]

    # Most recent first if --limit applies
    if len(events) > args.limit:
        print(f"[Showing last {args.limit} of {len(events)} events — use --limit to change]")
        events = events[-args.limit :]

    console = Console(width=120)
    table = Table(box=box.SIMPLE, expand=True, show_lines=False)
    table.add_column("Time (UTC)", style="dim", width=19, no_wrap=True)
    table.add_column("Event", style="bold cyan", width=20, no_wrap=True)
    table.add_column("Summary", ratio=1)

    for ev in events:
        ts = (ev.get("timestamp") or "")[:19].replace("T", " ")
        etype = ev.get("event_type") or "—"
        summary = _payload_summary(ev.get("payload", {}))
        table.add_row(ts, etype, summary)

    session_id = _session_id_from_path(path)
    console.print(table)
    console.print(
        f"[dim]{len(events)} event(s) | session {session_id[:8]}… | {path}[/dim]"
    )


# ---------------------------------------------------------------------------
# aria logs search
# ---------------------------------------------------------------------------

def _cmd_logs_search(args: argparse.Namespace) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich import box

    pattern = re.compile(args.pattern, re.IGNORECASE)
    files = _all_log_files()
    if args.session:
        path = _resolve_log_path(args.session)
        files = [path] if path else []

    console = Console(width=120)
    table = Table(box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Session", style="cyan", width=9, no_wrap=True)
    table.add_column("Time (UTC)", style="dim", width=19, no_wrap=True)
    table.add_column("Event", width=20, no_wrap=True)
    table.add_column("Match", ratio=1)

    hits = 0
    for path in files:
        sid = _session_id_from_path(path)[:8]
        for ev in _iter_events(path):
            if args.event_type and ev.get("event_type") != args.event_type:
                continue
            raw = json.dumps(ev.get("payload", {}))
            if not pattern.search(raw):
                continue
            ts = (ev.get("timestamp") or "")[:19].replace("T", " ")
            etype = ev.get("event_type") or "—"
            summary = _payload_summary(ev.get("payload", {}))
            table.add_row(sid, ts, etype, summary)
            hits += 1
            if hits >= args.limit:
                break
        if hits >= args.limit:
            break

    if hits == 0:
        print(f"No matches for {args.pattern!r}.")
        return

    console.print(table)
    console.print(f"[dim]{hits} match(es)[/dim]")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _cmd_logs(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="aria logs",
        description="Browse and search structured session log files.",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    # list
    lp = subparsers.add_parser("list", help="List all session log files.")
    lp.add_argument("--limit", type=int, default=30)

    # show
    sp = subparsers.add_parser("show", help="Display events from a session log.")
    sp.add_argument("session_id")
    sp.add_argument("--raw", action="store_true", help="Dump raw JSONL to stdout.")
    sp.add_argument("--limit", type=int, default=50,
                    help="Max events to show (default 50).")
    sp.add_argument("--event-type", default=None, metavar="TYPE",
                    help="Filter to a specific event type.")

    # search
    srp = subparsers.add_parser("search", help="Search for a pattern across log files.")
    srp.add_argument("pattern", help="Regex pattern to search for in event payloads.")
    srp.add_argument("--session", default=None, metavar="SESSION_ID",
                     help="Limit search to one session.")
    srp.add_argument("--event-type", default=None, metavar="TYPE")
    srp.add_argument("--limit", type=int, default=50)

    args = parser.parse_args(argv)

    dispatch = {
        "list": _cmd_logs_list,
        "show": _cmd_logs_show,
        "search": _cmd_logs_search,
    }
    dispatch[args.action](args)
