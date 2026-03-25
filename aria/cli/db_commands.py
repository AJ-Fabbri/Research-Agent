"""
`aria db` subcommand group — rich views and management for the session database.

Commands:
  aria db list   [--ticker X] [--status active|challenged|resolved]
                 [--since YYYY-MM-DD] [--outcome correct|incorrect|partial|pending]
                 [--limit N]
  aria db show   <session_id>
  aria db delete <session_id> [--force]
  aria db export [--format json|csv] [--output path]
  aria db stats
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

from aria.config import load_config
from aria.storage.db import ResearchDatabase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _db(args: argparse.Namespace) -> ResearchDatabase:
    config = load_config(getattr(args, "config", "aria_config.yaml"))
    return ResearchDatabase(Path(config.agent.db_path))


def _resolve(db: ResearchDatabase, prefix: str) -> Optional[str]:
    """Resolve prefix → full session ID, printing errors and returning None on failure."""
    try:
        full_id = db.resolve_session_id(prefix)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return None
    if full_id is None:
        print(f"No session found matching {prefix!r}.", file=sys.stderr)
    return full_id


def _memo_path_for(session_id: str) -> Optional[Path]:
    """Find a saved memo file that contains the first 8 chars of the session ID."""
    short = session_id[:8]
    matches = sorted(Path("memos").glob(f"*_{short}*.md"))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# aria db list
# ---------------------------------------------------------------------------

def _cmd_db_list(args: argparse.Namespace) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich import box

    db = _db(args)
    sessions = db.list_sessions(
        ticker=args.ticker,
        unresolved_only=(args.outcome == "pending") if args.outcome else args.unresolved,
        limit=args.limit,
    )

    # Local filtering not supported by list_sessions()
    if args.status:
        sessions = [s for s in sessions if s.get("thesis_status") == args.status]
    if args.since:
        sessions = [s for s in sessions if (s.get("created_at") or "") >= args.since]
    if args.outcome and args.outcome != "pending":
        sessions = [s for s in sessions if s.get("outcome_result") == args.outcome]

    if not sessions:
        print("No sessions found.")
        return

    console = Console()
    table = Table(box=box.SIMPLE_HEAD, show_lines=False, expand=True)
    table.add_column("ID", style="cyan", no_wrap=True, width=9)
    table.add_column("Date", width=10)
    table.add_column("Ticker", width=6)
    table.add_column("Status", width=11)
    table.add_column("Outcome", width=11)
    table.add_column("Model", width=22)
    table.add_column("Thesis", ratio=1)

    status_style = {"active": "green", "challenged": "yellow", "resolved": "dim"}
    outcome_style = {"correct": "green", "incorrect": "red", "partial": "yellow"}

    for s in sessions:
        sid = (s.get("id") or "")[:8]
        created = (s.get("created_at") or "")[:10]
        ticker = (s.get("ticker") or "—")[:6]
        status = s.get("thesis_status") or "active"
        outcome = s.get("outcome_result") or "pending"
        model = (s.get("model_name") or "—")[:22]
        thesis_raw = s.get("thesis") or s.get("query") or ""
        thesis = thesis_raw[:80] + "…" if len(thesis_raw) > 80 else thesis_raw

        table.add_row(
            sid,
            created,
            ticker,
            f"[{status_style.get(status, '')}]{status}[/]",
            f"[{outcome_style.get(outcome, 'dim')}]{outcome}[/]",
            model,
            thesis,
        )

    console.print(table)
    console.print(
        f"[dim]{len(sessions)} session(s). "
        "Use [bold]aria db show <id>[/bold] to view a full memo.[/dim]"
    )


# ---------------------------------------------------------------------------
# aria db show
# ---------------------------------------------------------------------------

def _cmd_db_show(args: argparse.Namespace) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box

    db = _db(args)
    full_id = _resolve(db, args.session_id)
    if not full_id:
        raise SystemExit(1)

    session = db.get_session(full_id)
    if not session:
        print(f"Session {full_id!r} not found.", file=sys.stderr)
        raise SystemExit(1)

    sources = db.get_sources_for_session(full_id)
    console = Console(width=100)

    # --- Header ---
    meta = (
        f"[bold cyan]{full_id[:8]}…[/bold cyan]"
        f"  [dim]|[/dim]  {(session.get('created_at') or '')[:10]}"
        f"  [dim]|[/dim]  {session.get('model_name') or '—'}"
        f"  [dim]|[/dim]  ticker: [bold]{session.get('ticker') or '—'}[/bold]"
        f"  [dim]|[/dim]  status: [bold]{session.get('thesis_status') or 'active'}[/bold]"
    )
    outcome = session.get("outcome_result")
    if outcome:
        meta += f"  [dim]|[/dim]  outcome: [bold]{outcome}[/bold]"
        if session.get("outcome_note"):
            meta += f" ({session['outcome_note']})"

    console.print(Panel(meta, title="Session", border_style="cyan"))

    # --- Query ---
    if session.get("query"):
        console.print(Panel(session["query"], title="[bold]Query[/bold]", border_style="blue"))

    # --- Analytical sections ---
    sections = [
        ("Thesis", "thesis"),
        ("Confidence", "confidence"),
        ("Baselines", "baselines"),
        ("Supporting Evidence", "supporting_evidence"),
        ("Counter Evidence", "counter_evidence"),
        ("Segment Behavior", "segment_behavior"),
        ("Failure Conditions", "failure_conditions"),
        ("Conclusion", "conclusion"),
    ]
    for title, key in sections:
        content = session.get(key)
        if content:
            console.print(Panel(content, title=f"[bold]{title}[/bold]", border_style="white"))

    # --- Sources ---
    if sources:
        src_table = Table(box=box.SIMPLE, show_header=True, expand=True)
        src_table.add_column("Purpose", width=14, style="dim")
        src_table.add_column("Title", ratio=1)
        src_table.add_column("URL", ratio=2, style="blue")
        purpose_style = {"background": "dim", "pro_thesis": "green", "counter_thesis": "red"}
        for src in sources:
            p = src.get("purpose") or ""
            src_table.add_row(
                f"[{purpose_style.get(p, '')}]{p}[/]",
                src.get("title") or "—",
                src.get("url") or "—",
            )
        console.print(Panel(src_table, title="[bold]Sources[/bold]", border_style="dim"))

    # --- Saved memo file ---
    memo = _memo_path_for(full_id)
    if memo and memo.exists():
        console.print(f"\n[dim]Memo file:[/dim] [bold]{memo}[/bold]")

    # --- Record outcome hint ---
    if not outcome:
        console.print(
            f"\n[dim]Record outcome:[/dim] "
            f"[bold]aria outcome {full_id} --result correct|incorrect|partial[/bold]"
        )


# ---------------------------------------------------------------------------
# aria db delete
# ---------------------------------------------------------------------------

def _cmd_db_delete(args: argparse.Namespace) -> None:
    db = _db(args)
    full_id = _resolve(db, args.session_id)
    if not full_id:
        raise SystemExit(1)

    session = db.get_session(full_id)
    if not session:
        print(f"Session {full_id!r} not found.", file=sys.stderr)
        raise SystemExit(1)

    # Show what will be deleted.
    thesis = (session.get("thesis") or session.get("query") or "")[:120]
    print(f"\nSession: {full_id[:8]}…")
    print(f"Date:    {(session.get('created_at') or '')[:10]}")
    print(f"Thesis:  {thesis}")

    memo = _memo_path_for(full_id)
    log_file = Path("logs") / f"session-{full_id}.jsonl"

    if not args.force:
        resp = input(
            "\nDelete this session and all related data (sources, outcomes, monitor runs)? [y/N] "
        ).strip().lower()
        if resp != "y":
            print("Aborted.")
            return

    ok = db.delete_session(full_id)
    if not ok:
        print("Delete failed — session not found.", file=sys.stderr)
        raise SystemExit(1)

    print(f"Deleted session {full_id[:8]}…")

    # Offer to delete associated files.
    for path, label in [(memo, "memo file"), (log_file if log_file.exists() else None, "log file")]:
        if path and path.exists():
            if args.force:
                path.unlink()
                print(f"Deleted {label}: {path}")
            else:
                resp = input(f"Also delete {label} ({path})? [y/N] ").strip().lower()
                if resp == "y":
                    path.unlink()
                    print(f"Deleted {label}: {path}")


# ---------------------------------------------------------------------------
# aria db export
# ---------------------------------------------------------------------------

def _cmd_db_export(args: argparse.Namespace) -> None:
    db = _db(args)
    sessions = db.export_sessions()

    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout

    try:
        if args.format == "csv":
            if not sessions:
                return
            writer = csv.DictWriter(out, fieldnames=list(sessions[0].keys()))
            writer.writeheader()
            writer.writerows(sessions)
        else:
            json.dump(sessions, out, indent=2, default=str)
            out.write("\n")
    finally:
        if args.output:
            out.close()
            print(f"Exported {len(sessions)} session(s) to {args.output}")


# ---------------------------------------------------------------------------
# aria db stats
# ---------------------------------------------------------------------------

def _cmd_db_stats(args: argparse.Namespace) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box

    db = _db(args)
    s = db.get_session_stats()
    console = Console()

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold")

    total = s["total_sessions"]
    with_out = s["sessions_with_outcomes"]
    bd = s["outcome_breakdown"]
    correct_pct = f"{bd['correct'] / with_out * 100:.0f}%" if with_out else "—"

    table.add_row("Total sessions", str(total))
    table.add_row("With outcomes", f"{with_out} / {total}")
    table.add_row("Correct / Incorrect / Partial",
                  f"[green]{bd['correct']}[/green] / "
                  f"[red]{bd['incorrect']}[/red] / "
                  f"[yellow]{bd['partial']}[/yellow]"
                  + (f"  ([green]{correct_pct}[/green] accuracy)" if with_out else ""))
    table.add_row("Active theses", f"[green]{s['active_theses']}[/green]")
    table.add_row("Challenged theses", f"[yellow]{s['challenged_theses']}[/yellow]")
    table.add_row("Monitor runs", str(s["total_monitor_runs"]))
    if s["oldest_session"]:
        table.add_row("Date range",
                      f"{s['oldest_session'][:10]}  →  {s['newest_session'][:10]}")  # type: ignore[index]

    console.print(Panel(table, title="[bold]ARIA Database Stats[/bold]", border_style="cyan"))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _cmd_db(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="aria db",
        description="View and manage the ARIA research database.",
    )
    parser.add_argument("--config", default="aria_config.yaml")
    subparsers = parser.add_subparsers(dest="action", required=True)

    # list
    lp = subparsers.add_parser("list", help="List sessions with rich table output.")
    lp.add_argument("--ticker", default=None)
    lp.add_argument("--status", choices=["active", "challenged", "resolved"], default=None)
    lp.add_argument("--since", default=None, metavar="YYYY-MM-DD")
    lp.add_argument("--outcome", choices=["correct", "incorrect", "partial", "pending"],
                    default=None)
    lp.add_argument("--unresolved", action="store_true")
    lp.add_argument("--limit", type=int, default=30)

    # show
    sp = subparsers.add_parser("show", help="Show full memo and sources for a session.")
    sp.add_argument("session_id")

    # delete
    dp = subparsers.add_parser("delete", help="Delete a session and all related data.")
    dp.add_argument("session_id")
    dp.add_argument("--force", action="store_true", help="Skip confirmation prompts.")

    # export
    ep = subparsers.add_parser("export", help="Export sessions to JSON or CSV.")
    ep.add_argument("--format", choices=["json", "csv"], default="json")
    ep.add_argument("--output", default=None, metavar="PATH")

    # stats
    subparsers.add_parser("stats", help="Show database statistics.")

    args = parser.parse_args(argv)

    dispatch = {
        "list": _cmd_db_list,
        "show": _cmd_db_show,
        "delete": _cmd_db_delete,
        "export": _cmd_db_export,
        "stats": _cmd_db_stats,
    }
    dispatch[args.action](args)
