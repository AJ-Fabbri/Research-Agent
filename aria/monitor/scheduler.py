"""
Thesis monitoring scheduler.

Two entry points:
  run_once(config)   — check all active theses immediately, print results, send notifications
  start_scheduler(config) — block and run checks on the configured interval (APScheduler)
"""
from __future__ import annotations

import sys
from typing import Optional

from aria.config import AriaConfig

from .checker import MonitorResult, ThesisChecker
from .notifier import DiscordNotifier


def run_once(config: AriaConfig, session_id: Optional[str] = None) -> list[MonitorResult]:
    """
    Run thesis checks immediately.

    If session_id is given, checks only that session.
    Otherwise checks all active sessions.
    Prints a summary and sends Discord notifications if configured.
    """
    checker = ThesisChecker(config)
    monitor_cfg = config.monitor

    if session_id:
        result = checker.check_session(session_id)
        results = [result] if result else []
    else:
        results = checker.check_all()

    _print_results(results)

    webhook = monitor_cfg.discord_webhook
    if webhook and not webhook.startswith("$"):
        notifier = DiscordNotifier(webhook)
        notifier.send_summary(results, notify_on=monitor_cfg.notify_on)
        challenged = [r for r in results if r.status == "challenged"]
        if challenged:
            print(f"\nDiscord: sent {len(challenged)} challenge alert(s).")

    return results


def start_scheduler(config: AriaConfig) -> None:
    """
    Start a blocking APScheduler job that runs thesis checks on the configured interval.
    Ctrl-C to stop.
    """
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError:
        print(
            "APScheduler is required for the scheduler. Install it with:\n"
            "  pip install apscheduler",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    interval_hours = config.monitor.interval_hours
    scheduler = BlockingScheduler()

    def _job() -> None:
        print("\n[ARIA Monitor] Running scheduled thesis check…")
        run_once(config)

    scheduler.add_job(_job, "interval", hours=interval_hours)
    print(
        f"[ARIA Monitor] Scheduler started — checking every {interval_hours}h. "
        f"Press Ctrl-C to stop."
    )

    # Run once immediately on startup so you don't wait an entire interval.
    _job()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n[ARIA Monitor] Scheduler stopped.")


# ------------------------------------------------------------------
# Private
# ------------------------------------------------------------------

def _print_results(results: list[MonitorResult]) -> None:
    if not results:
        print("No active theses to check.")
        return

    print(f"\n{'Session':8}  {'Ticker':6}  {'Status':11}  Summary")
    print("-" * 80)
    for r in results:
        sid = r.session_id[:8]
        ticker = (r.ticker or "—")[:6]
        status = r.status[:11]
        summary = r.summary[:55] + "…" if len(r.summary) > 55 else r.summary
        print(f"{sid}  {ticker:6}  {status:11}  {summary}")

    challenged = [r for r in results if r.status == "challenged"]
    if challenged:
        print(f"\n{len(challenged)} thesis(es) CHALLENGED:")
        for r in challenged:
            print(f"\n  Session {r.session_id[:8]} | {r.ticker or '—'}")
            print(f"  Thesis: {r.thesis[:120]}")
            print(f"  Assessment: {r.summary}")
            print(f"  → aria outcome {r.session_id} --result correct|incorrect|partial")
