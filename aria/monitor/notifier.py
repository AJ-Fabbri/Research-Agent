"""
Discord webhook notifier for thesis status changes.

Sends a formatted embed when a thesis is challenged or (optionally) on every check.
No discord.py dependency — uses a plain httpx POST to the webhook URL.
"""
from __future__ import annotations

from typing import Any

import httpx

from .checker import MonitorResult

# Discord embed colors
_COLOR_CHALLENGED = 0xE74C3C   # red
_COLOR_OK = 0x2ECC71            # green
_COLOR_ERROR = 0x95A5A6         # grey


def _truncate(text: str, max_len: int = 1024) -> str:
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


class DiscordNotifier:
    """
    Posts thesis monitoring results to a Discord channel via webhook.

    Usage:
        notifier = DiscordNotifier(webhook_url)
        notifier.send(result)
    """

    def __init__(self, webhook_url: str) -> None:
        self._url = webhook_url

    def send(self, result: MonitorResult, notify_on: str = "challenged") -> bool:
        """
        Post a monitoring result to Discord.

        notify_on:
          "challenged" — only post when status is "challenged"
          "all"        — post for every result

        Returns True on success, False on any HTTP/network error.
        """
        if notify_on == "challenged" and result.status != "challenged":
            return True  # nothing to send

        payload = self._build_payload(result)
        try:
            resp = httpx.post(self._url, json=payload, timeout=10)
            resp.raise_for_status()
            return True
        except Exception:
            return False

    def send_summary(
        self,
        results: list[MonitorResult],
        notify_on: str = "challenged",
    ) -> None:
        """Send a batch of results, one message per challenged thesis."""
        for result in results:
            self.send(result, notify_on=notify_on)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_payload(self, result: MonitorResult) -> dict[str, Any]:
        color = {
            "challenged": _COLOR_CHALLENGED,
            "ok": _COLOR_OK,
            "error": _COLOR_ERROR,
        }.get(result.status, _COLOR_ERROR)

        status_label = {
            "challenged": "CHALLENGED",
            "ok": "OK — thesis intact",
            "error": "ERROR",
        }.get(result.status, result.status.upper())

        ticker_str = result.ticker or "Unknown"
        title = f"ARIA Thesis Monitor | {ticker_str} — {status_label}"

        fields = [
            {
                "name": "Thesis",
                "value": _truncate(result.thesis, 512),
                "inline": False,
            },
            {
                "name": "Failure Conditions",
                "value": _truncate(result.failure_conditions, 512),
                "inline": False,
            },
            {
                "name": "Assessment",
                "value": _truncate(result.summary, 1024),
                "inline": False,
            },
            {
                "name": "Session ID",
                "value": f"`{result.session_id[:8]}…`",
                "inline": True,
            },
            {
                "name": "Checked",
                "value": result.checked_at[:10],
                "inline": True,
            },
        ]

        footer_text = (
            "Record outcome: aria outcome <session_id> --result correct|incorrect|partial"
            if result.status == "challenged"
            else "ARIA Thesis Monitor"
        )

        return {
            "embeds": [
                {
                    "title": title,
                    "color": color,
                    "fields": fields,
                    "footer": {"text": footer_text},
                }
            ]
        }
