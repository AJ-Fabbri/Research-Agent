"""
ThesisChecker: evaluates whether stored failure conditions have materialized.

For each active session, fetches fresh web data and financial prices, then
asks the LLM whether any of the named failure conditions have triggered.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from aria.config import AriaConfig
from aria.models import ModelRouter, TaskType
from aria.storage.db import ResearchDatabase
from aria.tools import FinancialDataClient, WebSearchClient

logger = logging.getLogger(__name__)


_MONITOR_SYSTEM_PROMPT = """\
You are a thesis monitoring agent. You will be given an original research thesis,
its specific failure conditions, and fresh market data and recent news.

Your job: determine whether any of the failure conditions have triggered.

Respond with EXACTLY this format — no other text before or after:

STATUS: ok
REASON: [one paragraph]

or:

STATUS: challenged
REASON: [one paragraph — name which condition(s) triggered and cite the evidence]

Rules:
- "challenged" means a specific failure condition has materially triggered, not just \
moved in that direction.
- "ok" means the thesis is intact — conditions have not triggered.
- Be conservative: lean toward "ok" unless a condition has clearly and specifically triggered.
- One paragraph, 3-5 sentences. Name the specific condition. Cite the data.
- For every factual claim in REASON, add a citation in the form (Source: Title, URL). \
If a fact comes from the financial data block, cite it as (Source: financial data).
"""


@dataclass
class MonitorResult:
    session_id: str
    ticker: Optional[str]
    thesis: str
    failure_conditions: str
    status: str          # "ok" | "challenged" | "error"
    summary: str
    checked_at: str


def _parse_monitor_response(text: str) -> Tuple[str, str]:
    """Extract (status, reason) from monitor LLM response."""
    status = "ok"
    reason = text.strip()

    m = re.search(r"STATUS:\s*(ok|challenged)", text, re.IGNORECASE)
    if m:
        status = m.group(1).lower()

    r = re.search(r"REASON:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if r:
        reason = r.group(1).strip()

    return status, reason


class ThesisChecker:
    """
    Evaluates active theses against their stored failure conditions using fresh data.

    For each session:
      1. Fetches 3-month price history for the ticker (if any).
      2. Runs a targeted web search for recent news.
      3. Asks the LLM whether any failure condition has triggered.
      4. Persists the result to monitor_runs and updates thesis_status if challenged.
    """

    def __init__(self, config: AriaConfig) -> None:
        self._config = config
        self._router = ModelRouter(config)
        self._web = WebSearchClient(config)
        self._financial = FinancialDataClient(config)
        self._db = ResearchDatabase(Path(config.agent.db_path))

    def check_all(self) -> List[MonitorResult]:
        """Check all active (unresolved) sessions. Returns list of MonitorResults."""
        sessions = self._db.get_active_sessions()
        results = []
        for session in sessions:
            result = self._check_session(session)
            results.append(result)
            self._db.save_monitor_run(
                session_id=result.session_id,
                status=result.status,
                summary=result.summary,
                checked_at=result.checked_at,
            )
            if result.status == "challenged":
                self._db.update_thesis_status(result.session_id, "challenged")
        return results

    def check_session(self, session_id: str) -> Optional[MonitorResult]:
        """Check a single session by ID."""
        session = self._db.get_session(session_id)
        if not session:
            return None
        result = self._check_session(session)
        self._db.save_monitor_run(
            session_id=result.session_id,
            status=result.status,
            summary=result.summary,
            checked_at=result.checked_at,
        )
        if result.status == "challenged":
            self._db.update_thesis_status(result.session_id, "challenged")
        return result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _check_session(self, session: Dict[str, Any]) -> MonitorResult:
        checked_at = datetime.now(timezone.utc).isoformat()
        session_id = session["id"]
        ticker = session.get("ticker")
        thesis = session.get("thesis") or ""
        failure_conditions = session.get("failure_conditions") or ""

        if not failure_conditions or not thesis:
            return MonitorResult(
                session_id=session_id,
                ticker=ticker,
                thesis=thesis,
                failure_conditions=failure_conditions,
                status="error",
                summary="No thesis or failure conditions stored — cannot evaluate.",
                checked_at=checked_at,
            )

        query_text = session.get("query") or ""
        fresh_data = self._gather_data(ticker, failure_conditions, query_text, thesis)

        try:
            status, summary = self._evaluate(thesis, failure_conditions, fresh_data)
        except Exception as exc:
            return MonitorResult(
                session_id=session_id,
                ticker=ticker,
                thesis=thesis,
                failure_conditions=failure_conditions,
                status="error",
                summary=f"LLM evaluation failed: {exc}",
                checked_at=checked_at,
            )

        return MonitorResult(
            session_id=session_id,
            ticker=ticker,
            thesis=thesis,
            failure_conditions=failure_conditions,
            status=status,
            summary=summary,
            checked_at=checked_at,
        )

    def _extract_company_name(self, ticker: str, thesis: str, query: str) -> str:
        """
        Ask the LLM to extract the full company name for this ticker from the
        session text, so web searches are not confused by ticker ambiguity.
        Returns the company name, or the bare ticker if extraction fails.
        """
        routed = self._router.select_model(TaskType.DIRECT_ANSWER)
        prompt = (
            f"What is the full company name for ticker symbol {ticker}?\n"
            f"Answer with just the company name — no explanation, no punctuation.\n\n"
            f"Context:\n{query}\n\n{thesis[:500]}"
        )
        try:
            response = routed.model.invoke([HumanMessage(content=prompt)])
            name = getattr(response, "content", str(response)).strip().strip('"\'')
            name = re.sub(r"<think>.*?</think>", "", name, flags=re.DOTALL).strip()
            return name if name else ticker
        except Exception:
            return ticker

    def _gather_data(
        self,
        ticker: Optional[str],
        failure_conditions: str,
        query: str = "",
        thesis: str = "",
    ) -> str:
        """Pull fresh financial and news data for the thesis check."""
        parts: List[str] = [f"Date: {date.today().isoformat()}"]

        # 1-year price return (matches the thesis tracking window)
        if ticker:
            try:
                start = (date.today() - timedelta(days=365)).isoformat()
                hist = self._financial.price_history(ticker, start=start)
                if not hist.empty:
                    close_col = "Close" if "Close" in hist.columns else hist.columns[-1]
                    first = float(hist[close_col].iloc[0])
                    last = float(hist[close_col].iloc[-1])
                    ret = (last - first) / first
                    parts.append(f"{ticker} 1-year price return: {ret:+.1%}")
                    parts.append(f"{ticker} current price: {last:.2f}")
            except Exception as exc:
                logger.warning("Price history failed for %s: %s", ticker, exc)
                parts.append(f"[Price data unavailable for {ticker}: {exc}]")

        # Two targeted searches:
        # 1. General earnings/revenue/guidance news.
        # 2. A search targeting the specific metrics named in the failure conditions.
        # Use company name (not bare ticker) to avoid ticker ambiguity (e.g. MTN Group).
        company = self._extract_company_name(ticker, thesis, query) if ticker else ""
        searches: List[str] = []
        if company:
            searches.append(f"{company} earnings revenue guidance news")
            # Build a focused query from the failure conditions (first 120 chars of
            # each condition line, joined) to surface the specific metrics we need.
            fc_terms = " ".join(
                line.strip()[:120]
                for line in failure_conditions.splitlines()
                if line.strip()
            )[:200]
            if fc_terms:
                searches.append(f"{company} {fc_terms}")
        else:
            searches.append("market conditions latest news")

        for query in searches:
            try:
                results = self._web.search(query, max_results=4)
                if results:
                    parts.append(f"\nSearch: {query!r}")
                    for r in results[:4]:
                        snippet = r.content[:400].replace("\n", " ")
                        parts.append(f"- [{r.title}]({r.url}): {snippet}")
            except Exception as exc:
                logger.warning("Web search failed in monitor (%r): %s", query, exc)
                parts.append(f"[Search failed for {query!r}: {exc}]")

        return "\n".join(parts)

    def _evaluate(
        self,
        thesis: str,
        failure_conditions: str,
        fresh_data: str,
    ) -> Tuple[str, str]:
        """Ask the LLM whether any failure condition has triggered."""
        routed = self._router.select_model(TaskType.DIRECT_ANSWER)

        user_content = (
            f"THESIS:\n{thesis}\n\n"
            f"FAILURE CONDITIONS (the specific events that would invalidate this thesis):\n"
            f"{failure_conditions}\n\n"
            f"FRESH DATA (as of today):\n{fresh_data}\n\n"
            f"Have any of the failure conditions triggered?"
        )

        messages = [
            SystemMessage(content=_MONITOR_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        response = routed.model.invoke(messages)
        text = getattr(response, "content", str(response))
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return _parse_monitor_response(text)
