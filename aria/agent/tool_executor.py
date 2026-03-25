"""
Tool executor for the ARIA agentic research loop.

Dispatches tool calls from the model to the appropriate data client,
maintains shared state across turns, and returns JSON-serializable results.

State tracked across the loop:
  memo_state      — sections written so far via store_finding()
  sources         — all web sources gathered, each tagged with search purpose
  _financial_data — prices fetched via get_financial_data(), injected as a
                    reminder in every store_finding() response so the model
                    uses real prices instead of training-data recollections
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from aria.tools import FinancialDataClient, WebSearchClient
from aria.tools.web_search import WebSearchResult

from .tool_schemas import MEMO_SECTIONS

_CONJUNCTION_RE = re.compile(r'\b(?:and|&|vs\.?|versus)\b', re.IGNORECASE)
_TICKER_RE = re.compile(r'\b[A-Z]{2,5}\b')
# A word that starts with a capital letter — proxy for a proper noun / named entity.
# Excludes sentence-start words by requiring at least one more word match on EACH side.
_CAP_WORD_RE = re.compile(r'\b[A-Z][A-Za-z0-9]+\b')


def _detect_multi_asset(query: str) -> bool:
    """
    Heuristic: true if the query appears to reference two or more named assets or sectors.
    Used to require a cross_asset search before synthesis.

    Triggers on:
      - Two or more distinct all-caps ticker tokens (e.g. "JPM SPY")
      - A conjunction where BOTH sides contain a capitalized word (named entity proxy),
        e.g. "JPMorgan and Treasury yields" — but NOT "current valuations and revised guidance"
        where neither side contains a proper noun.
    """
    if not query:
        return False
    tickers = _TICKER_RE.findall(query)
    if len(set(tickers)) >= 2:
        return True
    if _CONJUNCTION_RE.search(query):
        parts = _CONJUNCTION_RE.split(query, maxsplit=1)
        if len(parts) >= 2:
            left, right = parts[0].strip(), parts[1].strip()
            if left and right and _CAP_WORD_RE.search(left) and _CAP_WORD_RE.search(right):
                return True
    return False

# Maximum snippet length per web result injected into the model context.
# 700 chars balances grounding with context window budget:
# 6 searches × 6 results × 800 chars ≈ 7,200 tokens, leaving room for financial
# data and conversation history within a 9B model's ~16K context window.
_SNIPPET_CHARS = 800

_PERIOD_TO_DAYS: Dict[str, int] = {
    "1mo": 30,
    "3mo": 90,
    "1y": 365,
    "3y": 1095,
    "5y": 1825,
}

# Phrases that indicate the thesis section isn't committing to one direction.
# The conclusion section is for the final synthesis;
# these checks apply only to the research "thesis" section.
_THESIS_HEDGE_PHRASES = ["bull case", "base case", "bear case"]

# Direction vocabulary for the thesis/conclusion consistency gate.
_BULLISH_TERMS: frozenset[str] = frozenset({"undervalued", "outperform", "upgrade"})
_BEARISH_TERMS: frozenset[str] = frozenset({"overvalued", "underperform"})
_NEUTRAL_TERMS: frozenset[str] = frozenset({"fairly valued", "neutral", "market perform"})

# Words that, if repeated too many times, produce a style nudge in the store_finding response.
# Maps overused word → alternatives to suggest.
_OVERUSED_WORDS: Dict[str, str] = {
    "headwinds": "pressure / risk / challenge / drag / exposure",
}
_OVERUSE_THRESHOLD = 2  # nudge after this many occurrences across stored sections


def _recommendation_direction(text: str) -> str:
    """
    Extract the directional assessment from a text snippet.
    Returns 'bullish', 'bearish', 'neutral', or 'unclear'.
    Only 'unclear' suppresses the consistency gate — use it conservatively.
    """
    lower = text.lower()
    has_bullish = any(t in lower for t in _BULLISH_TERMS)
    has_bearish = any(t in lower for t in _BEARISH_TERMS)
    has_neutral = any(t in lower for t in _NEUTRAL_TERMS)
    count = sum([has_bullish, has_bearish, has_neutral])
    if count == 1:
        if has_bullish:
            return "bullish"
        if has_bearish:
            return "bearish"
        return "neutral"
    # Mixed signals — don't fire the gate
    return "unclear"

# Comparative markers required when a uniqueness claim is made.
_COMPARATIVE_MARKERS: tuple[str, ...] = (
    " unlike ", "compared to", " vs ", " vs.", "while other",
    "other banks", "competitors", "peers", "industry average",
)


@dataclass
class Source:
    """A web source tagged with the search purpose that surfaced it."""

    url: str
    title: str
    purpose: str  # background | pro_thesis | counter_thesis

    def to_dict(self) -> Dict[str, str]:
        return {"url": self.url, "title": self.title, "purpose": self.purpose}


class ToolExecutor:
    """
    Executes tool calls on behalf of the model and maintains shared loop state.

    Each instance lives for one agent run. The model accumulates findings
    into memo_state via store_finding(); the loop reads it to build the
    final ResearchMemo when finalize_memo() succeeds.
    """

    # Maximum web searches allowed per purpose. Enforced hard so the model
    # cannot burn all its steps in search mode and skip writing sections.
    # Must match the "MAX 2" cap stated in the system prompt.
    _MAX_SEARCHES_PER_PURPOSE = 2

    # Maximum financial data fetches per run. Enough to cover multiple tickers
    # (e.g. gold + S&P + DXY + BTC) without burning the step budget.
    _MAX_FINANCIAL_CALLS = 4

    def __init__(
        self,
        web_search: Optional[WebSearchClient],
        financial: Optional[FinancialDataClient],
        progress_callback: Optional[Callable[[str], None]] = None,
        query: str = "",
    ) -> None:
        self._web_search = web_search
        self._financial = financial
        self._progress = progress_callback or (lambda _: None)
        self.memo_state: Dict[str, str] = {}
        self.sources: List[Source] = []
        self._search_counts: Dict[str, int] = {}
        self._financial_data: Dict[str, Dict[str, Any]] = {}  # cache_key → price data
        self._return_series: Dict[str, np.ndarray] = {}  # ticker → daily returns (for cross-asset correlation)
        self._failed_tickers: set[str] = set()  # tickers that already returned errors
        self._financial_call_count: int = 0
        self._cross_asset_required: bool = _detect_multi_asset(query)
        self._cross_asset_searched: bool = False

    def execute(self, name: str, args: Dict[str, Any]) -> str:
        """Dispatch a tool call by name. Always returns a JSON string."""
        handlers: Dict[str, Callable[[Dict[str, Any]], str]] = {
            "web_search": self._web_search_tool,
            "get_financial_data": self._financial_tool,
            "store_finding": self._store_finding_tool,
            "finalize_memo": self._finalize_memo_tool,
        }
        handler = handlers.get(name)
        if handler is None:
            return _err(f"Unknown tool {name!r}.")
        return handler(args)

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _web_search_tool(self, args: Dict[str, Any]) -> str:
        query: str = (args.get("query") or "").strip()
        purpose: str = args.get("purpose") or "background"

        if not query:
            return _err("query must not be empty.")
        if self._web_search is None:
            return _err("Web search client is not configured.")

        if purpose == "cross_asset" and not self._cross_asset_required:
            return json.dumps({
                "error": (
                    "cross_asset search is not applicable to this query — "
                    "the query references only one asset or sector."
                ),
                "instruction": (
                    "Use purpose='pro_thesis' or purpose='counter_thesis' instead. "
                    "cross_asset is only for queries that explicitly name two or more assets."
                ),
            })

        count = self._search_counts.get(purpose, 0)
        if count >= self._MAX_SEARCHES_PER_PURPOSE:
            missing = [s for s in MEMO_SECTIONS if s not in self.memo_state]
            return json.dumps({
                "error": (
                    f"Search limit reached for purpose='{purpose}' "
                    f"({self._MAX_SEARCHES_PER_PURPOSE}/{self._MAX_SEARCHES_PER_PURPOSE} used). "
                    "You have enough evidence. Stop searching."
                ),
                "instruction": (
                    f"Call store_finding() now for each missing section: {missing}. "
                    "Then call finalize_memo()."
                ),
            })

        self._search_counts[purpose] = count + 1
        if purpose == "cross_asset":
            self._cross_asset_searched = True
        self._progress(f"  [{purpose}] Searching: {query}")
        results: List[WebSearchResult] = self._web_search.search(query, max_results=6)

        for r in results:
            self.sources.append(Source(url=r.url, title=r.title, purpose=purpose))

        if not results:
            return json.dumps({
                "purpose": purpose,
                "results": [],
                "note": "No results returned. Try a different query.",
            })

        return json.dumps({
            "purpose": purpose,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.content[:_SNIPPET_CHARS],
                }
                for r in results
            ],
        })

    def _financial_tool(self, args: Dict[str, Any]) -> str:
        ticker: str = (args.get("ticker") or "").upper().strip()
        period: str = args.get("period") or "1y"

        if not ticker:
            return _err("ticker must not be empty.")
        if self._financial is None:
            return _err("Financial data client is not configured.")

        # Return cached result without consuming a fetch slot.
        cache_key = f"{ticker}|{period}"
        if cache_key in self._financial_data:
            return json.dumps(self._financial_data[cache_key])

        if self._financial_call_count >= self._MAX_FINANCIAL_CALLS:
            return json.dumps({
                "error": f"Financial data limit reached ({self._MAX_FINANCIAL_CALLS} tickers fetched).",
                "fetched": list(self._financial_data.keys()),
                "instruction": "Use the prices already fetched. Do not call get_financial_data again.",
            })

        # Don't retry tickers that already failed — use an alternative symbol instead.
        if ticker in self._failed_tickers:
            return json.dumps({
                "error": f"'{ticker}' already failed earlier in this run.",
                "instruction": (
                    "Use a different ticker symbol. "
                    "Gold: GC=F or GLD. Silver: SI=F or SLV. "
                    "Oil: CL=F. DXY: DX-Y.NYB. S&P 500: ^GSPC or SPY."
                ),
            })

        days = _PERIOD_TO_DAYS.get(period, 365)
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=days)).isoformat()

        self._financial_call_count += 1
        self._progress(f"  [financial] Fetching {ticker} ({period}: {start} → {end})…")

        try:
            hist = self._financial.price_history(ticker, start=start, end=end)
        except Exception as exc:
            self._failed_tickers.add(ticker)
            return json.dumps({
                "error": f"Could not fetch data for {ticker!r}: {exc}",
                "instruction": (
                    f"'{ticker}' may be an incorrect or invented ticker. "
                    "Do NOT guess an alternative symbol from memory. "
                    "Use web_search to confirm the correct ticker for this company, "
                    "then call get_financial_data with the verified symbol."
                ),
            })

        if hist.empty:
            self._failed_tickers.add(ticker)
            return json.dumps({
                "error": f"No price data found for {ticker!r}.",
                "instruction": (
                    f"'{ticker}' may be an incorrect or invented ticker. "
                    "Do NOT guess an alternative symbol from memory. "
                    "Use web_search to confirm the correct ticker for this company, "
                    "then call get_financial_data with the verified symbol."
                ),
            })

        close = hist["Close"]
        prices = close.values.astype(float)
        n = len(prices)
        current = prices[-1]
        start_price = prices[0]
        period_return = round((current / start_price - 1) * 100, 2) if start_price else None
        period_high = round(float(hist["High"].max()), 2)
        period_low = round(float(hist["Low"].min()), 2)

        # Use dividend-adjusted close for volatility and correlation so that
        # ex-dividend price drops don't appear as fake negative return spikes.
        # Adj Close is present when auto_adjust=False; fall back to Close if absent.
        adj_col = "Adj Close" if "Adj Close" in hist.columns else "Close"
        adj_prices = hist[adj_col].values.astype(float)

        # Annualized volatility from adjusted daily returns (ddof=1 = sample std dev).
        daily_rets = np.diff(adj_prices) / adj_prices[:-1]
        vol = daily_std_pct = None
        if len(daily_rets) > 1:
            daily_std_pct = round(float(daily_rets.std(ddof=1) * 100), 4)
            vol = round(float(daily_std_pct / 100 * np.sqrt(252) * 100), 2)

        # Trend: linear regression slope as % of mean price per day.
        trend_direction = None
        if n >= 5:
            x = np.arange(n, dtype=float)
            coeffs = np.polyfit(x, prices, 1)
            slope = coeffs[0]
            trend_direction = "up" if slope > 0 else ("down" if slope < 0 else "flat")

        # Momentum: last 30 days vs prior 30 days on unadjusted close (price return).
        momentum_recent = momentum_prior = None
        if n >= 60:
            momentum_recent = round(float((prices[-1] / prices[-30] - 1) * 100), 2)
            momentum_prior = round(float((prices[-30] / prices[-60] - 1) * 100), 2)

        # Price percentile: where current close sits within the period's high-low range.
        price_range = period_high - period_low
        price_percentile = round((current - period_low) / price_range * 100, 1) if price_range > 0 else 50.0

        # Cross-asset return correlation vs every ticker already fetched this run.
        # Uses adjusted daily returns (same series as volatility) for consistency.
        # Aligned by tail — valid for same-exchange tickers; slight date skew possible
        # across different exchanges or futures calendars.
        correlations: Dict[str, float] = {}
        correlation_details: Dict[str, str] = {}
        for other_ticker, other_rets in self._return_series.items():
            min_len = min(len(daily_rets), len(other_rets))
            if min_len > 10:
                corr = float(np.corrcoef(daily_rets[-min_len:], other_rets[-min_len:])[0, 1])
                correlations[other_ticker] = round(corr, 3)
                correlation_details[other_ticker] = (
                    f"Pearson r={round(corr, 3)} over {min_len} overlapping trading days "
                    f"of adjusted daily returns"
                )
        self._return_series[ticker] = daily_rets

        data: Dict[str, Any] = {
            "ticker": ticker,
            "period": period,
            "period_start": start,
            "period_end": end,
            "days_analyzed": n,
            "period_start_price": round(float(start_price), 2),
            "current_price": round(current, 2),
            f"return_over_{period}_pct": period_return,  # key encodes period — cannot be mislabeled
            "period_high": period_high,
            "period_low": period_low,
            "price_percentile_in_range": price_percentile,
            "annualized_volatility_pct": vol,
            "volatility_calculation": (
                f"daily_std={daily_std_pct}% (sample, ddof=1, dividend-adjusted) "
                f"× √252 = {vol}% annualized — "
                f"meaning the stock has historically moved ≈{daily_std_pct}% per day"
            ) if vol is not None else None,
            "trend_direction": trend_direction,
            "momentum_last_30d_pct": momentum_recent,
            "momentum_prior_30d_pct": momentum_prior,
            "momentum_note": (
                "momentum_last_30d_pct and momentum_prior_30d_pct are 30-TRADING-DAY windows "
                "(≈ 6 calendar weeks each). They are NOT 3-month returns. "
                "Do NOT cite either as a '3-month' figure. "
                f"The only valid 3-month return for {ticker} is return_over_3mo_pct, "
                "and only if you fetched period=3mo."
            ),
        }
        if correlations:
            data["return_correlations"] = correlations
            data["return_correlation_details"] = correlation_details
            data["correlation_note"] = (
                "These are Pearson r coefficients of adjusted daily returns — NOT R². "
                "Range: +1.0 (lockstep) to 0.0 (uncorrelated) to -1.0 (inverse). "
                "Cite as 'r=X.XX' or 'correlation r=X.XX'. Never write R² for these values."
            )

        # Futures contracts: price is already per-unit (per troy oz for gold,
        # per barrel for oil, etc.). Do not divide by contract size.
        if "=F" in ticker:
            data["price_note"] = (
                f"{ticker} is a futures contract. current_price ({data['current_price']}) "
                "is the per-unit spot price — per troy oz for metals, per barrel for oil. "
                "Do NOT divide by contract size. Use this number directly in the memo."
            )

        # Persist for price grounding — injected into every subsequent store_finding response.
        self._financial_data[cache_key] = data
        return json.dumps(data)

    def _store_finding_tool(self, args: Dict[str, Any]) -> str:
        section: str = (args.get("section") or "").strip()
        content: str = (args.get("content") or "").strip()

        if not section:
            return _err("section is required.")

        # Enforce cross_asset search before any synthesis begins.
        if not self.memo_state and self._cross_asset_required and not self._cross_asset_searched:
            return json.dumps({
                "error": (
                    "Cross-asset search required before writing any section. "
                    "This query covers multiple assets or sectors."
                ),
                "instruction": (
                    "Call web_search(purpose='cross_asset') with a query that includes BOTH "
                    "terms together — e.g. the relationship, exposure, or interaction between them. "
                    "Then proceed with pro_thesis and counter_thesis searches."
                ),
            })

        if section not in MEMO_SECTIONS:
            return _err(f"Unknown section {section!r}. Valid: {MEMO_SECTIONS}.")
        if not content:
            return _err("content must not be empty.")

        # Block overwrites — each section is written once.
        if section in self.memo_state:
            missing = [s for s in MEMO_SECTIONS if s not in self.memo_state]
            return json.dumps({
                "error": f"Section '{section}' is already stored. Do not overwrite it.",
                "instruction": (
                    f"Move on. Write the remaining sections: {missing}. "
                    "Then call finalize_memo()."
                ),
                "missing": missing,
            })

        # Thesis quality gate: the thesis must commit to one directional claim.
        if section == "thesis":
            lower = content.lower()
            hedges = [p for p in _THESIS_HEDGE_PHRASES if p in lower]
            if hedges:
                return json.dumps({
                    "error": (
                        "Thesis must be a single directional claim. "
                        f"Remove hedging language: {hedges}."
                    ),
                    "instruction": (
                        "Rewrite as ONE testable statement: "
                        "'[Asset] will [direction] [level] by [date] because [mechanism].' "
                        "Bull/base/bear scenario framing belongs in the conclusion section."
                    ),
                })

        # Thesis citation gate: numeric claims must be grounded in searched sources.
        if section == "thesis":
            has_number = re.search(r'\d+(?:\.\d+)?%|\b\d{2,}\b', content)
            has_citation = re.search(r'\([A-Za-z][^)]+,\s*\d{4}\)', content)
            if has_number and not has_citation:
                return json.dumps({
                    "error": (
                        "Thesis contains numeric claims without inline citations. "
                        "Every factual data point must be grounded in a searched source."
                    ),
                    "instruction": (
                        "Add inline citations for every numeric claim: (Source Name, Year). "
                        "Only cite sources returned by web_search in this session. "
                        "If you cannot cite a number, remove it from the thesis — "
                        "the thesis is a directional claim, not a data summary."
                    ),
                })

        # Placeholder citation gate: block template text used literally instead of real citations.
        # Applies to all sections that require inline citations.
        if section in ("thesis", "supporting_evidence", "counter_evidence", "failure_conditions"):
            if re.search(r'\(Source Name,?\s*Year\)', content, re.IGNORECASE):
                return json.dumps({
                    "error": (
                        f"Section '{section}' contains a placeholder citation '(Source Name, Year)'. "
                        "You must cite actual sources from your web_search results."
                    ),
                    "instruction": (
                        "Replace every '(Source Name, Year)' with a real citation from your search results — "
                        "e.g. '(Bloomberg, 2026)' or '(Seeking Alpha, 2026)'. "
                        "If you have no source for a claim, remove the claim."
                    ),
                })

        # Uniqueness gate: "unique" requires a named competitor comparison.
        # Applies to thesis and supporting_evidence — these are the sections where
        # an unverified uniqueness claim can silently become load-bearing.
        if section in ("thesis", "supporting_evidence"):
            lower = content.lower()
            if "unique" in lower or "uniquely" in lower:
                if not any(m in lower for m in _COMPARATIVE_MARKERS):
                    return json.dumps({
                        "error": (
                            "Uniqueness claim requires comparative verification. "
                            "'unique' or 'uniquely' appears without naming a specific competitor."
                        ),
                        "instruction": (
                            "Either: (1) name a specific competitor and cite evidence that they "
                            "lack this capability — e.g. 'unlike Bank of America, which...' — "
                            "or (2) remove the uniqueness claim and state only what the source "
                            "confirms about this entity, without asserting it is exclusive to them."
                        ),
                    })

        # Conclusion quality gate: must make ONE clear assessment, no scenario matrix.
        if section == "conclusion":
            lower = content.lower()
            hedges = [p for p in _THESIS_HEDGE_PHRASES if p in lower]
            if hedges:
                return json.dumps({
                    "error": (
                        f"conclusion must make one clear assessment — no bull/base/bear framing. "
                        f"Remove: {hedges}."
                    ),
                    "instruction": (
                        "Write ONE valuation assessment (overvalued/undervalued/fairly valued): "
                        "time horizon, key catalyst, "
                        "and the single event that would change this view. "
                        "Do not describe multiple scenarios."
                    ),
                })

        # Confidence propagation gate: Medium/Low confidence must constrain the conclusion.
        if section == "conclusion":
            stored_confidence = self.memo_state.get("confidence", "").lower()
            if stored_confidence.startswith(("medium", "low")):
                _HEDGE_MARKERS = (
                    "if confirmed", "pending", "conditional", "subject to",
                    "assuming", "unverified", "not yet confirmed", "remains unclear",
                    "only if", "provided that",
                )
                lower = content.lower()
                if not any(m in lower for m in _HEDGE_MARKERS):
                    conf_level = "Medium" if stored_confidence.startswith("medium") else "Low"
                    return json.dumps({
                        "error": (
                            f"Confidence is {conf_level} but conclusion reads at full conviction. "
                            "The uncertainty must propagate into the conclusion."
                        ),
                        "instruction": (
                            f"Revise conclusion to reflect {conf_level} confidence. "
                            "Name the specific unverified assumption that drives the uncertainty "
                            "and make the recommendation conditional on it — e.g. "
                            "'appears undervalued only if [key claim] is confirmed by [verifiable event]'."
                        ),
                    })

        # Direction-consistency gate: conclusion must match the thesis direction.
        # Fires AFTER confidence propagation so the model gets one targeted error at a time.
        if section == "conclusion":
            thesis_text = self.memo_state.get("thesis", "")
            thesis_dir = _recommendation_direction(thesis_text)
            conclusion_dir = _recommendation_direction(content)
            if (
                thesis_dir != "unclear"
                and conclusion_dir != "unclear"
                and thesis_dir != conclusion_dir
            ):
                return json.dumps({
                    "error": (
                        f"Direction mismatch: the thesis recommends '{thesis_dir}' "
                        f"but the conclusion recommends '{conclusion_dir}'. "
                        "The Conclusion must be consistent with and flow from the Thesis."
                    ),
                    "instruction": (
                        f"Rewrite the conclusion with a '{thesis_dir}' valuation assessment "
                        "(e.g. 'appears undervalued' if the thesis is bullish, 'appears overvalued' if bearish). "
                        "State the time horizon, the primary catalyst, "
                        "and the single event that would change this view."
                    ),
                })

        self.memo_state[section] = content
        filled = list(self.memo_state.keys())
        missing = [s for s in MEMO_SECTIONS if s not in self.memo_state]
        n_filled = len(filled)
        n_total = len(MEMO_SECTIONS)

        self._progress(f"  [memo] Stored '{section}' ({n_filled}/{n_total} sections filled)")

        response: Dict[str, Any] = {
            "stored": section,
            "filled": filled,
            "missing": missing,
            "ready_to_finalize": len(missing) == 0,
        }

        # Overused-word nudge: count occurrences across all stored sections and nudge
        # the model to vary its language if a word appears too frequently.
        all_text = " ".join(self.memo_state.values()).lower()
        style_notes = []
        for word, alternatives in _OVERUSED_WORDS.items():
            count = all_text.count(word)
            if count > _OVERUSE_THRESHOLD:
                style_notes.append(
                    f"'{word}' has been used {count} times across the memo. "
                    f"Vary your language in remaining sections — alternatives: {alternatives}."
                )
        if style_notes:
            response["style_note"] = " | ".join(style_notes)

        # Price grounding: remind the model of real fetched prices after every write.
        # Prevents the model from drifting back to training-data price recollections.
        if self._financial_data:
            parts = []
            for _, d in self._financial_data.items():
                return_key = next((k for k in d if k.startswith("return_over_")), None)
                return_str = f", {return_key}={d[return_key]}%" if return_key else ""
                parts.append(
                    f"{d['ticker']}: start_price={d.get('period_start_price','?')}, "
                    f"current_price={d['current_price']}"
                    f"{return_str} (period={d['period']}, {d.get('period_start','?')} to {d.get('period_end','?')})"
                )
            response["price_context"] = (
                "VERIFIED PRICES FROM THIS SESSION — cite only these, no training-data estimates: "
                + "; ".join(parts)
            )

        return json.dumps(response)

    def _finalize_memo_tool(self, args: Optional[Dict[str, Any]] = None) -> str:
        missing = [s for s in MEMO_SECTIONS if s not in self.memo_state]
        if missing:
            return json.dumps({
                "error": "Memo is incomplete — do not finalize yet.",
                "missing_sections": missing,
                "instruction": (
                    f"Call store_finding for each missing section: {', '.join(missing)}."
                ),
            })

        # Source validation: flag domain names cited in key sections that don't
        # appear in any source URL. Catches invented citations without blocking
        # legitimate references that use publication names rather than domains.
        warnings = _validate_source_domains(self.memo_state, self.sources)

        self._progress(f"  [memo] All {len(MEMO_SECTIONS)} sections complete — finalizing.")
        result: Dict[str, Any] = {
            "status": "complete",
            "sections": list(self.memo_state.keys()),
        }
        if warnings:
            result["source_warnings"] = warnings
        return json.dumps(result)


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _validate_source_domains(
    memo_state: Dict[str, str],
    sources: List[Source],
) -> List[str]:
    """
    Check whether domain names cited in supporting_evidence and
    counter_evidence appear in the searched source URLs.

    Returns a list of warning strings (empty if everything checks out).
    Only flags explicit domain patterns (word.tld) — publication names
    without a TLD are not checked to avoid false positives.
    """
    # Build root domains from searched sources: e.g. "en-int.capital.com" → "capital.com"
    source_root_domains: set[str] = set()
    for s in sources:
        m = re.search(r"https?://(?:www\.)?([^/\s]+)", s.url)
        if m:
            parts = m.group(1).lower().split(".")
            if len(parts) >= 2:
                source_root_domains.add(f"{parts[-2]}.{parts[-1]}")

    domain_pattern = re.compile(
        r"\b([a-zA-Z][a-zA-Z0-9\-]+\.(?:com|org|net|gov|edu|io|co))\b"
    )

    warnings: List[str] = []
    for section_name in ("supporting_evidence", "counter_evidence"):
        text = memo_state.get(section_name, "")
        for match in domain_pattern.finditer(text):
            domain = match.group(1).lower()
            parts = domain.split(".")
            root = f"{parts[-2]}.{parts[-1]}"
            if root not in source_root_domains:
                warnings.append(
                    f"{section_name}: '{domain}' cited but not found in searched sources"
                )

    return warnings


def _err(message: str) -> str:
    """Return a JSON error string in a consistent shape."""
    return json.dumps({"error": message})
