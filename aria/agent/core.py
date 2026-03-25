"""
ARIA agentic research loop.

The model drives tool selection across multiple turns — calling web_search,
get_financial_data, and store_finding until the memo is complete, then signaling
done with finalize_memo. The loop renders the accumulated memo_state into a
structured ResearchMemo on success.

Checkpoint mode runs a fast single-shot thesis proposal first (no tool loop).
The CLI presents proposed theses for user selection, then calls run() again in
autonomous mode with the selected thesis as the query.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from aria.config import AriaConfig
from aria.models import ModelRouter, TaskType
from aria.outputs.formatter import ResearchMemo, format_memo_markdown
from aria.storage.db import ResearchDatabase
from aria.tools import DocumentIngestor, FinancialDataClient, RepoReader, WebSearchClient

from .tool_executor import ToolExecutor
from .tool_schemas import FINANCE_TOOLS, MEMO_SECTIONS

logger = logging.getLogger(__name__)


class AgentMode(Enum):
    AUTONOMOUS = "autonomous"
    CHECKPOINT = "checkpoint"


class OutputFormat(Enum):
    AUTO = "auto"
    MEMO = "memo"
    DIRECT = "direct"
    MATRIX = "matrix"


@dataclass
class AgentResult:
    content: str
    metadata: Dict[str, Any]


@dataclass
class AgentDependencies:
    config: AriaConfig
    router: ModelRouter
    web_search: Optional[WebSearchClient]
    financial: Optional[FinancialDataClient]
    documents: Optional[DocumentIngestor]
    repo: Optional[RepoReader]


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

# Tool workflow prompt for the autonomous agentic loop.
# For local models this is the entire system prompt (preserve context window).
# For API models the cognitive model from mental_model.md is prepended.
# The current date is injected at call time — see _build_analysis_system_prompt.
_ANALYSIS_SYSTEM_PROMPT = """\
You are ARIA — a research agent. Use the tools available to investigate \
the given claim, then write a structured research memo.

WORKFLOW — follow this order exactly:
1. web_search(purpose="background") — MAX 2 searches. Orient yourself.
   Start writing store_finding() calls as soon as you have evidence — do NOT wait until all searches are done.
2. get_financial_data() — fetch EVERY ticker relevant to the query (up to 4 calls). \
IMPORTANT: If the query names a company rather than an obvious ticker symbol, \
use web_search(purpose="background") to confirm the correct ticker BEFORE calling \
get_financial_data. Company names do not reliably map to tickers from memory — \
e.g. "Vail Resorts" is MTN, not VRE. Fetching the wrong ticker silently corrupts \
the entire memo. When in doubt, search first. \
Do NOT cite S&P, BTC, DXY, or any other asset from training-data memory — fetch it. \
The response includes quantitative analysis (trend, volatility, momentum, price percentile) — \
use these real metrics in supporting_evidence and baselines. \
Only skip if the query mentions no assets at all.
2b. web_search(purpose="cross_asset") — REQUIRED if the query names two or more assets, \
companies, or sectors. Search for the RELATIONSHIP or INTERACTION between them \
(e.g. 'JPMorgan private credit exposure contagion risk', not just each term alone). \
This step is MANDATORY before any store_finding call. If you skip it, store_finding will block you.
3. web_search(purpose="pro_thesis") — MAX 2 searches. Find supporting evidence.
4. web_search(purpose="counter_thesis") — MAX 2 searches. Find invalidating evidence.
5. store_finding() — call ONCE PER SECTION. Write all 8 sections now.
6. finalize_memo() — call after all 8 sections are stored.

HARD LIMIT: You may perform at most 3 web_search calls per purpose. \
After that the tool will refuse and tell you to start writing. \
Do not wait until all searches are done — start calling store_finding() \
as soon as you have enough evidence for a section.

MEMO SECTIONS — write each with store_finding before calling finalize_memo:
- thesis: Specific, directional, testable claim written in plain English that a smart \
non-specialist can read in one pass. No unexplained jargon in the headline sentences — \
terms like "margin floor protection", "demand inelasticity", "asymmetric downside protection", \
or "structural moat" must either be cut or immediately followed by a plain-English explanation \
of what they mean. If you include any numeric data point (%, price, growth rate), you MUST \
add an inline citation: (Source Name, Year). \
Do not cite a number you cannot source — state the direction and mechanism instead.
- supporting_evidence: 2–4 sentences citing real sources by name. Lead with the strongest \
single data point, and for each piece of evidence explain WHY it supports the thesis — \
not just that it exists.
- counter_evidence: The strongest counter-argument. Name the mechanism that would make the \
thesis wrong, and explain WHY that mechanism is a real threat rather than just citing it.
- segment_behavior: How structurally distinct subgroups behave differently under this thesis. \
Name the key structural difference and explain WHY it matters — what does it tell us about \
the thesis that a single aggregate view would miss?
- failure_conditions: Named, specific, monitorable events that would invalidate the thesis. \
For each condition, explain WHY crossing that threshold changes the picture. \
Vague risks are not failure conditions.
- confidence: High / Medium / Low — one sentence anchored to source quality and \
convergence across tiers.
- baselines: Only benchmarks from get_financial_data results. \
No yields, rates, or prices from memory. Add one-sentence context per benchmark.
- conclusion: The synthesis — pull the analysis together into a plain-English paragraph \
that explains WHY the evidence as a whole points where it does. What does the combination of \
supporting evidence, counter-evidence, and market pricing actually mean? Why does one side \
outweigh the other? Only after answering those questions: ONE valuation assessment \
(overvalued/undervalued/fairly valued) relative to a named benchmark, that MUST match the \
thesis above — if the thesis is bullish, the conclusion must say undervalued, not overvalued. \
Specific time horizon, the primary catalyst, and the single event that would change this view. \
NOT a new thesis and NOT a restatement of the thesis. Do NOT use bull/base/bear framing. \
Do NOT use buy/sell/hold or overweight/underweight language — describe what the evidence \
implies about valuation, not what anyone should do.

SEARCH STRATEGY — query construction matters:
- NEVER include specific price targets or numeric thresholds from the question in search \
queries. Searching "$2,500 gold" surfaces forecast articles that echo the same number back \
— pure confirmation bias.
- Search for MECHANISMS and DRIVERS, not price levels. \
E.g. instead of "will gold break $2,500", search "gold mining supply constraints 2026" or \
"central bank gold buying Q1 2026" or "real yield gold price correlation".
- Diversify query types across searches. Do not search multiple price-forecast aggregator \
sites — they recycle the same analyst quotes. Mix sources: \
  background — macro regime, recent events, policy decisions, geopolitical context; \
  pro_thesis — evidence for the specific driver (supply/demand data, institutional flows, \
  technical momentum, policy tailwinds); \
  counter_thesis — evidence against the driver (competing safe-havens, demand weakness, \
  policy headwinds, structural constraints).

RULES:
- Clarity: Write for a smart non-specialist. Every fact, data point, or mechanism you cite must \
do two things: (1) state what it is in plain English, and (2) state explicitly why it matters — \
what does it imply for the overall picture? Never leave the significance implicit. Do not just \
report that something exists or happened; explain what it means and why the reader should care. \
Make every causal link explicit: "X happened, which means Y, which matters because Z." \
Jargon is only acceptable if immediately followed by a plain-English translation in the same \
sentence (e.g. "EBITDA — the company's operating profit before accounting adjustments").
- Do not invent sources. Only cite results returned by web_search.
- Every factual claim in thesis, supporting_evidence, and counter_evidence must end with an \
inline citation: (Source Name, Year). E.g. "(Kitco, 2026)" or "(Bloomberg, 2026)". \
In the thesis, if you cannot cite a number, remove it — state direction and mechanism only.
- Every claim must reference a baseline.
- Weight evidence by source tier: primary data > peer-reviewed > institutional \
> journalism > commentary.
- Be direct and dense. One number that proves the thesis beats ten that support it.
- Cite both legs of any contrastive claim. Any sentence linking a negative indicator to \
a positive conclusion with "but", "but also", "while", or "however" must end with a \
citation supporting the positive leg. If you cannot cite it, state the negative plainly \
and omit the positive hedge — do not launder unsupported conclusions through \
balanced-sounding language.
- Load-bearing claims require quoted anchors. If a claim is the primary reason the thesis \
holds — removing it would invalidate the thesis — you must quote a specific phrase from \
the source verbatim in quotation marks before the citation. \
E.g. '"central bank purchases increased 68% in Q4" (Bloomberg, 2026)'. \
A source name alone is not sufficient for a load-bearing claim.
- Uniqueness claims require named comparisons. If you assert that an entity has a \
"unique" capability or advantage, you must name a specific competitor and cite evidence \
that they lack it. Confirming the capability exists is not the same as confirming it is \
exclusive. If you cannot cite the differential, remove the uniqueness framing.

DATA INTEGRITY:
- Every dollar price you cite for any asset MUST be a value from get_financial_data in this \
session (current_price, period_start_price, period_high, or period_low). \
Do NOT cite any price from training memory or web search snippets. \
If you have not fetched an asset's price this session, do not quote its price.
- return_correlations values from get_financial_data are Pearson r (range −1 to +1). \
They are NOT R². Never write "R²" or "R-squared" — write "r" or "correlation coefficient".
- A single number can only appear once in the memo. If gold returned 68.7% over 1y, \
cite 68.7% in every section — do not introduce a different figure elsewhere.
- NEVER cite a return percentage for a period you did not fetch. \
If you fetched GC=F with period=1y, you have only the 1y return. \
Do not write "approximately" or estimate returns for unfetched periods. \
If you need the 3-month return, call get_financial_data again with period=3mo.
- momentum_last_30d_pct is NOT a 3-month return. It covers 30 trading days (≈ 6 calendar \
weeks). Never describe it as "3-month performance", "recent quarter", or any quarterly framing. \
If you cite momentum, label it explicitly as "30-trading-day momentum".
"""

# Prompt for the checkpoint clarification phase.
# Single-shot — no tool loop. Sharpens the user's vague question into ONE precise,
# testable research question before the full agentic analysis runs.
# Current date is injected at call time.
_CLARIFICATION_SYSTEM_PROMPT = """\
You are ARIA. Confirm the user's research question is ready to investigate, or minimally reframe it if it is too vague.

Output format (use exactly):

**Research question:** [the user's question, lightly edited for clarity if needed]

**Focus:** [one sentence — which mechanism or driver will be investigated]

RULES:
- If the question is already clear (has a subject, a direction, and a mechanism), repeat it almost verbatim.
- Do NOT add price targets, dates, or numeric thresholds that the user did not provide.
- Do NOT editorialize or reframe a clear question into something more "precise" — preserve intent.
- Under 80 words total.
"""


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------

class AgentLoop:
    """
    Coordinates model routing, tool execution, and memo assembly for ARIA.

    Autonomous mode  — the model drives a tool-calling loop until finalize_memo
                       succeeds, then the accumulated memo_state is rendered to
                       a structured ResearchMemo and saved to the database.

    Checkpoint mode  — a fast single-shot thesis proposal runs first. The CLI
                       displays the proposals, the user selects which to pursue,
                       then run() is called again in autonomous mode with the
                       selected thesis as the query.

    Fallback         — if the model stops calling tools before finalizing (common
                       with smaller local models that don't support function
                       calling reliably), the loop returns whatever content the
                       model produced, with fallback_used=True in metadata.
                       Fallback runs are not saved to the database.
    """

    def __init__(self, deps: AgentDependencies) -> None:
        self._deps = deps

    @classmethod
    def from_config(cls, config: AriaConfig) -> "AgentLoop":
        router = ModelRouter(config)
        return cls(AgentDependencies(
            config=config,
            router=router,
            web_search=WebSearchClient(config),
            financial=FinancialDataClient(config),
            documents=DocumentIngestor(config),
            repo=RepoReader(config),
        ))

    def run(
        self,
        query: str,
        mode: Optional[AgentMode] = None,
        output_format: OutputFormat = OutputFormat.AUTO,
        search_query: Optional[str] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> AgentResult:
        """
        Execute an ARIA research pass.

        In checkpoint mode the proposal phase runs here; the CLI handles user
        selection and calls run() a second time in autonomous mode for analysis.

        stream_callback is used in the proposal phase (single-shot, streamable)
        and in the autonomous fallback path (model produced text, not tool calls).
        During the normal agentic tool loop, progress is printed directly to stdout
        so the user can see what the agent is doing in real time.
        """
        mode = mode or AgentMode(self._deps.config.mode.default)
        if mode is AgentMode.CHECKPOINT:
            return self._run_proposal(query, stream_callback)
        return self._run_agentic(query, stream_callback)

    # ------------------------------------------------------------------
    # Checkpoint: fast thesis proposal (single-shot)
    # ------------------------------------------------------------------

    def _run_proposal(
        self,
        query: str,
        stream_callback: Optional[Callable[[str], None]],
    ) -> AgentResult:
        """
        Quick background search + single-shot thesis proposal.

        Not agentic — the model does not call tools here. We run the search
        ourselves to give the model current context without consuming the
        user's time in a full tool loop. The proposal phase should be fast.
        """
        routed = self._deps.router.select_model(TaskType.DIRECT_ANSWER)

        system_content = f"Today's date: {_today()}.\n\n{_CLARIFICATION_SYSTEM_PROMPT}"
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=query),
        ]
        # Always collect silently then emit the stripped result.
        # Reasoning models emit <think> blocks that span many chunks; stripping
        # mid-stream is fragile, and the clarification phase is short enough that
        # the latency difference is imperceptible.
        content = _invoke(routed.model, messages, stream_callback=None)
        if stream_callback is not None:
            stream_callback(content)

        return AgentResult(
            content=content,
            metadata={
                "model_name": routed.name,
                "is_local": routed.is_local,
                "mode": "checkpoint_proposal",
                "session_id": None,
                "web_search_sources": [],
            },
        )

    # ------------------------------------------------------------------
    # Autonomous: agentic tool loop
    # ------------------------------------------------------------------

    def _run_agentic(
        self,
        query: str,
        stream_callback: Optional[Callable[[str], None]],
    ) -> AgentResult:
        """
        Full agentic loop. The model calls tools until finalize_memo succeeds.

        On success  — memo_state is rendered via format_memo_markdown and the
                      session is saved to the SQLite database.
        On fallback — the model's last text response is returned as-is, with
                      fallback_used=True. Fallback runs are not saved to the DB
                      because an incomplete memo provides no outcome to track.
        """
        session_id = str(uuid.uuid4())
        routed = self._deps.router.select_model(TaskType.RESEARCH_MEMO)
        max_steps = self._deps.config.agent.max_steps

        executor = ToolExecutor(
            web_search=self._deps.web_search,
            financial=self._deps.financial,
            progress_callback=lambda msg: print(msg, flush=True),
            query=query,
        )

        system_content = _build_analysis_system_prompt()
        # bind_tools() attaches the tool schemas to every invocation.
        # Works against both cloud (Anthropic/OpenAI) and local (LM Studio/Ollama)
        # endpoints via the OpenAI-compatible function-calling spec.
        model_with_tools = routed.model.bind_tools(FINANCE_TOOLS)

        messages: List[Any] = [
            SystemMessage(content=system_content),
            HumanMessage(content=query),
        ]

        steps_taken = 0
        fallback_used = False
        content = ""
        consecutive_text_steps = 0

        for step in range(max_steps):
            steps_taken += 1
            response: AIMessage = model_with_tools.invoke(messages)
            messages.append(response)

            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls:
                raw = _strip_thinking(getattr(response, "content", "") or "")
                consecutive_text_steps += 1

                if consecutive_text_steps >= 3:
                    # Three consecutive steps with no tool calls — genuine stall.
                    fallback_used = True
                    content = raw
                    break

                # Qwen and other local models often emit planning text between tool
                # calls without <think> tags. Inject a context-aware nudge and
                # continue rather than immediately falling back.
                missing = [s for s in MEMO_SECTIONS if s not in executor.memo_state]
                if missing:
                    nudge = (
                        "Stop generating text. Call the next tool immediately. "
                        f"Missing sections: {missing}. "
                        "Call store_finding() for each, then finalize_memo()."
                    )
                else:
                    nudge = "All sections complete. Call finalize_memo() now."
                messages.append(HumanMessage(content=nudge))
                continue

            # A tool call resets the consecutive text counter.
            consecutive_text_steps = 0

            finalized = False
            for call in tool_calls:
                name: str = call["name"]
                args: Dict[str, Any] = call.get("args", {})
                call_id: str = call.get("id", f"call_{step}_{name}")

                result_str = executor.execute(name, args)
                messages.append(
                    ToolMessage(content=result_str, tool_call_id=call_id)
                )

                # finalize_memo returns {"status": "complete"} on success.
                # On incomplete memo it returns {"error": ...} — the model must
                # fill missing sections and call finalize_memo again.
                if name == "finalize_memo":
                    result = json.loads(result_str)
                    if "error" not in result:
                        finalized = True

            if finalized:
                content = _render_memo(
                    executor.memo_state,
                    partial=False,
                    query=query,
                    session_id=session_id,
                    model_name=routed.name,
                    sources=[s.to_dict() for s in executor.sources],
                )
                break

        else:
            # Exhausted max_steps without a successful finalize_memo call.
            fallback_used = True
            last_ai = next(
                (m for m in reversed(messages) if isinstance(m, AIMessage)), None
            )
            content = _strip_thinking((getattr(last_ai, "content", "") or "") if last_ai else "")
            # If the model made partial progress, render what we have.
            if not content and executor.memo_state:
                content = _render_memo(
                    executor.memo_state,
                    partial=True,
                    query=query,
                    model_name=routed.name,
                    sources=[s.to_dict() for s in executor.sources],
                )

        metadata: Dict[str, Any] = {
            "session_id": session_id if not fallback_used else None,
            "model_name": routed.name,
            "is_local": routed.is_local,
            "mode": "autonomous",
            "steps_taken": steps_taken,
            "fallback_used": fallback_used,
            "web_search_sources": [s.to_dict() for s in executor.sources],
        }

        # Persist successful runs. DB failures never surface to the caller.
        if not fallback_used:
            try:
                db = ResearchDatabase(Path(self._deps.config.agent.db_path))
                db.save_session(session_id, query, executor.memo_state, metadata)
            except Exception as exc:
                logger.warning("Failed to save session %s: %s", session_id, exc)

        return AgentResult(content=content, metadata=metadata)


# ---------------------------------------------------------------------------
# Module-level helpers (not part of the public API)
# ---------------------------------------------------------------------------

def _today() -> str:
    """Return the current date in a format LLMs parse reliably."""
    today = date.today()
    return f"{today.strftime('%B %d, %Y')} ({today.isoformat()})"


def _build_analysis_system_prompt() -> str:
    """
    Builds the system prompt for the agentic analysis loop.
    The current date is always injected at the top so the model can reason
    about recency without guessing.
    """
    date_line = f"Today's date: {_today()}.\n\n"
    return date_line + _ANALYSIS_SYSTEM_PROMPT



def _invoke(
    model: Any,
    messages: List[Any],
    stream_callback: Optional[Callable[[str], None]],
) -> str:
    """Invoke a model with optional token-level streaming. Returns the full content."""
    if stream_callback is not None:
        parts: List[str] = []
        in_think = False
        for chunk in model.stream(messages):
            part = getattr(chunk, "content", "") or ""
            if not isinstance(part, str) or not part:
                continue
            # Buffer and suppress <think>…</think> reasoning blocks in real time.
            parts.append(part)
            combined = "".join(parts)
            if "<think>" in combined and not in_think:
                in_think = True
            if in_think and "</think>" in combined:
                in_think = False
                # Print only what comes after the closing tag.
                after = combined[combined.rfind("</think>") + len("</think>"):]
                if after:
                    stream_callback(after)
            elif not in_think and not combined.endswith(("<think>",)):
                stream_callback(part)
        return _strip_thinking("".join(parts))

    response = model.invoke(messages)
    return _strip_thinking(getattr(response, "content", str(response)))


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> reasoning blocks that some local models emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _render_memo(
    memo_state: Dict[str, str],
    partial: bool,
    query: str = "",
    session_id: Optional[str] = None,
    model_name: str = "",
    sources: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Render accumulated store_finding() calls into a formatted ResearchMemo."""
    placeholder = "_Not recorded._"
    memo = ResearchMemo(
        thesis=memo_state.get("thesis", placeholder),
        supporting_evidence=memo_state.get("supporting_evidence", placeholder),
        counter_evidence=memo_state.get("counter_evidence", placeholder),
        segment_behavior=memo_state.get("segment_behavior", placeholder),
        failure_conditions=memo_state.get("failure_conditions", placeholder),
        confidence=memo_state.get("confidence", placeholder),
        baselines=memo_state.get("baselines", placeholder),
        conclusion=memo_state.get("conclusion", placeholder),
        sources=sources or [],
        partial=partial,
    )
    return format_memo_markdown(
        memo,
        query=query,
        session_id=session_id,
        model_name=model_name,
    )
