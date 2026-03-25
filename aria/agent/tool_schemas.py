"""
Tool schemas for the ARIA agentic research loop.

Each schema is a plain dict matching the OpenAI function-calling spec.
LangChain's ChatOpenAI.bind_tools() accepts this format and it works against
both cloud API endpoints (Anthropic, OpenAI) and local servers (LM Studio, Ollama).

To add a domain beyond finance, define a new tool list here and pass it to
AgentLoop._run_agentic(). Each domain controls which tools the model can call
and therefore what data sources it can access.
"""
from __future__ import annotations

from typing import Any, Dict, List

# The 8 sections that constitute a complete ARIA research memo.
# store_finding() writes them one at a time; finalize_memo() validates all are present.
MEMO_SECTIONS: List[str] = [
    "thesis",
    "supporting_evidence",
    "counter_evidence",
    "segment_behavior",
    "failure_conditions",
    "confidence",
    "baselines",
    "conclusion",
]

# Per-section guidance injected into the store_finding description.
# Keeps the model's section content consistent without a separate prompt.
_SECTION_GUIDANCE: Dict[str, str] = {
    "thesis": (
        "A causal claim: '[Evidence/driver] points to [outcome] because [mechanism].' "
        "1–2 sentences. Direct and declarative — not a scenario summary. "
        "State the direction of the claim (e.g. appears undervalued, overvalued, or fairly valued) "
        "and the mechanism driving it. "
        "E.g. 'Persistent central bank gold accumulation points to sustained price support "
        "because institutional demand is price-inelastic and absorbs ETF outflows — "
        "gold appears undervalued relative to equities on a risk-adjusted basis.'"
    ),
    "supporting_evidence": (
        "2–4 sentences citing specific sources by name. "
        "Lead with the single strongest data point. "
        "End every factual claim with an inline citation: (Source Name, Year). "
        "E.g. 'Central bank buying reached $17B/yr (Bloomberg, 2026).' "
        "CONTRASTIVE CLAIMS: any sentence linking a negative indicator to a positive conclusion "
        "with 'but', 'but also', 'while', or 'however' must end with a citation supporting "
        "the positive leg — if you cannot cite it, state the negative plainly without the hedge."
    ),
    "counter_evidence": (
        "The strongest counter-argument to the thesis. "
        "Name the mechanism that would make the thesis wrong. "
        "End every factual claim with an inline citation: (Source Name, Year)."
    ),
    "segment_behavior": (
        "How structurally distinct subgroups (e.g. large/small cap, domestic/international) "
        "behave differently under this thesis. Name the key structural difference."
    ),
    "failure_conditions": (
        "Named, specific, monitorable events that would invalidate the thesis. Be precise — "
        "vague risks are not failure conditions."
    ),
    "confidence": (
        "High / Medium / Low. One sentence of justification anchored to source quality "
        "and convergence across tiers."
    ),
    "baselines": (
        "Explicit comparison benchmarks from get_financial_data calls only — do not include "
        "interest rates, yields, or any price from memory. "
        "Only cite the return for the exact period you fetched (e.g. if you fetched period=1y, "
        "cite the 1y return — never estimate a 3mo return from 1y data). "
        "Add a one-sentence interpretation of each benchmark relative to the thesis. "
        "E.g. 'S&P 500 +16.9% over 1y (^GSPC, fetched 2025-03-15 to 2026-03-15) — "
        "risk-on appetite remains elevated despite war premium.'"
    ),
    "conclusion": (
        "A brief conclusory paragraph (3–5 sentences). Synthesize the evidence into a valuation "
        "assessment (overvalued / undervalued / fairly valued) relative to a named benchmark, "
        "with a time horizon, the key catalyst that supports this view, and the single event "
        "that would change this view. No bull/bear/base case framing. "
        "Do not use buy/sell/hold or overweight/underweight language."
    ),
}

# ---------------------------------------------------------------------------
# Individual tool schemas
# ---------------------------------------------------------------------------

WEB_SEARCH: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for evidence. "
            "Call this at least twice per analysis — once with purpose='pro_thesis' "
            "and once with purpose='counter_thesis'. "
            "Use purpose='background' for initial orientation before forming a thesis."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused, specific search query.",
                },
                "purpose": {
                    "type": "string",
                    "enum": ["background", "cross_asset", "pro_thesis", "counter_thesis"],
                    "description": (
                        "Why you are searching. This label is recorded in the output and "
                        "shown to the user: "
                        "'background' — initial orientation before forming a thesis; "
                        "'cross_asset' — REQUIRED when the query names two or more assets or sectors: "
                        "search for the relationship, interaction, or exposure between them "
                        "(e.g. 'JPMorgan private credit exposure risk 2026'). "
                        "Must be called before any store_finding; "
                        "'pro_thesis' — build supporting evidence; "
                        "'counter_thesis' — find invalidating evidence."
                    ),
                },
            },
            "required": ["query", "purpose"],
        },
    },
}

GET_FINANCIAL_DATA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_financial_data",
        "description": (
            "Fetch historical price data and quantitative analysis for a ticker symbol. "
            "Use this when the query mentions a stock, ETF, commodity, or index. "
            "Returns: current price, period return, high/low, price percentile in range, "
            "annualized volatility, linear trend direction (slope as % per day), 30-day momentum, "
            "and cross-asset return_correlations (Pearson r, range -1 to +1). "
            "IMPORTANT: return_correlations values are Pearson r — NOT R². Do not label them as R². "
            "Use these metrics in supporting_evidence and baselines — not just the spot price. "
            "If a ticker returns an error, do NOT retry it — use an alternative. "
            "Common alternatives: gold → GC=F (futures) or GLD (ETF); "
            "silver → SI=F or SLV; oil → CL=F; DXY → DX-Y.NYB; "
            "S&P 500 → ^GSPC or SPY; Bitcoin → BTC-USD."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": (
                        "Uppercase ticker symbol. "
                        "Examples: 'NVDA', 'SPY', 'QQQ', 'BTC-USD', '^VIX', 'GC=F', 'GLD'."
                    ),
                },
                "period": {
                    "type": "string",
                    "enum": ["1mo", "3mo", "1y", "3y", "5y"],
                    "description": "Look-back window for price history.",
                },
            },
            "required": ["ticker", "period"],
        },
    },
}

STORE_FINDING: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "store_finding",
        "description": (
            "Record a completed section of the research memo. "
            "Call this once per section as you complete it — do not batch. "
            "Returns the current memo state so you know which sections remain. "
            f"Required sections ({len(MEMO_SECTIONS)}): {', '.join(MEMO_SECTIONS)}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "enum": MEMO_SECTIONS,
                    "description": "The memo section to record.",
                },
                "content": {
                    "type": "string",
                    "description": (
                        "The content for this section. Guidance per section: "
                        + " | ".join(
                            f"{k}: {v}" for k, v in _SECTION_GUIDANCE.items()
                        )
                    ),
                },
            },
            "required": ["section", "content"],
        },
    },
}

FINALIZE_MEMO: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "finalize_memo",
        "description": (
            "Signal that the research memo is complete and ready to deliver. "
            f"Call this only after all {len(MEMO_SECTIONS)} sections have been written "
            "with store_finding. "
            "Returns an error listing missing sections if the memo is incomplete — "
            "do not finalize until all sections are stored."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

# ---------------------------------------------------------------------------
# Domain tool lists
# ---------------------------------------------------------------------------

# Financial research: the canonical tool set for v0.1.
# Future domains (e.g. POLICY_TOOLS, SCIENCE_TOOLS) follow the same pattern:
# replace GET_FINANCIAL_DATA with domain-specific data tools and pass the list
# to AgentLoop._run_agentic().
FINANCE_TOOLS: List[Dict[str, Any]] = [
    WEB_SEARCH,
    GET_FINANCIAL_DATA,
    STORE_FINDING,
    FINALIZE_MEMO,
]
