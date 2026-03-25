"""
Output formatters for ARIA research memos.

The ResearchMemo dataclass is the canonical output of a completed agentic run.
format_memo_markdown() renders it to a professional markdown document suitable
for display, saving, or sharing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional


@dataclass
class ResearchMemo:
    # The eight analytical sections written by the model via store_finding().
    thesis: str
    supporting_evidence: str
    counter_evidence: str
    segment_behavior: str
    failure_conditions: str
    confidence: str         # "High / Medium / Low — one-sentence reason"
    baselines: str          # explicit comparison benchmarks
    conclusion: str         # valuation assessment: direction, horizon, triggers

    # Display context — populated by the agent loop, not the model.
    sources: List[Dict[str, str]] = field(default_factory=list)
    partial: bool = False  # True when the memo is incomplete (fallback path)


def format_memo_markdown(
    memo: ResearchMemo,
    *,
    query: str = "",
    session_id: Optional[str] = None,
    model_name: str = "",
) -> str:
    """
    Render a ResearchMemo to a professional markdown document.

    Optional keyword args add a header line with context that is available
    from the agent loop but not part of the analytical content:
      query       — the original research question
      session_id  — UUID for outcome tracking (aria outcome <id> --result ...)
      model_name  — model used for the analysis
    """
    lines: List[str] = []

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    lines.append("# ARIA Research Memo")
    lines.append("")

    if query:
        lines.append(f"**Query:** {query}")
        lines.append("")

    meta_parts: List[str] = [f"**Date:** {date.today().strftime('%B %d, %Y')}"]
    if model_name:
        meta_parts.append(f"**Model:** {model_name}")
    if session_id:
        meta_parts.append(f"**Session:** `{session_id[:8]}…`")
    lines.append(" | ".join(meta_parts))
    lines.append("")
    lines.append("---")
    lines.append("")

    # ------------------------------------------------------------------
    # Thesis block — compact, high-signal executive view
    # ------------------------------------------------------------------
    lines.append("## Thesis")
    lines.append("")
    lines.append(memo.thesis)
    lines.append("")
    lines.append(f"**Confidence:** {memo.confidence}")
    lines.append("")
    lines.append(f"**Baselines:** {memo.baselines}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ------------------------------------------------------------------
    # Analytical sections
    # ------------------------------------------------------------------
    _section(lines, "Supporting Evidence", memo.supporting_evidence)
    _section(lines, "Counter Evidence", memo.counter_evidence)
    _section(lines, "Segment Behavior", memo.segment_behavior)
    _section(lines, "Failure Conditions", memo.failure_conditions)

    # ------------------------------------------------------------------
    # Conclusion — the actionable landing point of the analysis
    # ------------------------------------------------------------------
    lines.append("## Conclusion")
    lines.append("")
    lines.append(memo.conclusion)
    lines.append("")
    lines.append("---")
    lines.append("")

    # ------------------------------------------------------------------
    # Sources — grouped by search purpose
    # ------------------------------------------------------------------
    if memo.sources:
        lines.append("## Sources")
        lines.append("")
        _sources_group(lines, "Background", memo.sources, "background")
        _sources_group(lines, "Supporting thesis", memo.sources, "pro_thesis")
        _sources_group(lines, "Counter-thesis", memo.sources, "counter_thesis")
        lines.append("---")
        lines.append("")

    # ------------------------------------------------------------------
    # Partial-memo warning (fallback / interrupted run)
    # ------------------------------------------------------------------
    if memo.partial:
        lines.append(
            "*⚠ Partial memo — the analysis was interrupted before all sections "
            "were completed. Treat findings as preliminary.*"
        )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _section(lines: List[str], title: str, content: str) -> None:
    lines.append(f"## {title}")
    lines.append("")
    lines.append(content)
    lines.append("")
    lines.append("---")
    lines.append("")


def _sources_group(
    lines: List[str],
    label: str,
    sources: List[Dict[str, str]],
    purpose: str,
) -> None:
    group = [s for s in sources if s.get("purpose") == purpose]
    if not group:
        return
    lines.append(f"**{label}**")
    seen: set[str] = set()
    for s in group:
        url = s.get("url", "")
        if url in seen:
            continue
        seen.add(url)
        title = s.get("title") or url
        lines.append(f"- [{title}]({url})")
    lines.append("")
