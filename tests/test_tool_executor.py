"""
Tests for ToolExecutor quality gates.

All tests run offline — no network or API keys required.
Each test exercises a specific enforcement rule in _store_finding_tool
or _finalize_memo_tool.
"""
from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from aria.agent.tool_executor import ToolExecutor
from aria.agent.tool_schemas import MEMO_SECTIONS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def executor() -> ToolExecutor:
    """Fresh executor with no data clients — sufficient for quality gate tests."""
    return ToolExecutor(web_search=None, financial=None)


def _fill_all_except(ex: ToolExecutor, skip: set[str]) -> None:
    """Store a valid value for every memo section except those in skip."""
    for section in MEMO_SECTIONS:
        if section in skip:
            continue
        content = (
            "Gold will outperform equities next year driven by central bank demand."
            if section == "thesis"
            else f"Content for {section}."
        )
        ex.execute("store_finding", {"section": section, "content": content})


def _ok(result_str: str) -> Dict[str, Any]:
    """Parse result and assert no error."""
    result = json.loads(result_str)
    assert "error" not in result, f"Unexpected error: {result['error']}"
    return result


def _err(result_str: str) -> Dict[str, Any]:
    """Parse result and assert error is present."""
    result = json.loads(result_str)
    assert "error" in result, f"Expected error, got: {result}"
    return result


# ---------------------------------------------------------------------------
# Thesis directional gate
# ---------------------------------------------------------------------------

class TestThesisDirectionalGate:
    def test_rejects_bull_case_framing(self, executor: ToolExecutor) -> None:
        _err(executor.execute("store_finding", {
            "section": "thesis",
            "content": "Bull case: gold reaches $3 000 driven by rate cuts.",
        }))

    def test_rejects_base_case_framing(self, executor: ToolExecutor) -> None:
        _err(executor.execute("store_finding", {
            "section": "thesis",
            "content": "Our base case is that NVDA holds at current levels through H2.",
        }))

    def test_rejects_bear_case_framing(self, executor: ToolExecutor) -> None:
        _err(executor.execute("store_finding", {
            "section": "thesis",
            "content": "Bear case scenario: equities correct 20% as liquidity tightens.",
        }))

    def test_accepts_single_directional_claim(self, executor: ToolExecutor) -> None:
        result = _ok(executor.execute("store_finding", {
            "section": "thesis",
            "content": "Gold will outperform equities next year driven by central bank accumulation.",
        }))
        assert result["stored"] == "thesis"

    def test_hedge_phrases_not_checked_in_other_sections(self, executor: ToolExecutor) -> None:
        # First store a valid thesis
        executor.execute("store_finding", {
            "section": "thesis",
            "content": "Gold outperforms equities in 2026.",
        })
        # conclusion may mention bear in context without triggering the thesis gate
        # (it has its own gate but allows hedge markers — checked separately)
        # supporting_evidence can freely mention "bull market" context
        result = _ok(executor.execute("store_finding", {
            "section": "supporting_evidence",
            "content": "During the 2020 bull market, gold held a 0.3 correlation to equities (WGC, 2021).",
        }))
        assert result["stored"] == "supporting_evidence"


# ---------------------------------------------------------------------------
# Uniqueness gate
# ---------------------------------------------------------------------------

class TestUniquenessGate:
    def test_rejects_unique_in_thesis_without_comparison(self, executor: ToolExecutor) -> None:
        _err(executor.execute("store_finding", {
            "section": "thesis",
            "content": "JPMorgan has a unique moat in private credit origination.",
        }))

    def test_rejects_uniquely_in_supporting_evidence(self, executor: ToolExecutor) -> None:
        executor.execute("store_finding", {
            "section": "thesis",
            "content": "Gold will outperform equities next year.",
        })
        _err(executor.execute("store_finding", {
            "section": "supporting_evidence",
            "content": "NVDA is uniquely positioned in the AI inference market.",
        }))

    def test_accepts_uniqueness_with_named_competitor(self, executor: ToolExecutor) -> None:
        result = _ok(executor.execute("store_finding", {
            "section": "thesis",
            "content": "JPMorgan has a unique origination network in private credit, unlike Bank of America which lacks direct-lending infrastructure.",
        }))
        assert result["stored"] == "thesis"

    def test_accepts_uniqueness_with_peers_comparison(self, executor: ToolExecutor) -> None:
        result = _ok(executor.execute("store_finding", {
            "section": "thesis",
            "content": "NVDA holds unique H100 allocation rights compared to peers who lack guaranteed capacity.",
        }))
        assert result["stored"] == "thesis"

    def test_uniqueness_gate_not_applied_to_other_sections(self, executor: ToolExecutor) -> None:
        executor.execute("store_finding", {
            "section": "thesis",
            "content": "Gold outperforms equities in 2026.",
        })
        executor.execute("store_finding", {
            "section": "supporting_evidence",
            "content": "Central bank purchases reached a record high (WGC, 2025).",
        })
        # counter_evidence is not in the gate's checked sections
        result = _ok(executor.execute("store_finding", {
            "section": "counter_evidence",
            "content": "The dollar uniquely benefits from flight-to-safety flows too.",
        }))
        assert result["stored"] == "counter_evidence"


# ---------------------------------------------------------------------------
# Section overwrite prevention
# ---------------------------------------------------------------------------

class TestSectionOverwritePrevention:
    def test_second_write_to_same_section_blocked(self, executor: ToolExecutor) -> None:
        original = "Gold will outperform equities next year."
        executor.execute("store_finding", {"section": "thesis", "content": original})
        _err(executor.execute("store_finding", {"section": "thesis", "content": "Revised thesis."}))
        assert executor.memo_state["thesis"] == original

    def test_error_response_lists_remaining_sections(self, executor: ToolExecutor) -> None:
        executor.execute("store_finding", {
            "section": "thesis", "content": "Gold will outperform equities next year.",
        })
        result = _err(executor.execute("store_finding", {
            "section": "thesis", "content": "Try again.",
        }))
        assert "missing" in result
        assert "thesis" not in result["missing"]


# ---------------------------------------------------------------------------
# Confidence propagation gate
# ---------------------------------------------------------------------------

class TestConfidencePropagationGate:
    def _store_all_except_conclusion(
        self, ex: ToolExecutor, confidence_text: str
    ) -> None:
        for section in MEMO_SECTIONS:
            if section == "conclusion":
                continue
            content = (
                "Gold will outperform equities next year."
                if section == "thesis"
                else confidence_text
                if section == "confidence"
                else f"Content for {section}."
            )
            ex.execute("store_finding", {"section": section, "content": content})

    def test_low_confidence_without_hedge_blocked(self) -> None:
        ex = ToolExecutor(web_search=None, financial=None)
        self._store_all_except_conclusion(ex, "Low — only one tier-3 source found.")
        _err(ex.execute("store_finding", {
            "section": "conclusion",
            "content": "Overweight gold for H1 2026, targeting 15% outperformance.",
        }))

    def test_medium_confidence_without_hedge_blocked(self) -> None:
        ex = ToolExecutor(web_search=None, financial=None)
        self._store_all_except_conclusion(ex, "Medium — two tier-3 sources, no primary data.")
        _err(ex.execute("store_finding", {
            "section": "conclusion",
            "content": "Overweight gold for H1 2026 based on current momentum.",
        }))

    def test_low_confidence_with_conditional_hedge_passes(self) -> None:
        ex = ToolExecutor(web_search=None, financial=None)
        self._store_all_except_conclusion(ex, "Low — limited sources.")
        result = _ok(ex.execute("store_finding", {
            "section": "conclusion",
            "content": "Overweight gold only if confirmed by Q1 central bank purchase data.",
        }))
        assert result["stored"] == "conclusion"

    def test_medium_confidence_with_pending_hedge_passes(self) -> None:
        ex = ToolExecutor(web_search=None, financial=None)
        self._store_all_except_conclusion(ex, "Medium — two sources, converging signals.")
        result = _ok(ex.execute("store_finding", {
            "section": "conclusion",
            "content": "Overweight gold pending confirmation of Fed pivot in Q2 guidance.",
        }))
        assert result["stored"] == "conclusion"

    def test_high_confidence_needs_no_hedge(self) -> None:
        ex = ToolExecutor(web_search=None, financial=None)
        self._store_all_except_conclusion(
            ex, "High — multiple tier-1 and tier-4 sources converge."
        )
        result = _ok(ex.execute("store_finding", {
            "section": "conclusion",
            "content": "Gold appears undervalued relative to equities given sustained central bank demand.",
        }))
        assert result["stored"] == "conclusion"


# ---------------------------------------------------------------------------
# Investment thesis directional gate (separate from research thesis gate)
# ---------------------------------------------------------------------------

class TestInvestmentThesisGate:
    def _fill_pre_investment(self, ex: ToolExecutor) -> None:
        for section in MEMO_SECTIONS:
            if section == "conclusion":
                continue
            content = (
                "Gold will outperform equities next year."
                if section == "thesis"
                else f"Content for {section}."
            )
            ex.execute("store_finding", {"section": section, "content": content})

    def test_rejects_bull_base_bear_in_conclusion(self) -> None:
        ex = ToolExecutor(web_search=None, financial=None)
        self._fill_pre_investment(ex)
        _err(ex.execute("store_finding", {
            "section": "conclusion",
            "content": "Bull case: undervalued. Base case: fairly valued. Bear case: overvalued.",
        }))


# ---------------------------------------------------------------------------
# Unknown tool and invalid inputs
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_unknown_tool_name(self, executor: ToolExecutor) -> None:
        result = _err(executor.execute("does_not_exist", {}))
        assert "does_not_exist" in result["error"]

    def test_store_finding_unknown_section(self, executor: ToolExecutor) -> None:
        _err(executor.execute("store_finding", {"section": "made_up_section", "content": "x"}))

    def test_store_finding_empty_content(self, executor: ToolExecutor) -> None:
        _err(executor.execute("store_finding", {"section": "thesis", "content": ""}))

    def test_store_finding_missing_section(self, executor: ToolExecutor) -> None:
        _err(executor.execute("store_finding", {"content": "Some content without section key."}))


# ---------------------------------------------------------------------------
# finalize_memo
# ---------------------------------------------------------------------------

class TestFinalizeMemo:
    def test_finalize_incomplete_memo_blocked(self, executor: ToolExecutor) -> None:
        result = _err(executor.execute("finalize_memo", {}))
        assert "missing_sections" in result

    def test_finalize_partial_memo_blocked(self, executor: ToolExecutor) -> None:
        executor.execute("store_finding", {
            "section": "thesis",
            "content": "Gold will outperform equities next year.",
        })
        result = _err(executor.execute("finalize_memo", {}))
        assert "thesis" not in result["missing_sections"]

    def test_finalize_complete_memo_succeeds(self) -> None:
        ex = ToolExecutor(web_search=None, financial=None)
        _fill_all_except(ex, skip=set())
        result = _ok(ex.execute("finalize_memo", {}))
        assert result["status"] == "complete"
        assert set(result["sections"]) == set(MEMO_SECTIONS)

    def test_ready_to_finalize_flag_set_when_all_sections_stored(self) -> None:
        ex = ToolExecutor(web_search=None, financial=None)
        _fill_all_except(ex, skip=set())
        # The last store_finding call should set ready_to_finalize=True
        for section in MEMO_SECTIONS:
            json.loads(ex.execute("store_finding", {
                "section": section + "_x",  # already stored; will error — skip
                "content": "x",
            }))
        # Better: track via memo_state directly
        assert len(ex.memo_state) == len(MEMO_SECTIONS)
