"""
Offline tests for AgentLoop.

Uses scripted fake models so no network calls or API keys are required. Covers:
  - Normal agentic path: model calls store_finding for each section then
    finalize_memo → loop renders a complete, structured ResearchMemo.
  - Fallback path: model produces text without calling any tools → loop
    returns raw text with fallback_used=True.
  - Metadata contract: all expected keys present in AgentResult.metadata.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from aria.agent import AgentLoop, AgentMode, OutputFormat
from aria.agent.core import AgentDependencies
from aria.agent.tool_schemas import MEMO_SECTIONS
from aria.config import load_config
from aria.models.router import RoutedModel


# ---------------------------------------------------------------------------
# Fake model infrastructure
# ---------------------------------------------------------------------------

def _tool_call(name: str, args: Dict[str, Any], call_id: str) -> Dict[str, Any]:
    return {"name": name, "args": args, "id": call_id}


class FakeResponse:
    """Simulates an AIMessage with optional tool_calls."""

    def __init__(
        self, content: str = "", tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class ScriptedModel:
    """
    Returns a pre-defined sequence of FakeResponses on successive invoke() calls.
    bind_tools() is a no-op — it returns self so the agentic loop works unchanged.
    """

    def __init__(self, responses: List[FakeResponse]) -> None:
        self._responses = iter(responses)

    def bind_tools(self, tools: Any) -> "ScriptedModel":
        return self

    def invoke(self, messages: List[Any]) -> FakeResponse:
        return next(self._responses)


class FallbackModel:
    """
    Simulates a local model that doesn't support function calling —
    always produces plain text without any tool_calls.
    """

    def bind_tools(self, tools: Any) -> "FallbackModel":
        return self

    def invoke(self, messages: List[Any]) -> FakeResponse:
        return FakeResponse(content="Raw analysis text — no tool calls made.")


class FakeRouter:
    def __init__(self, model: Any, name: str = "fake-model") -> None:
        self._model = model
        self._name = name

    def select_model(self, task_type: Any) -> RoutedModel:
        return RoutedModel(model=self._model, is_local=True, name=self._name)


def _make_deps(model: Any, name: str = "fake-model") -> AgentDependencies:
    config = load_config("aria_config.yaml")
    return AgentDependencies(
        config=config,
        router=FakeRouter(model=model, name=name),  # type: ignore[arg-type]
        web_search=None,  # type: ignore[arg-type]
        financial=None,  # type: ignore[arg-type]
        documents=None,  # type: ignore[arg-type]
        repo=None,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_agentic_loop_builds_structured_memo() -> None:
    """
    Happy path: model calls store_finding for each memo section then calls
    finalize_memo. Loop should render a complete ResearchMemo and return
    fallback_used=False.
    """
    # One turn per memo section (store_finding), then one turn for finalize_memo.
    store_turns = [
        FakeResponse(tool_calls=[
            _tool_call(
                "store_finding",
                {"section": section, "content": f"Content for {section}."},
                f"id_{i}",
            )
        ])
        for i, section in enumerate(MEMO_SECTIONS)
    ]
    finalize_turn = FakeResponse(
        tool_calls=[_tool_call("finalize_memo", {}, "id_final")]
    )

    model = ScriptedModel(store_turns + [finalize_turn])
    agent = AgentLoop(_make_deps(model))

    result = agent.run(
        "Is NVDA overvalued relative to the semiconductor sector?",
        mode=AgentMode.AUTONOMOUS,
    )

    # Every section's content must appear in the rendered markdown output.
    for section in MEMO_SECTIONS:
        assert f"Content for {section}." in result.content, (
            f"Expected content for section '{section}' to appear in output."
        )

    assert result.metadata["model_name"] == "fake-model"
    assert result.metadata["fallback_used"] is False
    assert result.metadata["mode"] == "autonomous"
    # At minimum: one turn per section + one finalize turn.
    assert result.metadata["steps_taken"] >= len(MEMO_SECTIONS) + 1


def test_fallback_path_returns_model_text() -> None:
    """
    When the model produces text without calling any tools (e.g. a local model
    that doesn't support function calling), the loop should fall back gracefully
    and return that text with fallback_used=True.
    """
    agent = AgentLoop(_make_deps(FallbackModel()))

    result = agent.run(
        "Test thesis about European equities vs US equities.",
        mode=AgentMode.AUTONOMOUS,
        output_format=OutputFormat.DIRECT,
    )

    assert "Raw analysis text" in result.content
    assert result.metadata["model_name"] == "fake-model"
    assert result.metadata["fallback_used"] is True


def test_metadata_keys_always_present() -> None:
    """AgentResult.metadata must always expose the full set of expected keys."""
    agent = AgentLoop(_make_deps(FallbackModel()))
    result = agent.run("Quick check.", mode=AgentMode.AUTONOMOUS)

    expected_keys = (
        "model_name",
        "is_local",
        "mode",
        "steps_taken",
        "fallback_used",
        "web_search_sources",
    )
    for key in expected_keys:
        assert key in result.metadata, f"Missing metadata key: {key!r}"
