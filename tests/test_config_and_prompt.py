from __future__ import annotations

from aria.agent.core import _build_analysis_system_prompt
from aria.config import AriaConfig, load_config


def test_load_config_defaults() -> None:
    config = load_config("aria_config.yaml")
    assert isinstance(config, AriaConfig)
    assert config.model.default == "claude-sonnet-4-6"
    assert config.mode.default in {"checkpoint", "autonomous"}


def test_analysis_prompt_contains_required_instructions() -> None:
    prompt = _build_analysis_system_prompt()
    assert "finalize_memo" in prompt
    assert "store_finding" in prompt


def test_analysis_prompt_injects_date() -> None:
    prompt = _build_analysis_system_prompt()
    assert "Today's date:" in prompt
