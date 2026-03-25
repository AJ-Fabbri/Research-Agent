"""
Model- and prompt-related utilities for ARIA.

Exposes ModelRouter and TaskType for model selection.
System prompt construction lives in aria.agent.core (_build_analysis_system_prompt),
co-located with the agentic loop that uses it.
"""

from .router import ModelRouter, TaskType

__all__ = ["ModelRouter", "TaskType"]
