"""
Core ARIA agent loop.

The AgentLoop coordinates:
- system prompt construction,
- model routing,
- tool access,
- and high-level reasoning steps (thesis, adversarial pass, segmentation, etc.).
"""

from .core import AgentLoop, AgentMode, OutputFormat

__all__ = ["AgentLoop", "AgentMode", "OutputFormat"]

