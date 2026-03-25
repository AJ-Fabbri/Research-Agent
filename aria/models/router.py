from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:  # pragma: no cover - optional dependency
    ChatAnthropic = None  # type: ignore

from aria.config import AriaConfig


class TaskType(Enum):
    """High-level task categories for routing decisions."""

    RESEARCH_MEMO = auto()
    DIRECT_ANSWER = auto()
    DOCUMENT_ANALYSIS = auto()
    SENSITIVE_DATA = auto()
    QUICK_FACT = auto()
    CODE_GENERATION = auto()


@dataclass
class RoutedModel:
    """Container for a selected model and metadata."""

    model: BaseChatModel
    is_local: bool
    name: str


class ModelRouter:
    """
    Choose between API and local models based on config and task type.

    This is a thin abstraction over LangChain chat models; it does not
    implement the agent loop itself.
    """

    def __init__(self, config: AriaConfig) -> None:
        self._config = config

    @property
    def config(self) -> AriaConfig:
        return self._config

    def select_model(self, task_type: TaskType) -> RoutedModel:
        """
        Select an appropriate chat model given the task type and configuration.
        """
        routing = self._config.model.routing
        privacy_mode = self._config.model.privacy_mode
        local_base_url = self._config.model.local_base_url

        # Explicit routing overrides always take priority.
        if routing == "api":
            return self._api_model()
        if routing == "local":
            return self._local_model()

        # Privacy mode forces local regardless of task type.
        if privacy_mode:
            return self._local_model()

        # Auto-routing: route by task type.
        if task_type in (TaskType.RESEARCH_MEMO, TaskType.CODE_GENERATION, TaskType.DOCUMENT_ANALYSIS):
            return self._api_model()
        if task_type is TaskType.SENSITIVE_DATA:
            return self._local_model()
        # QUICK_FACT / DIRECT_ANSWER: prefer local if a server is configured.
        if local_base_url:
            try:
                return self._local_model()
            except RuntimeError:
                pass
        return self._api_model()

    def _api_model(self) -> RoutedModel:
        model_name = self._config.model.default

        if "claude" in model_name and ChatAnthropic is not None:
            model: BaseChatModel = ChatAnthropic(model=model_name)
            return RoutedModel(model=model, is_local=False, name=model_name)

        model = ChatOpenAI(model=model_name)
        return RoutedModel(model=model, is_local=False, name=model_name)

    def _local_model(self) -> RoutedModel:
        """
        Construct a local chat model via an OpenAI-compatible endpoint.

        Attempts to resolve the actual loaded model name from the server's
        /v1/models endpoint (works with LM Studio and Ollama). Falls back
        to the name in config if the server is unreachable.
        """
        config_name = self._config.model.local or "local"
        base_url = self._config.model.local_base_url
        api_key = self._config.model.local_api_key or "lm-studio"

        # Prefer the name of the model actually loaded on the server.
        # This populates the "Model:" field in memos correctly without
        # requiring users to keep the config in sync with LM Studio.
        actual_name = _query_loaded_model(base_url, api_key) or config_name

        kwargs: Dict[str, Any] = {"model": actual_name}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        model = ChatOpenAI(**kwargs)
        return RoutedModel(model=model, is_local=True, name=actual_name)


def _query_loaded_model(base_url: Optional[str], api_key: str) -> Optional[str]:
    """
    Fetch the first model ID from a local server's GET /v1/models endpoint.

    Returns None on any failure — timeouts, connection errors, unexpected
    response shapes — so callers can fall back gracefully.
    """
    if not base_url:
        return None
    try:
        url = base_url.rstrip("/") + "/models"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {api_key}"}
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
        models = data.get("data", [])
        if models:
            return models[0].get("id") or None
    except (OSError, json.JSONDecodeError, KeyError, IndexError):
        pass
    return None
