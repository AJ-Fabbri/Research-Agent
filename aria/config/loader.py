from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ModelConfig:
    default: str = "claude-sonnet-4-6"
    local: Optional[str] = None
    # Local OpenAI-compatible server (e.g. LM Studio: http://localhost:1234/v1)
    local_base_url: Optional[str] = None
    local_api_key: Optional[str] = None  # often "lm-studio" or any placeholder
    privacy_mode: bool = False
    routing: str = "auto"  # auto | api | local


@dataclass
class ModeConfig:
    default: str = "checkpoint"  # autonomous | checkpoint


@dataclass
class FinancialApisConfig:
    yfinance: bool = True
    alpha_vantage_key: Optional[str] = None
    bloomberg_key: Optional[str] = None


@dataclass
class DocumentsConfig:
    ingest_path: str = "./documents"
    vector_store: str = "./chroma_db"


@dataclass
class RepoConfig:
    root: str = "./"


@dataclass
class DataSourcesConfig:
    web_search: bool = True
    financial_apis: FinancialApisConfig = field(default_factory=FinancialApisConfig)
    documents: DocumentsConfig = field(default_factory=DocumentsConfig)
    repo: RepoConfig = field(default_factory=RepoConfig)


@dataclass
class OutputConfig:
    default_format: str = "auto"  # auto | memo | direct | matrix
    save_path: str = "./outputs"
    export_docx: bool = True


@dataclass
class AgentConfig:
    max_steps: int = 20
    log_reasoning_trace: bool = True
    db_path: str = "aria_research.db"


@dataclass
class MonitorConfig:
    enabled: bool = False
    interval_hours: int = 168  # weekly
    discord_webhook: Optional[str] = None
    notify_on: str = "challenged"  # challenged | all


@dataclass
class AriaConfig:
    model: ModelConfig
    mode: ModeConfig
    data_sources: DataSourcesConfig
    output: OutputConfig
    agent: AgentConfig
    monitor: MonitorConfig = field(default_factory=MonitorConfig)


def _expand_env(obj: Any) -> Any:
    """
    Recursively expand environment variables in strings within a nested structure.
    """
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


def _dict_to_config(data: Dict[str, Any]) -> AriaConfig:
    model_cfg = data.get("model", {}) or {}
    mode_cfg = data.get("mode", {}) or {}
    ds_cfg = data.get("data_sources", {}) or {}
    out_cfg = data.get("output", {}) or {}
    agent_cfg = data.get("agent", {}) or {}
    monitor_cfg = data.get("monitor", {}) or {}

    financial_cfg = (ds_cfg.get("financial_apis") or {}) if ds_cfg else {}
    documents_cfg = (ds_cfg.get("documents") or {}) if ds_cfg else {}
    repo_cfg = (ds_cfg.get("repo") or {}) if ds_cfg else {}

    return AriaConfig(
        model=ModelConfig(**model_cfg),
        mode=ModeConfig(**mode_cfg),
        data_sources=DataSourcesConfig(
            web_search=ds_cfg.get("web_search", True),
            financial_apis=FinancialApisConfig(**financial_cfg),
            documents=DocumentsConfig(**documents_cfg),
            repo=RepoConfig(**repo_cfg),
        ),
        output=OutputConfig(**out_cfg),
        agent=AgentConfig(**agent_cfg),
        monitor=MonitorConfig(**monitor_cfg),
    )


def load_config(path: Optional[str] = None) -> AriaConfig:
    """
    Load the ARIA configuration from a YAML file, expanding environment variables.

    If `path` is not provided, defaults to `aria_config.yaml` in the current
    working directory.
    """
    config_path = Path(path) if path is not None else Path("aria_config.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    expanded = _expand_env(raw)
    if not isinstance(expanded, dict):
        raise ValueError("Top-level config must be a mapping.")

    return _dict_to_config(expanded)

